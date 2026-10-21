"""AlphaEarth center-pixel MLP baseline for 11-class basal-area composition."""

import argparse
import copy
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.collate import nfi_collate_fn
from training.data.cross_validation import configure_outer_fold, cv_enabled, outer_fold
from training.data.dataset import NFISiteDataset, dequantize_alphaearth_patch
from src.data.label_schema import SPECIES_NUM_CLASSES
from src.models.baselines.alphaearth import AlphaEarthCompositionMLP
from training.data.samplers import build_sampler
from training.data.utils import build_transforms
from evaluation.artifacts import save_evaluation_artifacts
from src.utils.config import load_training_config
from training.losses import composition_l1_loss, soft_target_kl_loss
from src.utils.seed import seed as seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/baselines/alphaearth_mlp_baseline/s42")
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--outer-fold", type=int, required=True)
    return parser.parse_args()


def configure_dataset(args):
    cfg = load_training_config(
        config=args.config, outer_fold=args.outer_fold, overrides=args.overrides,
    )
    configure_outer_fold(cfg, args.outer_fold)
    for modality in ("rgb", "lidar", "chelsa", "dem"):
        cfg.modalities[modality].enabled = False
    cfg.modalities.sentinel2.enabled = True
    cfg.augmentation.geometric = False
    OmegaConf.update(cfg, "model.sentinel2.s2_mode", "satellite_embedding_patch", force_add=True)
    return cfg


def preflight(cfg):
    index = pd.read_csv(cfg.path.index)
    failures = []
    for site_id in index.site_id.astype(str):
        with h5py.File(Path(cfg.path.hdf5) / f"{site_id}.h5", "r") as handle:
            vector = dequantize_alphaearth_patch(
                handle["satellite_embedding_patch/vectors"][()]
            )
            if vector.shape != (9, 64) or not np.isfinite(vector).all():
                failures.append(f"{site_id}:{vector.shape}")
    if failures:
        raise RuntimeError(f"Invalid AEF patches at {len(failures)} sites: {failures[:5]}")
    print(f"AEF preflight passed: {len(index)} center vectors available", flush=True)


def make_loader(cfg, split, args, shuffle=False, sampler=None, dataset=None):
    dataset = dataset if dataset is not None else NFISiteDataset(
        cfg, split, build_transforms(cfg, split)
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=nfi_collate_fn,
    ), dataset


def center_embedding(batch, device):
    embedding = batch["sentinel2"]["embedding"]
    if embedding.ndim != 3 or embedding.shape[1:] != (9, 64):
        raise ValueError(f"Expected (B,9,64) AEF patches, got {tuple(embedding.shape)}.")
    finite = embedding[torch.isfinite(embedding)]
    if finite.numel() and finite.abs().max().item() > 1.0:
        raise ValueError("AEF patch was not dequantized to the signed unit range.")
    return embedding[:, 4, :].to(device, non_blocking=True)


def loss_value(outputs, batch, device):
    species_target = batch["species_target"].to(device)
    leaf_target = batch["leaf_target"].to(device)
    species_kl = soft_target_kl_loss(outputs["species_logits"], species_target)
    species_l1 = composition_l1_loss(outputs["species_prob"], species_target)
    leaf_kl = soft_target_kl_loss(outputs["leaf_logits"], leaf_target)
    leaf_l1 = composition_l1_loss(outputs["leaf_prob"], leaf_target)
    return 0.2 * species_kl + species_l1 + 0.2 * leaf_kl + leaf_l1


@torch.no_grad()
def predict(model, loader, device, split):
    model.eval()
    payload = {"split": split, "site_id": [], "species_target": [], "species_prob": [],
               "leaf_target": [], "leaf_prob": []}
    for batch in loader:
        outputs = model(center_embedding(batch, device))
        payload["site_id"].extend(batch["site_id"])
        payload["species_target"].append(batch["species_target"].numpy())
        payload["species_prob"].append(outputs["species_prob"].cpu().numpy())
        payload["leaf_target"].append(batch["leaf_target"].numpy())
        payload["leaf_prob"].append(outputs["leaf_prob"].cpu().numpy())
    for key in ("species_target", "species_prob", "leaf_target", "leaf_prob"):
        payload[key] = np.concatenate(payload[key], axis=0)
    return payload


def main():
    args = parse_args()
    seed_everything(args.seed)
    cfg = configure_dataset(args)
    if cv_enabled(cfg) and args.epochs != 80:
        raise ValueError("Spatial CV baselines must train for exactly 80 epochs.")
    if not args.skip_preflight:
        preflight(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("AlphaEarth MLP training requires CUDA.")
    device = torch.device("cuda")
    train_loader, train_dataset = make_loader(cfg, "train", args)
    sampler = build_sampler(train_dataset, cfg, is_train=True, distributed=False, seed=args.seed)
    train_loader, _ = make_loader(cfg, "train", args, sampler=sampler, dataset=train_dataset)
    test_loader, _ = make_loader(cfg, "test", args)

    model = AlphaEarthCompositionMLP(dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    for epoch in range(1, args.epochs + 1):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        for batch in train_loader:
            outputs = model(center_embedding(batch, device))
            loss = loss_value(outputs, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), model.parameters()):
                    target.mul_(args.ema_decay).add_(source.detach(), alpha=1 - args.ema_decay)
                for target, source in zip(ema.buffers(), model.buffers()):
                    target.copy_(source)
            total += loss.item()
        scheduler.step()
        print(f"epoch={epoch:03d} loss={total/max(len(train_loader),1):.4f}", flush=True)

    final_state = copy.deepcopy(ema.state_dict())
    model.load_state_dict(final_state)
    output_dir = Path(args.output_dir)
    payload = predict(model, test_loader, device, "test")
    if cv_enabled(cfg):
        payload["outer_fold"] = np.full(len(payload["site_id"]), outer_fold(cfg), dtype=np.int64)
    save_evaluation_artifacts([payload], cfg, output_dir, source="alphaearth_mlp")
    checkpoint_path = Path(args.checkpoint_out) if args.checkpoint_out else output_dir / "last.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": final_state,
        "epoch": args.epochs,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "outer_fold": outer_fold(cfg) if cv_enabled(cfg) else None,
    }, checkpoint_path)
    print(f"wrote baseline artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
