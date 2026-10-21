"""ImageNet ResNet-50 RGB baseline for 11-class basal-area composition."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.collate import nfi_collate_fn
from training.data.cross_validation import configure_outer_fold, cv_enabled, outer_fold
from training.data.dataset import NFISiteDataset
from src.data.label_schema import SPECIES_NUM_CLASSES
from src.models.baselines.resnet import ResNet50Composition
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/baselines/resnet50_baseline/s42")
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--outer-fold", type=int, required=True)
    return parser.parse_args()


def configure_dataset(args):
    cfg = load_training_config(
        config=args.config, outer_fold=args.outer_fold, overrides=args.overrides,
    )
    configure_outer_fold(cfg, args.outer_fold)
    cfg.modalities.rgb.enabled = True
    cfg.modalities.rgb.multiseason = False
    for modality in ("lidar", "sentinel2", "chelsa", "dem"):
        cfg.modalities[modality].enabled = False
    cfg.augmentation.geometric = True
    return cfg


def make_loader(cfg, split, args, sampler=None, dataset=None):
    dataset = dataset if dataset is not None else NFISiteDataset(
        cfg, split, transform=build_transforms(cfg, split)
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=nfi_collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    ), dataset


def compute_loss(outputs, batch, device):
    species_target = batch["species_target"].to(device, non_blocking=True)
    leaf_target = batch["leaf_target"].to(device, non_blocking=True)
    stand_target = batch["stand_type_id"].to(device, non_blocking=True).long()
    return (
        0.2 * soft_target_kl_loss(outputs["species_logits"], species_target)
        + composition_l1_loss(outputs["species_prob"], species_target)
        + 0.2 * soft_target_kl_loss(outputs["leaf_logits"], leaf_target)
        + composition_l1_loss(outputs["leaf_prob"], leaf_target)
        + F.cross_entropy(outputs["stand_logits"], stand_target)
    )


@torch.no_grad()
def predict(model, loader, device, split):
    model.eval()
    payload = {
        "split": split,
        "site_id": [],
        "stand_target": [],
        "stand_prob": [],
        "leaf_target": [],
        "leaf_prob": [],
        "species_target": [],
        "species_prob": [],
        "is_pure": [],
        "dominant_species_class_id": [],
        "dominant_species_proportion": [],
        "nfi_record_count": [],
        "total_expanded_basal_area_m2_ha": [],
    }
    for batch in loader:
        batch["rgb"]["data"] = batch["rgb"]["data"].to(device, non_blocking=True)
        outputs = model(batch)
        payload["site_id"].extend(batch["site_id"])
        for key, value in (
            ("stand_target", batch["stand_type_id"]),
            ("stand_prob", outputs["stand_prob"]),
            ("leaf_target", batch["leaf_target"]),
            ("leaf_prob", outputs["leaf_prob"]),
            ("species_target", batch["species_target"]),
            ("species_prob", outputs["species_prob"]),
            ("is_pure", batch["is_pure"]),
            ("dominant_species_class_id", batch["dominant_species_class_id"]),
            ("dominant_species_proportion", batch["dominant_species_proportion"]),
            ("nfi_record_count", batch["nfi_record_count"]),
            ("total_expanded_basal_area_m2_ha", batch["total_expanded_basal_area_m2_ha"]),
        ):
            payload[key].append(value.detach().cpu().numpy())
    for key in payload.keys() - {"split", "site_id"}:
        payload[key] = np.concatenate(payload[key], axis=0)
    return payload


def update_ema(ema, model, decay):
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.mul_(decay).add_(source.detach(), alpha=1.0 - decay)
        for target, source in zip(ema.buffers(), model.buffers()):
            target.copy_(source)


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("ResNet-50 baseline training requires CUDA.")
    device = torch.device("cuda")
    cfg = configure_dataset(args)
    if cv_enabled(cfg) and args.epochs != 80:
        raise ValueError("Spatial CV baselines must train for exactly 80 epochs.")

    train_loader, train_dataset = make_loader(cfg, "train", args)
    sampler = build_sampler(train_dataset, cfg, is_train=True, distributed=False, seed=args.seed)
    train_loader, _ = make_loader(cfg, "train", args, sampler=sampler, dataset=train_dataset)
    test_loader, test_dataset = make_loader(cfg, "test", args)

    model = ResNet50Composition(
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    print(
        f"train={len(train_dataset)} test={len(test_dataset)} classes={SPECIES_NUM_CLASSES} "
        f"pretrained={not args.no_pretrained} freeze_backbone={args.freeze_backbone}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        for batch in train_loader:
            batch["rgb"]["data"] = batch["rgb"]["data"].to(device, non_blocking=True)
            outputs = model(batch)
            loss = compute_loss(outputs, batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema, model, args.ema_decay)
            total += loss.item()
        scheduler.step()
        print(
            f"epoch={epoch:03d} loss={total/max(len(train_loader), 1):.4f}",
            flush=True,
        )

    final_state = copy.deepcopy(ema.state_dict())
    model.load_state_dict(final_state)
    output_dir = Path(args.output_dir)
    payload = predict(model, test_loader, device, "test")
    if cv_enabled(cfg):
        payload["outer_fold"] = np.full(len(payload["site_id"]), outer_fold(cfg), dtype=np.int64)
    save_evaluation_artifacts(
        [payload], cfg, output_dir, source="resnet50"
    )
    checkpoint_path = Path(args.checkpoint_out) if args.checkpoint_out else output_dir / "last.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": final_state,
            "epoch": args.epochs,
            "species_classes": SPECIES_NUM_CLASSES,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "outer_fold": outer_fold(cfg) if cv_enabled(cfg) else None,
        },
        checkpoint_path,
    )
    metadata = {
        "model": "resnet50_imagenet_linear",
        "pretrained": not args.no_pretrained,
        "freeze_backbone": args.freeze_backbone,
        "epochs": args.epochs,
        "outer_fold": outer_fold(cfg) if cv_enabled(cfg) else None,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote baseline artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
