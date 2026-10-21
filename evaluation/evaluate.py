#!/usr/bin/env python3
"""Evaluate an 11-class basal-area model and save reusable result artifacts."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.collate import nfi_collate_fn
from training.data.cross_validation import configure_outer_fold, cv_enabled, outer_fold
from training.data.dataset import NFISiteDataset
from training.data.utils import build_transforms
from evaluation.artifacts import save_evaluation_artifacts
from src.utils.config import config_from_args, complete_config, apply_overrides


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-fold", type=int, default=None)
    parser.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(args, checkpoint):
    if isinstance(checkpoint, dict) and "cfg" in checkpoint:
        saved = OmegaConf.create(checkpoint["cfg"])
        cfg = complete_config(saved)
        if "cross_validation" not in saved:
            cfg.cross_validation.enabled = False
        return apply_overrides(cfg, args.overrides)
    return config_from_args(args)

def get_device(args, cfg):
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() and str(cfg.device) == "cuda" else "cpu")


def output_dir_for(args):
    if args.output_dir:
        return Path(args.output_dir)
    checkpoint = Path(args.checkpoint)
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    return run_dir / "evaluation"


def strip_module_prefix(state_dict):
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_model(cfg, checkpoint, device):
    from src.models import build_model

    # The checkpoint includes the complete backbone; no separate pretrained file
    # is needed to evaluate a fitted model.
    model_cfg = OmegaConf.merge(cfg, {"path": {"rgb_pretrained_weights": None}})
    model = build_model(model_cfg)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(strip_module_prefix(state_dict), strict=True)
    return model.to(device).eval()


def make_loader(cfg, split, batch_size, num_workers):
    dataset = NFISiteDataset(cfg, split, transform=build_transforms(cfg, split))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
        collate_fn=nfi_collate_fn,
    )


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


@torch.no_grad()
def run_inference(model, loader, device, split):
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
    prediction_keys = (
        "stand_target",
        "stand_prob",
        "leaf_target",
        "leaf_prob",
        "species_target",
        "species_prob",
    )
    metadata_keys = (
        "is_pure",
        "dominant_species_class_id",
        "dominant_species_proportion",
        "nfi_record_count",
        "total_expanded_basal_area_m2_ha",
    )
    for batch in tqdm(loader, desc=f"inference {split}", dynamic_ncols=True):
        batch = move_to_device(batch, device)
        outputs = model(batch)
        payload["site_id"].extend(batch["site_id"])
        payload["stand_target"].append(batch["stand_type_id"].detach().cpu())
        payload["stand_prob"].append(outputs["stand_prob"].detach().float().cpu())
        payload["leaf_target"].append(batch["leaf_target"].detach().float().cpu())
        payload["leaf_prob"].append(outputs["leaf_prob"].detach().float().cpu())
        payload["species_target"].append(batch["species_target"].detach().float().cpu())
        payload["species_prob"].append(outputs["species_prob"].detach().float().cpu())
        for key in metadata_keys:
            payload[key].append(batch[key].detach().cpu())
    for key in prediction_keys + metadata_keys:
        payload[key] = torch.cat(payload[key], dim=0).numpy()
    return payload


def evaluate_checkpoint(args, expected_fold=None):
    """Evaluate one saved model, optionally enforcing its outer-fold identity."""
    seed_everything(args.seed)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = load_config(args, checkpoint)
    if expected_fold is not None:
        if not cv_enabled(cfg) or outer_fold(cfg) != expected_fold:
            raise ValueError(f"Checkpoint {checkpoint_path} does not belong to outer fold {expected_fold}.")
        if int(checkpoint.get("epoch", -1)) != 80:
            raise ValueError(f"Checkpoint {checkpoint_path} must contain the final epoch-80 model.")
    configure_outer_fold(cfg, args.outer_fold)
    device = get_device(args, cfg)
    batch_size = args.batch_size or int(cfg.dataloader.batch_size)
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.dataloader.num_workers)
    model = load_model(cfg, checkpoint, device)

    payloads = [
        run_inference(model, make_loader(cfg, split, batch_size, num_workers), device, split)
        for split in args.splits
    ]
    if cv_enabled(cfg):
        for payload in payloads:
            payload["outer_fold"] = np.full(
                len(payload["site_id"]), outer_fold(cfg), dtype=np.int64
            )
    output_dir = output_dir_for(args)
    metrics = save_evaluation_artifacts(payloads, cfg, output_dir, source=checkpoint_path)
    print(f"Wrote evaluation artifacts: {output_dir}")
    for row in metrics:
        print(
            f"{row['split']}: n={row['n_sites']} "
            f"species_r2_macro={row.get('species_r2_macro')} "
            f"species_rmse={row.get('species_rmse_overall')} "
            f"species_f1={row.get('species_dominant_macro_f1')}"
        )

    return cfg


def main():
    evaluate_checkpoint(parse_args())


if __name__ == "__main__":
    main()
