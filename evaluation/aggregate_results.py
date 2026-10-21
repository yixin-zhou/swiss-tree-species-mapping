#!/usr/bin/env python3
"""Aggregate five untouched outer-fold predictions into one pooled OOF result."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.artifacts import metric_row, save_evaluation_artifacts
from src.utils.config import complete_config, load_training_config


ARRAY_KEYS = (
    "stand_target",
    "stand_prob",
    "leaf_target",
    "leaf_prob",
    "species_target",
    "species_prob",
    "is_pure",
    "dominant_species_class_id",
    "dominant_species_proportion",
    "nfi_record_count",
    "total_expanded_basal_area_m2_ha",
    "outer_fold",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--expected-sites", type=int, default=3992)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--set", dest="overrides", nargs="+", default=[])
    return parser.parse_args()


def find_raw_prediction(fold_root):
    direct = fold_root / "raw_predictions.npz"
    if direct.exists():
        return direct
    direct_eval = fold_root / "evaluation" / "raw_predictions.npz"
    if direct_eval.exists():
        return direct_eval
    candidates = sorted(
        fold_root.glob("*/evaluation/raw_predictions.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No fold evaluation found below {fold_root}")
    return candidates[0]


def load_fold(path, fold):
    raw = np.load(path, allow_pickle=False)
    payload = {
        "split": f"fold_{fold}",
        "site_id": raw["site_id"].astype(str).tolist(),
    }
    for key in ARRAY_KEYS:
        if key in raw:
            payload[key] = raw[key]
    n_sites = len(payload["site_id"])
    if "outer_fold" not in payload:
        payload["outer_fold"] = np.full(n_sites, fold, dtype=np.int64)
    if not np.all(np.asarray(payload["outer_fold"]) == fold):
        raise ValueError(f"Fold metadata mismatch in {path}")
    return payload


def aggregate_results(args, dataset_cfg=None):
    """Summarize held-out fold artifacts against the evaluated dataset index."""
    root = Path(args.experiment_root)
    output = Path(args.output_dir) if args.output_dir else root / "oof_evaluation"
    fold_payloads = []
    sources = []
    for fold in range(5):
        source = find_raw_prediction(root / f"fold_{fold}")
        payload = load_fold(source, fold)
        fold_payloads.append(payload)
        sources.append(str(source))
        print(f"fold={fold} n={len(payload['site_id'])} source={source}", flush=True)

    site_ids = [site_id for payload in fold_payloads for site_id in payload["site_id"]]
    if len(site_ids) != args.expected_sites:
        raise ValueError(f"Expected {args.expected_sites} OOF rows, found {len(site_ids)}")
    if len(set(site_ids)) != len(site_ids):
        duplicates = pd.Series(site_ids)[pd.Series(site_ids).duplicated()].unique().tolist()
        raise ValueError(f"OOF site IDs are not unique: {duplicates[:10]}")

    if dataset_cfg is None:
        dataset_cfg = load_training_config(config=args.config, overrides=args.overrides)
    index = pd.read_csv(dataset_cfg.path.index)
    expected_ids = set(index.site_id.astype(str))
    if set(site_ids) != expected_ids:
        raise ValueError(
            f"OOF IDs differ from index: missing={len(expected_ids - set(site_ids))}, "
            f"extra={len(set(site_ids) - expected_ids)}"
        )

    pooled = {"split": "oof", "site_id": site_ids}
    for key in ARRAY_KEYS:
        values = [payload[key] for payload in fold_payloads if key in payload]
        if values:
            if len(values) != len(fold_payloads):
                raise ValueError(f"Only some folds contain {key}")
            pooled[key] = np.concatenate(values, axis=0)

    cfg = complete_config(
        dataset_cfg,
        {"cross_validation": {"enabled": False}},
    )
    save_evaluation_artifacts([pooled], cfg, output, source=root)

    fold_rows = []
    for payload in fold_payloads:
        metrics = metric_row(payload, cfg)
        metrics["outer_fold"] = int(payload["outer_fold"][0])
        fold_rows.append(metrics)
    frame = pd.DataFrame(fold_rows).sort_values("outer_fold")
    frame.to_csv(output / "fold_metrics.csv", index=False)
    numeric = frame.select_dtypes(include=[np.number]).drop(
        columns=[column for column in ("outer_fold", "n_sites") if column in frame],
        errors="ignore",
    )
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=1)})
    summary.to_csv(output / "fold_metrics_mean_std.csv", index_label="metric")

    (output / "oof_manifest.json").write_text(
        json.dumps({"n_sites": len(site_ids), "fold_sources": sources}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote pooled OOF evaluation: {output}", flush=True)

    return output


def main():
    aggregate_results(parse_args())


if __name__ == "__main__":
    main()
