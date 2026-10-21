"""Random Forest baseline for the complete 11-class basal-area labels."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.dataset import dequantize_alphaearth_patch
from training.data.cross_validation import configure_outer_fold, cv_enabled, outer_fold, select_cv_rows
from src.data.label_schema import (
    LEAF_NUM_CLASSES,
    SPECIES_NUM_CLASSES,
)
from evaluation.artifacts import save_evaluation_artifacts
from src.utils.config import load_training_config


GROUND_CLASS = 2
SE_DIM = 64
RGB_CROP_SIZE_M = 28.0
RGB_SOURCE_SIZE_M = 50.0
LIDAR_CROP_RADIUS_M = 14.0
FEATURE_NAMES = (
    [f"rgb_{channel}_{stat}" for stat in ("mean", "std") for channel in ("r", "g", "b")]
    + [
        "lidar_height_mean",
        "lidar_height_std",
        "lidar_height_p50",
        "lidar_height_p95",
        "lidar_height_max",
        "lidar_canopy_cover_2m",
    ]
    + [f"aef_{index}" for index in range(SE_DIM)]
    + [f"chelsa_bio{index + 1}" for index in range(19)]
    + ["dem_elevation", "dem_slope_deg", "dem_aspect_sin", "dem_aspect_cos"]
)


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _nearest_year_index(indices, years, target_year):
    candidates = []
    for index in indices:
        try:
            year = int(float(_decode(years[index])))
        except (TypeError, ValueError, IndexError):
            continue
        candidates.append((abs(year - target_year), year, index))
    return min(candidates)[2] if candidates else indices[0]


def _rgb_features(handle):
    data = handle["swissimage/data"][()]
    seasons = [
        _decode(value)
        for value in handle["swissimage/season"][()]
    ]
    summer = [index for index, season in enumerate(seasons) if season == "summer"]
    if summer:
        years = handle["swissimage/year"][()] if "swissimage/year" in handle else None
        nfi_year = int(handle.attrs["nfi_year"])
        index = _nearest_year_index(summer, years, nfi_year) if years is not None else summer[0]
    else:
        index = 0
    image = data[index]
    _, height, width = image.shape
    crop_h = int(round(height * RGB_CROP_SIZE_M / RGB_SOURCE_SIZE_M))
    crop_w = int(round(width * RGB_CROP_SIZE_M / RGB_SOURCE_SIZE_M))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    image = image[:, top : top + crop_h, left : left + crop_w].astype(np.float32) / 255.0
    return np.concatenate((image.mean(axis=(1, 2)), image.std(axis=(1, 2)))).astype(np.float32)


def _lidar_features(handle):
    group = handle["lidar"]
    version = "new" if "new" in group else "old"
    points = group[version][()]
    x = points["X"].astype(np.float64)
    y = points["Y"].astype(np.float64)
    z_abs = points["Z"].astype(np.float64)
    classes = points["Classification"].astype(np.int64)
    if z_abs.size == 0:
        return np.zeros(6, dtype=np.float32)

    left, bottom, right, top = handle["bounds_epsg2056"][()].astype(np.float64)
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (bottom + top)
    crop = (x - center_x) ** 2 + (y - center_y) ** 2 <= LIDAR_CROP_RADIUS_M ** 2
    non_ground = crop & (classes != GROUND_CLASS)
    if not np.any(non_ground):
        return np.zeros(6, dtype=np.float32)

    x = x[non_ground]
    y = y[non_ground]
    z_abs = z_abs[non_ground]
    dem = handle["dem/data"][()].astype(np.float64)[0]
    height_px, width_px = dem.shape
    col = np.clip((x - left) / (right - left) * (width_px - 1), 0, width_px - 1)
    row = np.clip((top - y) / (top - bottom) * (height_px - 1), 0, height_px - 1)
    col0 = np.floor(col).astype(np.int64)
    row0 = np.floor(row).astype(np.int64)
    col1 = np.clip(col0 + 1, 0, width_px - 1)
    row1 = np.clip(row0 + 1, 0, height_px - 1)
    wc = col - col0
    wr = row - row0
    dem_z = (
        dem[row0, col0] * (1 - wc) * (1 - wr)
        + dem[row0, col1] * wc * (1 - wr)
        + dem[row1, col0] * (1 - wc) * wr
        + dem[row1, col1] * wc * wr
    )
    height = z_abs - dem_z
    return np.asarray(
        [
            height.mean(),
            height.std(),
            np.percentile(height, 50),
            np.percentile(height, 95),
            height.max(),
            (height > 2.0).mean(),
        ],
        dtype=np.float32,
    )


def _aef_features(handle):
    vector = dequantize_alphaearth_patch(
        handle["satellite_embedding_patch/vectors"][()]
    )
    if vector.shape != (9, SE_DIM):
        raise ValueError(f"Expected a (9,{SE_DIM}) AEF patch, got {vector.shape}.")
    center = vector[4]
    if not np.isfinite(center).all() or np.max(np.abs(center)) > 1.00001:
        raise ValueError("Invalid or non-dequantized AlphaEarth center embedding.")
    return center


def _terrain_features(handle):
    dem = handle["dem/data"][()].astype(np.float64)[0]
    height, width = dem.shape
    transform = handle["dem/transform"][()]
    pixel_size = abs(float(transform[0])) if transform is not None else 0.5
    elevation = dem[height // 2, width // 2]
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    x = (xx * pixel_size).reshape(-1)
    y = (-yy * pixel_size).reshape(-1)
    z = dem.reshape(-1)
    valid = np.isfinite(z)
    if valid.sum() < 10:
        return np.asarray([elevation, 0.0, 0.0, 1.0], dtype=np.float32)
    design = np.stack((x[valid], y[valid], np.ones(int(valid.sum()))), axis=1)
    coefficient, *_ = np.linalg.lstsq(design, z[valid], rcond=None)
    slope = np.degrees(np.arctan(np.hypot(coefficient[0], coefficient[1])))
    aspect = np.arctan2(coefficient[1], coefficient[0])
    return np.asarray([elevation, slope, np.sin(aspect), np.cos(aspect)], dtype=np.float32)


def extract_features(handle):
    features = np.concatenate(
        (
            _rgb_features(handle),
            _lidar_features(handle),
            _aef_features(handle),
            handle["chelsa/values"][()].astype(np.float32),
            _terrain_features(handle),
        )
    ).astype(np.float32)
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise ValueError(
            f"Invalid RF feature vector: shape={features.shape}, "
            f"finite={bool(np.isfinite(features).all())}."
        )
    return features


def load_split(index, hdf5_root, split, cfg):
    features, species, leaf, site_ids = [], [], [], []
    selected = select_cv_rows(index, cfg, split)
    for site_id in selected["site_id"].astype(str):
        with h5py.File(Path(hdf5_root) / f"{site_id}.h5", "r") as handle:
            features.append(extract_features(handle))
            species.append(handle["labels/species_composition"][()])
            leaf.append(handle["labels/leaf_type_composition"][()])
        site_ids.append(site_id)
    species = np.stack(species).astype(np.float32)
    leaf = np.stack(leaf).astype(np.float32)
    if species.shape[1] != SPECIES_NUM_CLASSES:
        raise ValueError(f"Expected {SPECIES_NUM_CLASSES} species classes, got {species.shape}.")
    if leaf.shape[1] != LEAF_NUM_CLASSES:
        raise ValueError(f"Expected {LEAF_NUM_CLASSES} leaf classes, got {leaf.shape}.")
    for name, target in (("species", species), ("leaf", leaf)):
        if not np.isfinite(target).all() or np.any(target < 0):
            raise ValueError(f"Invalid {name} targets in split={split}.")
        if not np.allclose(target.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"{name} targets do not sum to one in split={split}.")
    return (
        np.stack(features).astype(np.float32),
        species,
        leaf,
        site_ids,
    )


def renormalize(prediction):
    prediction = np.clip(prediction, 0.0, None)
    denominator = prediction.sum(axis=1, keepdims=True)
    return prediction / np.where(denominator > 0, denominator, 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/baselines/rf_baseline/s42")
    parser.add_argument("--outer-fold", type=int, required=True)
    args = parser.parse_args()

    cfg = load_training_config(
        config=args.config, outer_fold=args.outer_fold, overrides=args.overrides,
    )
    configure_outer_fold(cfg, args.outer_fold)
    index = pd.read_csv(cfg.path.index)
    print("extracting train features", flush=True)
    x_train, species_train, leaf_train, _ = load_split(index, cfg.path.hdf5, "train", cfg)
    print("extracting test features", flush=True)
    x_test, species_test, leaf_test, site_ids = load_split(index, cfg.path.hdf5, "test", cfg)
    print(
        f"train={len(x_train)} test={len(x_test)} features={x_train.shape[1]} classes={SPECIES_NUM_CLASSES}",
        flush=True,
    )

    model_args = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
        random_state=args.seed,
    )
    species_model = RandomForestRegressor(**model_args).fit(x_train, species_train)
    leaf_model = RandomForestRegressor(**model_args).fit(x_train, leaf_train)
    payload = {
        "split": "test",
        "site_id": site_ids,
        "species_target": species_test,
        "species_prob": renormalize(species_model.predict(x_test)),
        "leaf_target": leaf_test,
        "leaf_prob": renormalize(leaf_model.predict(x_test)),
    }
    if cv_enabled(cfg):
        payload["outer_fold"] = np.full(len(site_ids), outer_fold(cfg), dtype=np.int64)
    output_dir = Path(args.output_dir)
    save_evaluation_artifacts([payload], cfg, output_dir, source="random_forest")
    metadata = {
        "model": "random_forest",
        "label_classes": SPECIES_NUM_CLASSES,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "seed": args.seed,
        "outer_fold": outer_fold(cfg) if cv_enabled(cfg) else None,
        "feature_names": FEATURE_NAMES,
        "species_feature_importance": dict(
            zip(FEATURE_NAMES, species_model.feature_importances_.tolist())
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote baseline artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
