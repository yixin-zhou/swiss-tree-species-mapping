"""Shared spatial cross-validation selection and fold configuration helpers."""

import json

import numpy as np
from omegaconf import OmegaConf

from src.config_defaults import CV_FOLD_COLUMN, CV_FOLD_SOURCE, CV_FOLD_SEED, CV_NORMALIZATION_FILE


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def cv_config(cfg):
    return _cfg_get(cfg, "cross_validation", {})


def cv_enabled(cfg):
    return bool(_cfg_get(cv_config(cfg), "enabled", False))


def outer_fold(cfg):
    value = _cfg_get(cv_config(cfg), "outer_fold", None)
    if value is None:
        raise ValueError("cross_validation.outer_fold must be set for a CV run.")
    value = int(value)
    num_folds = int(_cfg_get(cv_config(cfg), "num_folds", 5))
    if not 0 <= value < num_folds:
        raise ValueError(f"outer fold must be in [0, {num_folds - 1}], got {value}.")
    return value


def configure_outer_fold(cfg, fold=None):
    """Set the outer fold and inject training-fold normalization statistics."""
    if not cv_enabled(cfg):
        if fold is not None:
            raise ValueError("--outer-fold was supplied but cross-validation is disabled.")
        return cfg

    if fold is not None:
        OmegaConf.update(cfg, "cross_validation.outer_fold", int(fold), merge=False)
    fold = outer_fold(cfg)

    stats_path = CV_NORMALIZATION_FILE
    if not stats_path.exists():
        raise FileNotFoundError(f"CV normalization file not found: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    expected_source = CV_FOLD_SOURCE
    expected_seed = CV_FOLD_SEED
    if int(payload.get("num_folds", -1)) != int(cfg.cross_validation.num_folds):
        raise ValueError("CV normalization fold count does not match the protocol.")
    # Record provenance in saved run metadata, not as user tuning parameters.
    OmegaConf.update(cfg, "cross_validation.fold_source", expected_source, force_add=True)
    OmegaConf.update(cfg, "cross_validation.fold_seed", expected_seed, force_add=True)
    fold_stats = payload["folds"][str(fold)]
    OmegaConf.update(
        cfg, "modalities.chelsa.norm_mean", fold_stats["chelsa_mean"], merge=False
    )
    OmegaConf.update(
        cfg, "modalities.chelsa.norm_std", fold_stats["chelsa_std"], merge=False
    )
    OmegaConf.update(
        cfg, "cross_validation.n_train", int(fold_stats["n_train"]), merge=False
    )
    OmegaConf.update(
        cfg, "cross_validation.n_test", int(fold_stats["n_test"]), merge=False
    )
    return cfg


def select_cv_rows(index, cfg, split):
    """Select the four-fold training pool or untouched outer test fold."""
    if not cv_enabled(cfg):
        if "split" not in index.columns:
            raise ValueError("There is no 'split' column in index.csv")
        return index[index["split"] == split].copy()

    if split not in {"train", "test"}:
        raise ValueError(
            f"CV runs support only split='train' or split='test'; got {split!r}. "
            "No inner validation split is used."
        )

    cv_cfg = cv_config(cfg)
    column = CV_FOLD_COLUMN
    source_column = f"{column}_source"
    seed_column = f"{column}_seed"
    required = {column, source_column, seed_column}
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"CV index columns are missing: {missing}")

    expected_source = CV_FOLD_SOURCE
    expected_seed = CV_FOLD_SEED
    sources = set(index[source_column].astype(str).unique())
    seeds = set(index[seed_column].astype(int).unique())
    if sources != {expected_source}:
        raise ValueError(f"Unexpected {source_column} values: {sorted(sources)}")
    if seeds != {expected_seed}:
        raise ValueError(f"Unexpected {seed_column} values: {sorted(seeds)}")

    folds = index[column].to_numpy(dtype=np.int64)
    num_folds = int(_cfg_get(cv_cfg, "num_folds", 5))
    observed = set(np.unique(folds).tolist())
    expected = set(range(num_folds))
    if observed != expected:
        raise ValueError(f"Expected CV folds {sorted(expected)}, found {sorted(observed)}")

    held_out = outer_fold(cfg)
    mask = folds != held_out if split == "train" else folds == held_out
    selected = index.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"No rows selected for CV split={split}, outer_fold={held_out}.")
    return selected


def cv_metadata(cfg):
    if not cv_enabled(cfg):
        return {}
    return {
        "outer_fold": outer_fold(cfg),
        "cv_fold_column": CV_FOLD_COLUMN,
        "cv_source": CV_FOLD_SOURCE,
        "cv_seed": CV_FOLD_SEED,
    }
