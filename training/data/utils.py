from training.data.augmentation import GeometricAugmentation
from src.data.transforms import (
    ChelsaTransform,
    Compose,
    Sentinel2AnnualCompositeTransform,
    Sentinel2Monthly12Transform,
    Sentinel2Transform,
    SiteLocationTransform,
    SwissLidarTransform,
    SwissimageTransform,
)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_transforms(cfg, split):
    transforms = [
        SiteLocationTransform(),
    ]

    if bool(cfg.modalities.rgb.enabled):
        rgb_cfg = cfg.modalities.rgb
        norm_mean = rgb_cfg.get("norm_mean", None) if hasattr(rgb_cfg, "get") else None
        norm_std = rgb_cfg.get("norm_std", None) if hasattr(rgb_cfg, "get") else None
        transforms.append(
            SwissimageTransform(
                multiseason=rgb_cfg.multiseason,
                crop_size_m=rgb_cfg.crop_size_m,
                resize_size=rgb_cfg.resize_size,
                norm_mean=list(norm_mean) if norm_mean is not None else None,
                norm_std=list(norm_std) if norm_std is not None else None,
                # Eval uses the leaf-on candidate nearest to the NFI year;
                # training keeps the random leaf-on year pick as augmentation.
                deterministic=(split != "train"),
            )
        )

    if bool(cfg.modalities.sentinel2.enabled):
        # s2_mode is a model-side concept (single source of truth); the transform
        # must match the branch. monthly_composite/monthly12 -> selected monthly
        # composites + per-month mask.
        model_s2 = _cfg_get(_cfg_get(cfg, "model", {}), "sentinel2", {})
        s2_mode = str(_cfg_get(model_s2, "s2_mode", "legacy"))
        if s2_mode in ("satellite_embedding", "satellite_embedding_patch"):
            pass
        elif s2_mode in ("monthly12", "monthly_composite"):
            months = _cfg_get(cfg.modalities.sentinel2, "months", None)
            transforms.append(
                Sentinel2Monthly12Transform(
                    min_valid_ratio=_cfg_get(cfg.modalities.sentinel2, "min_valid_ratio", 0.30),
                    months=list(months) if months is not None else None,
                )
            )
        elif s2_mode == "annual_composite":
            transforms.append(
                Sentinel2AnnualCompositeTransform(
                    composite_method=_cfg_get(cfg.modalities.sentinel2, "composite_method", "median"),
                )
            )
        else:
            transforms.append(
                Sentinel2Transform(
                    temporal_mode=cfg.modalities.sentinel2.temporal_mode,
                    time_range=cfg.modalities.sentinel2.time_range,
                    growing_season=cfg.modalities.sentinel2.growing_season,
                    composite_method=cfg.modalities.sentinel2.composite_method,
                )
            )

    if bool(cfg.modalities.chelsa.enabled):
        chelsa_cfg = cfg.modalities.chelsa
        transforms.append(
            ChelsaTransform(
                add_terrain=_cfg_get(chelsa_cfg, "terrain", False),
                mean=_cfg_get(chelsa_cfg, "norm_mean", None),
                std=_cfg_get(chelsa_cfg, "norm_std", None),
            )
        )

    if bool(cfg.modalities.lidar.enabled):
        raster_cfg = _cfg_get(cfg.modalities.lidar, "canopy_raster", {})
        transforms.append(
            SwissLidarTransform(
                split=split,
                version=cfg.modalities.lidar.version,
                crop_radius_m=cfg.modalities.lidar.crop_radius_m,
                height_norm=cfg.modalities.lidar.height_norm,
                compute_canopy_raster=_cfg_get(raster_cfg, "enabled", False),
                canopy_raster_crop_m=_cfg_get(raster_cfg, "crop_m", 28.0),
                canopy_raster_resolution_m=_cfg_get(raster_cfg, "resolution_m", 1.0),
                canopy_raster_ground_class=_cfg_get(raster_cfg, "ground_class", 2),
                canopy_raster_percentile=_cfg_get(raster_cfg, "percentile", 75.0),
                canopy_raster_min_points=_cfg_get(raster_cfg, "min_points", 3),
            )
        )

    # Geometric augmentation runs LAST (on the final per-modality tensors) and
    # ONLY on training splits.
    aug_cfg = _cfg_get(cfg, "augmentation", {})
    if str(split).startswith("train") and bool(_cfg_get(aug_cfg, "geometric", False)):
        transforms.append(
            GeometricAugmentation(
                p=_cfg_get(aug_cfg, "geometric_p", 1.0),
                rotations=_cfg_get(aug_cfg, "geometric_rotations", True),
                flips=_cfg_get(aug_cfg, "geometric_flips", True),
                apply_sentinel2=_cfg_get(aug_cfg, "geometric_sentinel2", True),
            )
        )

    return Compose(transforms)
