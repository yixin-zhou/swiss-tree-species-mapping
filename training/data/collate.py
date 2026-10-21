import torch


def nfi_collate_fn(samples):
    batch = {
        "site_id": [sample["site_id"] for sample in samples],
        "split": [sample["split"] for sample in samples],
        "h5_path": [sample["h5_path"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "stand_type": [sample["stand_type"] for sample in samples],
    }

    tensor_keys = [
        "stand_type_id",
        "leaf_target",
        "species_target",
        "leaf_basal_area_m2_ha",
        "species_basal_area_m2_ha",
        "total_expanded_basal_area_m2_ha",
        "nfi_record_count",
        "dominant_species_class_id",
        "dominant_species_proportion",
        "is_pure",
    ]
    for key in tensor_keys:
        batch[key] = torch.stack([sample[key] for sample in samples])

    if "location_wgs84" in samples[0]:
        batch["location_wgs84"] = torch.stack([
            sample["location_wgs84"]
            for sample in samples
        ])

    if "rgb" in samples[0] and samples[0]["rgb"] is not None:
        batch["rgb"] = {
            "data": torch.stack([sample["rgb"]["data"] for sample in samples]),
            "season_id": torch.stack([sample["rgb"]["season_id"] for sample in samples]),
            "is_fallback": torch.stack([sample["rgb"]["is_fallback"] for sample in samples]),
            "view_mask": torch.stack([sample["rgb"]["view_mask"] for sample in samples]),
            "year": [sample["rgb"]["year"] for sample in samples],
            "season": [sample["rgb"]["season"] for sample in samples],
            "selected_indices": [sample["rgb"]["selected_indices"] for sample in samples],
            "selection_rule": [sample["rgb"]["selection_rule"] for sample in samples],
        }

    if "sentinel2" in samples[0] and samples[0]["sentinel2"] is not None:
        s2_0 = samples[0]["sentinel2"]
        if "embedding" in s2_0:
            s2_batch = {
                "embedding": torch.stack([sample["sentinel2"]["embedding"] for sample in samples]),
                "temporal_mode": [sample["sentinel2"]["temporal_mode"] for sample in samples],
            }
            if "year" in s2_0:
                s2_batch["year"] = torch.tensor([
                    sample["sentinel2"]["year"] for sample in samples
                ], dtype=torch.long)
        else:
            s2_batch = {
                "data": torch.stack([sample["sentinel2"]["data"] for sample in samples]),
                "month": torch.stack([sample["sentinel2"]["month"] for sample in samples]),
                "temporal_mode": [sample["sentinel2"]["temporal_mode"] for sample in samples],
            }
            # tensor keys present depend on the transform (monthly12 vs legacy composite)
            for key in ("month_valid", "valid_ratio", "valid_ratio_before_fill"):
                if key in s2_0 and torch.is_tensor(s2_0[key]):
                    s2_batch[key] = torch.stack([sample["sentinel2"][key] for sample in samples])
            for key in ("time_range", "composite_method"):
                if key in s2_0:
                    s2_batch[key] = [sample["sentinel2"][key] for sample in samples]
        batch["sentinel2"] = s2_batch

    if "chelsa" in samples[0] and samples[0]["chelsa"] is not None:
        batch["chelsa"] = {
            "values": torch.stack([sample["chelsa"]["values"] for sample in samples]),
            "normalized": [sample["chelsa"].get("normalized", False) for sample in samples],
        }

    if "dem" in samples[0] and samples[0]["dem"] is not None:
        batch["dem"] = {
            "data": torch.stack([sample["dem"]["data"] for sample in samples]),
        }

        if "transform" in samples[0]["dem"]:
            batch["dem"]["transform"] = torch.stack([
                sample["dem"]["transform"]
                for sample in samples
            ])

    if "lidar" in samples[0] and samples[0]["lidar"] is not None:
        batch["lidar"] = {
            "points": [sample["lidar"]["points"] for sample in samples],
            "classification": [
                sample["lidar"]["classification"]
                for sample in samples
            ],
            "version": [sample["lidar"]["version"] for sample in samples],
            "num_points_raw": torch.tensor([
                sample["lidar"]["num_points_raw"]
                for sample in samples
            ], dtype=torch.long),
            "num_points_after_crop": torch.tensor([
                sample["lidar"]["num_points_after_crop"]
                for sample in samples
            ], dtype=torch.long),
            "num_ground_removed": torch.tensor([
                sample["lidar"]["num_ground_removed"]
                for sample in samples
            ], dtype=torch.long),
            "num_points": torch.tensor([
                sample["lidar"]["num_points"]
                for sample in samples
            ], dtype=torch.long),
            "crop_radius_m": torch.tensor([
                sample["lidar"]["crop_radius_m"]
                for sample in samples
            ], dtype=torch.float32),
            "height_norm": [sample["lidar"]["height_norm"] for sample in samples],
        }
        if "canopy_raster" in samples[0]["lidar"]:
            batch["lidar"]["canopy_raster"] = torch.stack(
                [sample["lidar"]["canopy_raster"] for sample in samples], dim=0
            )  # (B, 3, grid, grid), fixed-size -> safe to stack

    return batch
