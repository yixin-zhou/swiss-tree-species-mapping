import numpy as np
import torch


GROUND_CLASS = 2


def remove_ground_points(points, classification, *, context="LiDAR sample"):
    """Remove LAS ground-class points while preserving point/class alignment."""
    if classification is None:
        raise KeyError(f"Missing LiDAR classification for {context}; cannot remove ground points.")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected LiDAR points with shape (N, >=3), got {tuple(points.shape)}.")

    classification = classification.reshape(-1)
    if points.shape[0] != classification.shape[0]:
        raise ValueError(
            f"LiDAR point/classification length mismatch for {context}: "
            f"{points.shape[0]} points versus {classification.shape[0]} classes."
        )

    keep = classification != GROUND_CLASS
    num_ground_removed = int((~keep).sum().item())
    points = points[keep]
    classification = classification[keep]
    if points.shape[0] == 0:
        raise ValueError(
            f"No non-ground LiDAR points remain for {context} after removing "
            f"LAS classification={GROUND_CLASS}."
        )
    return points, classification, num_ground_removed


class SwissLidarTransform:
    def __init__(
        self,
        split,
        version="random",
        crop_radius_m=14,
        height_norm="dem",
        compute_canopy_raster=False,
        canopy_raster_crop_m=28.0,
        canopy_raster_resolution_m=1.0,
        canopy_raster_ground_class=2,
        canopy_raster_percentile=75.0,
        canopy_raster_min_points=3,
    ):
        valid_versions = ["random", "old", "new"]
        valid_height_norms = ["dem", "median", "raw"]

        if version not in valid_versions:
            raise ValueError(
                f"Unknown lidar version='{version}'. "
                f"Expected one of {valid_versions}."
            )

        if height_norm not in valid_height_norms:
            raise ValueError(
                f"Unknown height_norm='{height_norm}'. "
                f"Expected one of {valid_height_norms}."
            )

        self.split = split
        self.version = version
        self.crop_radius_m = float(crop_radius_m)
        self.height_norm = height_norm

        # --- optional upper-canopy raster (for LidarCanopyRasterBranch) ---
        # Independent of the circular crop_radius_m above: uses its own SQUARE
        # crop (matching SwissimageTransform's crop_size_m footprint) computed
        # from the full, uncropped point cloud, so a circular exclusion of the
        # square's corners never happens.
        self.compute_canopy_raster = bool(compute_canopy_raster)
        self.canopy_raster_crop_m = float(canopy_raster_crop_m)
        self.canopy_raster_resolution_m = float(canopy_raster_resolution_m)
        self.canopy_raster_ground_class = int(canopy_raster_ground_class)
        self.canopy_raster_percentile = float(canopy_raster_percentile)
        self.canopy_raster_min_points = int(canopy_raster_min_points)
        self._raster_grid = int(round(self.canopy_raster_crop_m / self.canopy_raster_resolution_m))

    def __call__(self, sample):
        if "lidar" not in sample or sample["lidar"] is None:
            return sample

        lidar = sample["lidar"]
        version = self._select_version(lidar)

        xyz_abs = lidar[version]["xyz"].float()  # (N, 3), absolute LV95
        classification = lidar[version].get("classification", None)

        bounds = self._get_bounds(sample)
        xyz_full = self._to_centered_xyz(xyz_abs, bounds)  # centered, NOT yet cropped/height-normalized

        canopy_raster = None
        if self.compute_canopy_raster:
            canopy_raster = self._compute_canopy_raster(xyz_full, classification, sample, bounds)

        num_points_raw = xyz_full.shape[0]

        crop_mask = self._circular_crop_mask(xyz_full)
        xyz = xyz_full[crop_mask]

        if classification is not None:
            classification = classification[crop_mask]

        if xyz.shape[0] == 0:
            site_id = sample.get("site_id", "unknown")
            raise ValueError(
                f"No LiDAR points left after circular crop for site_id={site_id}, "
                f"version={version}, crop_radius_m={self.crop_radius_m}."
            )

        xyz = self._normalize_height(xyz, sample, bounds)
        site_id = sample.get("site_id", "unknown")
        xyz, classification, num_ground_removed = remove_ground_points(
            xyz,
            classification,
            context=f"site_id={site_id}, version={version}",
        )

        sample["lidar"] = {
            "points": xyz,
            "classification": classification,
            "version": version,
            "num_points_raw": int(num_points_raw),
            "num_points_after_crop": int(crop_mask.sum().item()),
            "num_ground_removed": num_ground_removed,
            "num_points": int(xyz.shape[0]),
            "crop_radius_m": self.crop_radius_m,
            "height_norm": self.height_norm,
        }
        if canopy_raster is not None:
            sample["lidar"]["canopy_raster"] = canopy_raster

        return sample

    def _compute_canopy_raster(self, xyz_full, classification_full, sample, bounds):
        """(3, grid, grid) raster: channels = [P50, P90, P90-P50] of upper-canopy
        points (height > site-level P75, ground points excluded) per grid cell.
        Uses its own SQUARE crop (canopy_raster_crop_m), independent of the
        circular crop_radius_m used for the sparse-conv `points` path above.
        """
        grid = self._raster_grid
        empty = torch.zeros(3, grid, grid)

        half = self.canopy_raster_crop_m / 2.0
        square_mask = self._square_crop_mask(xyz_full, half)
        xyz = xyz_full[square_mask]
        cls = classification_full[square_mask] if classification_full is not None else None

        if xyz.shape[0] == 0:
            return empty

        xyz = self._normalize_height(xyz, sample, bounds)

        if cls is not None:
            non_ground = cls != self.canopy_raster_ground_class
            xyz = xyz[non_ground]

        if xyz.shape[0] == 0:
            return empty

        heights = xyz[:, 2]
        canopy_thresh = torch.quantile(heights, self.canopy_raster_percentile / 100.0)
        canopy_mask = heights > canopy_thresh
        cx = xyz[canopy_mask, 0]
        cy = xyz[canopy_mask, 1]
        ch = heights[canopy_mask]

        if ch.numel() == 0:
            return empty

        res = self.canopy_raster_resolution_m
        col = torch.clamp(torch.floor((cx + half) / res).long(), 0, grid - 1)
        # Raster rows run north to south, while centered LiDAR y increases north.
        row = torch.clamp(torch.floor((half - cy) / res).long(), 0, grid - 1)
        cell_id = row * grid + col

        order = torch.argsort(cell_id)
        cell_id_sorted = cell_id[order]
        ch_sorted = ch[order]

        unique_cells, counts = torch.unique_consecutive(cell_id_sorted, return_counts=True)
        starts = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long), counts[:-1]]), dim=0
        )

        p50_flat = torch.zeros(grid * grid)
        p90_flat = torch.zeros(grid * grid)
        for uc, cnt, st in zip(unique_cells.tolist(), counts.tolist(), starts.tolist()):
            if cnt < self.canopy_raster_min_points:
                continue
            vals = ch_sorted[st:st + cnt]
            p50_flat[uc] = torch.quantile(vals, 0.5)
            p90_flat[uc] = torch.quantile(vals, 0.9)

        p50_grid = p50_flat.view(grid, grid)
        p90_grid = p90_flat.view(grid, grid)

        raster = torch.zeros(3, grid, grid)
        raster[0] = p50_grid
        raster[1] = p90_grid
        raster[2] = p90_grid - p50_grid
        return raster

    def _square_crop_mask(self, xyz, half_size_m):
        return (xyz[:, 0].abs() <= half_size_m) & (xyz[:, 1].abs() <= half_size_m)

    def _select_version(self, lidar):
        available = [key for key in ["old", "new"] if key in lidar]

        if len(available) == 0:
            raise ValueError("LiDAR is enabled, but neither 'old' nor 'new' is available.")

        if self.version in ["old", "new"]:
            if self.version not in available:
                raise ValueError(
                    f"Requested lidar version='{self.version}', "
                    f"but available versions are {available}."
                )
            return self.version

        if self.version == "random":
            if self.split.startswith("train") and len(available) > 1:
                return np.random.choice(available).item()

            if "new" in available:
                return "new"

            return "old"

        raise ValueError(f"Unknown lidar version='{self.version}'.")

    @staticmethod
    def _get_bounds(sample):
        metadata = sample.get("metadata", {})
        bounds = metadata.get("bounds_epsg2056", None)

        if bounds is None:
            raise KeyError("Missing sample['metadata']['bounds_epsg2056'].")

        return bounds.float()

    @staticmethod
    def _to_centered_xyz(xyz_abs, bounds):
        left, bottom, right, top = bounds

        center_x = 0.5 * (left + right)
        center_y = 0.5 * (bottom + top)

        xyz = xyz_abs.clone()
        xyz[:, 0] = xyz[:, 0] - center_x
        xyz[:, 1] = xyz[:, 1] - center_y

        return xyz

    def _circular_crop_mask(self, xyz):
        xy_dist2 = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        return xy_dist2 <= self.crop_radius_m ** 2

    def _normalize_height(self, xyz, sample, bounds):
        if self.height_norm == "raw":
            return xyz

        xyz = xyz.clone()

        if self.height_norm == "median":
            xyz[:, 2] = xyz[:, 2] - torch.median(xyz[:, 2])
            return xyz

        if self.height_norm == "dem":
            dem_z = self._sample_dem_height(xyz, sample, bounds)
            xyz[:, 2] = xyz[:, 2] - dem_z
            return xyz

        raise ValueError(f"Unknown height_norm='{self.height_norm}'.")

    @staticmethod
    def _sample_dem_height(xyz_centered, sample, bounds):
        if "dem" not in sample or sample["dem"] is None:
            raise KeyError(
                "height_norm='dem' requires sample['dem']['data'], "
                "but DEM is missing."
            )

        dem = sample["dem"]["data"]

        if dem.ndim == 3:
            dem = dem[0]  # (H, W)

        if dem.ndim != 2:
            raise ValueError(f"Expected DEM shape (1, H, W) or (H, W), got {tuple(dem.shape)}.")

        dem = dem.float()

        left, bottom, right, top = bounds
        center_x = 0.5 * (left + right)
        center_y = 0.5 * (bottom + top)

        abs_x = xyz_centered[:, 0] + center_x
        abs_y = xyz_centered[:, 1] + center_y

        h, w = dem.shape

        col = (abs_x - left) / (right - left) * (w - 1)
        row = (top - abs_y) / (top - bottom) * (h - 1)

        col = col.clamp(0, w - 1)
        row = row.clamp(0, h - 1)

        col0 = torch.floor(col).long()
        row0 = torch.floor(row).long()
        col1 = (col0 + 1).clamp(max=w - 1)
        row1 = (row0 + 1).clamp(max=h - 1)

        wc = col - col0.float()
        wr = row - row0.float()

        z00 = dem[row0, col0]
        z01 = dem[row0, col1]
        z10 = dem[row1, col0]
        z11 = dem[row1, col1]

        dem_z = (
            z00 * (1 - wc) * (1 - wr)
            + z01 * wc * (1 - wr)
            + z10 * (1 - wc) * wr
            + z11 * wc * wr
        )

        if not torch.isfinite(dem_z).all():
            finite = torch.isfinite(dem_z)
            fallback = torch.median(dem_z[finite]) if finite.any() else torch.tensor(0.0)
            dem_z = torch.where(finite, dem_z, fallback)

        return dem_z
