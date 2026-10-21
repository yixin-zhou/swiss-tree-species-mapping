import torch


class Sentinel2Monthly12Transform:
    """Cross-year monthly median composites, QAI-masked,
    NO gap-fill. Keeps NaN where a pixel had no valid observation that month; the
    branch does masked spatial mean. Emits per-month validity (>= min_valid_ratio).
    """

    def __init__(self, min_valid_ratio=0.30, nodata_value=-9999, scale_factor=10000.0, months=None):
        self.min_valid_ratio = float(min_valid_ratio)
        self.nodata_value = nodata_value
        self.scale_factor = float(scale_factor)
        self.months = [int(m) for m in (months or range(1, 13))]
        if not self.months:
            raise ValueError("Sentinel2Monthly12Transform requires at least one month.")
        if any(m < 1 or m > 12 for m in self.months):
            raise ValueError(f"Invalid Sentinel-2 month list: {self.months}.")

    def __call__(self, sample):
        if "sentinel2" not in sample or sample["sentinel2"] is None:
            return sample
        s2 = sample["sentinel2"]
        boa = s2["boa"].float()                      # (S, 10, 5, 5)
        qai = s2.get("qai_valid_mask", None)
        dates = s2.get("date", None)
        if dates is None:
            raise ValueError("Sentinel2Monthly12Transform requires date information.")

        valid = torch.isfinite(boa) & (boa != self.nodata_value)
        if qai is not None:
            valid = valid & qai.bool()               # (S,1,5,5) broadcasts over bands
        boa = torch.where(valid, boa, torch.full_like(boa, float("nan")))
        if self.scale_factor and self.scale_factor > 0:
            boa = boa / self.scale_factor

        months = torch.tensor([int(str(d)[5:7]) for d in dates], dtype=torch.long)
        comps, ratios = [], []
        for m in self.months:
            idx = torch.where(months == m)[0]
            if idx.numel() == 0:
                comp = torch.full_like(boa[0], float("nan"))
            else:
                comp = torch.nanmedian(boa[idx], dim=0).values   # (10,5,5), NaN if all-NaN
            valid_pix = torch.isfinite(comp).all(dim=0)          # (5,5) all bands valid
            ratios.append(valid_pix.float().mean())
            comps.append(comp)

        data = torch.stack(comps, dim=0)             # (12,10,5,5) with NaN
        ratios = torch.stack(ratios)                 # (12,)
        s2["data"] = data
        s2["month"] = torch.tensor(self.months, dtype=torch.long)
        s2["month_valid"] = ratios >= self.min_valid_ratio   # (12,) bool
        s2["valid_ratio"] = ratios
        s2["temporal_mode"] = "monthly12"
        return sample


class Sentinel2AnnualCompositeTransform:
    """Single annual Sentinel-2 composite over all available observations."""

    def __init__(
        self,
        composite_method="median",
        nodata_value=-9999,
        scale_factor=10000.0,
        fill_value=0.0,
    ):
        if composite_method not in ["median", "mean"]:
            raise ValueError(
                f"Unknown composite_method='{composite_method}'. Expected 'median' or 'mean'."
            )
        self.composite_method = composite_method
        self.nodata_value = nodata_value
        self.scale_factor = float(scale_factor)
        self.fill_value = float(fill_value)

    def __call__(self, sample):
        if "sentinel2" not in sample or sample["sentinel2"] is None:
            return sample

        s2 = sample["sentinel2"]
        boa = s2["boa"].float()  # (S, 10, 5, 5)
        qai = s2.get("qai_valid_mask", None)

        valid = torch.isfinite(boa) & (boa != self.nodata_value)
        if qai is not None:
            valid = valid & qai.bool()
        boa = torch.where(valid, boa, torch.full_like(boa, float("nan")))
        if self.scale_factor and self.scale_factor > 0:
            boa = boa / self.scale_factor

        if self.composite_method == "median":
            comp = torch.nanmedian(boa, dim=0).values
        else:
            comp = torch.nanmean(boa, dim=0)

        valid_pix = torch.isfinite(comp).all(dim=0)
        valid_ratio = valid_pix.float().mean()
        comp = Sentinel2Transform._fill_2d_nan_by_neighbor_mean_per_band(comp)
        comp = torch.nan_to_num(comp, nan=self.fill_value)

        s2["data"] = comp.unsqueeze(0)  # (1,10,5,5)
        s2["month"] = torch.tensor([7], dtype=torch.long)
        s2["valid_ratio_before_fill"] = torch.tensor([float(valid_ratio)], dtype=torch.float32)
        s2["temporal_mode"] = "annual_composite"
        s2["composite_method"] = self.composite_method
        return sample


class Sentinel2Transform:
    def __init__(
        self,
        temporal_mode="composite",
        time_range="growing_season",
        growing_season=(5, 6, 7, 8, 9, 10),
        composite_method="median",
        valid_threshold=0.7,
        nodata_value=-9999,
        scale_factor=10000.0,
        fill_value=0.0,
    ):
        if temporal_mode not in ["raw", "composite"]:
            raise ValueError(
                f"Unknown temporal_mode='{temporal_mode}'. "
                "Expected 'raw' or 'composite'."
            )

        if time_range not in ["growing_season", "all_year"]:
            raise ValueError(
                f"Unknown time_range='{time_range}'. "
                "Expected 'growing_season' or 'all_year'."
            )

        if composite_method not in ["median", "mean"]:
            raise ValueError(
                f"Unknown composite_method='{composite_method}'. "
                "Expected 'median' or 'mean'."
            )

        self.temporal_mode = temporal_mode
        self.time_range = time_range
        self.growing_season = list(growing_season)
        self.composite_method = composite_method
        self.valid_threshold = float(valid_threshold)
        self.nodata_value = nodata_value
        self.scale_factor = float(scale_factor)
        self.fill_value = float(fill_value)

    def __call__(self, sample):
        if "sentinel2" not in sample or sample["sentinel2"] is None:
            return sample

        s2 = sample["sentinel2"]

        boa = s2["boa"].float()  # (S, 10, 5, 5)
        dates = s2.get("date", None)
        qai_mask = s2.get("qai_valid_mask", None)

        boa = self._apply_valid_mask(boa, qai_mask)

        if self.scale_factor is not None and self.scale_factor > 0:
            boa = boa / self.scale_factor

        if self.temporal_mode == "raw":
            s2["data"] = torch.nan_to_num(boa, nan=self.fill_value)
            s2["temporal_mode"] = "raw"
            return sample

        months = self._target_months()
        date_months = self._parse_months(dates)

        composites = []
        valid_ratios = []

        for month in months:
            month_comp = self._monthly_composite(boa, date_months, month)
            valid_ratio = self._valid_pixel_ratio(month_comp)

            composites.append(month_comp)
            valid_ratios.append(valid_ratio)

        data = torch.stack(composites, dim=0)  # (T, 10, 5, 5)
        valid_ratios = torch.tensor(valid_ratios, dtype=torch.float32)

        data = self._temporal_fill_low_quality_months(data, valid_ratios)
        data = self._spatial_fill_remaining_nan(data)
        data = torch.nan_to_num(data, nan=self.fill_value)

        s2["data"] = data
        s2["month"] = torch.tensor(months, dtype=torch.long)
        s2["valid_ratio_before_fill"] = valid_ratios
        s2["temporal_mode"] = "composite"
        s2["time_range"] = self.time_range
        s2["composite_method"] = self.composite_method

        return sample

    def _apply_valid_mask(self, boa, qai_mask):
        valid = torch.isfinite(boa) & (boa != self.nodata_value)

        if qai_mask is not None:
            qai_mask = qai_mask.bool()  # (S, 1, H, W)
            valid = valid & qai_mask

        return torch.where(
            valid,
            boa,
            torch.tensor(float("nan"), dtype=boa.dtype, device=boa.device),
        )

    def _target_months(self):
        if self.time_range == "growing_season":
            return self.growing_season

        if self.time_range == "all_year":
            return list(range(1, 13))

        raise ValueError(f"Unknown time_range='{self.time_range}'.")

    @staticmethod
    def _parse_months(dates):
        if dates is None:
            raise ValueError("Sentinel-2 composite mode requires date information.")

        months = []

        for date in dates:
            date = str(date)

            try:
                month = int(date[5:7])
            except Exception as e:
                raise ValueError(f"Cannot parse month from Sentinel-2 date: {date}") from e

            months.append(month)

        return torch.tensor(months, dtype=torch.long)

    def _monthly_composite(self, boa, date_months, month):
        idx = torch.where(date_months == month)[0]

        if idx.numel() == 0:
            return torch.full_like(boa[0], float("nan"))

        month_data = boa[idx]  # (K, C, H, W)

        if self.composite_method == "median":
            return torch.nanmedian(month_data, dim=0).values

        if self.composite_method == "mean":
            return torch.nanmean(month_data, dim=0)

        raise ValueError(f"Unknown composite_method='{self.composite_method}'.")

    @staticmethod
    def _valid_pixel_ratio(composite):
        finite = torch.isfinite(composite)  # (C, H, W)
        valid_pixel = finite.all(dim=0)  # (H, W), all bands valid
        return valid_pixel.float().mean().item()

    def _temporal_fill_low_quality_months(self, data, valid_ratios):
        filled = data.clone()
        num_steps = filled.shape[0]

        good = valid_ratios >= self.valid_threshold

        for t in range(num_steps):
            if good[t]:
                continue

            prev_idx = self._find_previous_good(good, t)
            next_idx = self._find_next_good(good, t)

            if prev_idx is not None and next_idx is not None:
                filled[t] = torch.nanmean(
                    torch.stack([filled[prev_idx], filled[next_idx]], dim=0),
                    dim=0,
                )
            elif prev_idx is not None:
                filled[t] = filled[prev_idx]
            elif next_idx is not None:
                filled[t] = filled[next_idx]
            else:
                filled[t] = self._global_temporal_fallback(filled)

        return filled

    @staticmethod
    def _find_previous_good(good, t):
        for i in range(t - 1, -1, -1):
            if bool(good[i]):
                return i
        return None

    @staticmethod
    def _find_next_good(good, t):
        for i in range(t + 1, len(good)):
            if bool(good[i]):
                return i
        return None

    @staticmethod
    def _global_temporal_fallback(data):
        if torch.isfinite(data).any():
            return torch.nanmedian(data, dim=0).values

        return torch.zeros_like(data[0])

    def _spatial_fill_remaining_nan(self, data):
        filled = data.clone()

        for t in range(filled.shape[0]):
            filled[t] = self._fill_2d_nan_by_neighbor_mean_per_band(filled[t])

        return filled

    @classmethod
    def _fill_2d_nan_by_neighbor_mean_per_band(cls, image):
        filled = image.clone()
        for c in range(filled.shape[0]):
            filled[c] = cls._fill_2d_nan_by_neighbor_mean(filled[c])
        return filled

    @staticmethod
    def _fill_2d_nan_by_neighbor_mean(image):
        image = image.clone()

        if torch.isfinite(image).all():
            return image

        if not torch.isfinite(image).any():
            return torch.zeros_like(image)

        for _ in range(8):
            nan_mask = ~torch.isfinite(image)

            if not nan_mask.any():
                break

            padded = torch.nn.functional.pad(
                image[None, None],
                pad=(1, 1, 1, 1),
                mode="replicate",
            )[0, 0]

            neighbors = torch.stack([
                padded[:-2, :-2],
                padded[:-2, 1:-1],
                padded[:-2, 2:],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
                padded[2:, :-2],
                padded[2:, 1:-1],
                padded[2:, 2:],
            ])

            valid_neighbors = torch.isfinite(neighbors)
            neighbor_sum = torch.where(
                valid_neighbors,
                neighbors,
                torch.zeros_like(neighbors),
            ).sum(dim=0)
            neighbor_count = valid_neighbors.float().sum(dim=0)

            can_fill = nan_mask & (neighbor_count > 0)
            neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1.0)

            image = torch.where(can_fill, neighbor_mean, image)

        if ~torch.isfinite(image).all():
            fallback = torch.nanmedian(image)
            if not torch.isfinite(fallback):
                fallback = torch.tensor(0.0, dtype=image.dtype, device=image.device)
            image = torch.nan_to_num(image, nan=float(fallback))

        return image
