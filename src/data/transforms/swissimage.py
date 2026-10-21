import numpy as np
import torch
from torchvision.transforms import v2


class SwissimageTransform:
    # Default = SwissImage-computed stats (used with the SAT-493M backbone so far).
    # For DINOv3 LVD-1689M (web) backbones, pass ImageNet stats instead.
    DEFAULT_MEAN = (0.430, 0.411, 0.296)
    DEFAULT_STD = (0.213, 0.156, 0.143)

    def __init__(
        self,
        multiseason,
        crop_size_m=28,
        resize_size=256,
        source_size_m=50,
        norm_mean=None,
        norm_std=None,
        deterministic=False,
    ):
        # deterministic=True is used for val/test: choose the leaf-on image
        # closest to the NFI year. Training samples a leaf-on year uniformly.
        self.deterministic = bool(deterministic)
        self.multiseason = bool(multiseason)
        self.crop_size_m = float(crop_size_m)
        self.resize_size = int(resize_size)
        self.source_size_m = float(source_size_m)
        self.norm_mean = tuple(norm_mean) if norm_mean is not None else self.DEFAULT_MEAN
        self.norm_std = tuple(norm_std) if norm_std is not None else self.DEFAULT_STD

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((self.resize_size, self.resize_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])

    def __call__(self, sample):
        if "rgb" not in sample or sample["rgb"] is None:
            return sample

        rgb = sample["rgb"]
        data = rgb["data"]  # (T, 3, H, W), uint8
        seasons = rgb.get("season")
        years = rgb.get("year")
        nfi_year = self._nfi_year(sample)

        if self.multiseason:
            selected = self._select_summer_winter(
                seasons, len(data), years=years, nfi_year=nfi_year
            )
        else:
            selected = [self._select_summer(
                seasons, len(data), years=years, nfi_year=nfi_year
            )]


        indices = [item["index"] for item in selected]
        season_ids = [item["season_id"] for item in selected]
        is_fallback = [item["is_fallback"] for item in selected]
        view_mask = [item["view_mask"] for item in selected]

        transformed = torch.stack([
            self.transform(self._center_crop(data[i]))
            for i in indices
        ])

        if self.multiseason:
            rgb["data"] = transformed  # (2, 3, resize_size, resize_size)
            rgb["year"] = [years[i] for i in indices] if years is not None else None
            rgb["season"] = [seasons[i] for i in indices] if seasons is not None else None
            rgb["season_id"] = torch.tensor(season_ids, dtype=torch.long)
            rgb["is_fallback"] = torch.tensor(is_fallback, dtype=torch.bool)
            rgb["view_mask"] = torch.tensor(view_mask, dtype=torch.bool)
        else:
            rgb["data"] = transformed[0]  # (3, resize_size, resize_size)
            rgb["year"] = years[indices[0]] if years is not None else None
            rgb["season"] = seasons[indices[0]] if seasons is not None else None
            rgb["season_id"] = torch.tensor(season_ids[0], dtype=torch.long)
            rgb["is_fallback"] = torch.tensor(is_fallback[0], dtype=torch.bool)
            rgb["view_mask"] = torch.tensor(view_mask[0], dtype=torch.bool)

        rgb["selected_indices"] = indices
        rgb["selection_rule"] = (
            "nearest_leaf_on_to_nfi_year" if self.deterministic else "random_leaf_on"
        )
        rgb["crop_size_m"] = self.crop_size_m
        rgb["resize_size"] = self.resize_size

        return sample

    def _center_crop(self, image):
        _, height, width = image.shape

        crop_h = int(round(height * self.crop_size_m / self.source_size_m))
        crop_w = int(round(width * self.crop_size_m / self.source_size_m))

        if crop_h > height or crop_w > width:
            raise ValueError(
                f"crop_size_m={self.crop_size_m} is too large for image shape "
                f"{tuple(image.shape)} with source_size_m={self.source_size_m}."
            )

        top = (height - crop_h) // 2
        left = (width - crop_w) // 2

        return image[:, top:top + crop_h, left:left + crop_w]

    def _select_summer(self, seasons, n_images, years=None, nfi_year=None):
        summer_indices = SwissimageTransform._find_indices(seasons, "summer")

        if len(summer_indices) > 0:
            if self.deterministic:
                idx = self._nearest_year_index(summer_indices, years, nfi_year)
            else:
                idx = np.random.choice(summer_indices).item()
            return {
                "index": idx,
                "season_id": 0,
                "is_fallback": False,
                "view_mask": True,
            }

        idx = 0 if self.deterministic else np.random.randint(n_images)
        return {
            "index": idx,
            "season_id": 0,
            "is_fallback": True,
            "view_mask": True,
        }

    def _select_summer_winter(self, seasons, n_images, years=None, nfi_year=None):
        summer = self._select_summer(
            seasons, n_images, years=years, nfi_year=nfi_year
        )

        winter_indices = SwissimageTransform._find_indices(seasons, "winter")

        if len(winter_indices) > 0:
            winter_idx = winter_indices[0] if self.deterministic else np.random.choice(winter_indices).item()
            winter = {
                "index": winter_idx,
                "season_id": 1,
                "is_fallback": False,
                "view_mask": True,
            }
        else:
            winter = {
                "index": summer["index"],
                "season_id": 1,
                "is_fallback": True,
                "view_mask": False,
            }

        return [summer, winter]

    @staticmethod
    def _nfi_year(sample):
        metadata = sample.get("metadata", {})
        for source in (metadata.get("index", {}), metadata.get("attrs", {})):
            if "nfi_year" in source:
                try:
                    return int(float(source["nfi_year"]))
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _nearest_year_index(candidate_indices, years, nfi_year):
        if years is None or nfi_year is None:
            return candidate_indices[0]

        candidates = []
        for index in candidate_indices:
            try:
                year = int(float(years[index]))
            except (TypeError, ValueError, IndexError):
                continue
            candidates.append((abs(year - nfi_year), year, index))

        if not candidates:
            return candidate_indices[0]
        return min(candidates)[2]

    @staticmethod
    def _find_indices(seasons, target_season):
        if seasons is None or len(seasons) == 0:
            return []

        return [
            i for i, season in enumerate(seasons)
            if str(season).lower() == target_season
        ]
