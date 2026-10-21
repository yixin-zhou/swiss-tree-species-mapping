from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.label_schema import (
    LABEL_SCHEMA_VERSION,
    LEAF_NUM_CLASSES,
    LEAF_TYPE_ORDER,
    SPECIES_CLASS_ORDER,
    SPECIES_NUM_CLASSES,
    STAND_TYPE_TO_ID,
    decode_json_attr,
)
from training.data.cross_validation import cv_metadata, select_cv_rows


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def dequantize_alphaearth_patch(quantized):
    quantized = np.asarray(quantized, dtype=np.float32)
    valid = np.isfinite(quantized) & (quantized > -128)
    embedding = ((quantized / 127.5) ** 2) * np.sign(quantized)
    return np.where(valid, embedding, np.nan).astype(np.float32)


class NFISiteDataset(Dataset):
    def __init__(self, cfg, split, transform=None):
        self.cfg = cfg
        self.split = split
        self.transform = transform

        self.hdf5_root = Path(cfg.path.hdf5)
        self.index_path = Path(cfg.path.index)

        self.use_rgb = bool(cfg.modalities.rgb.enabled)
        self.use_sentinel2 = bool(cfg.modalities.sentinel2.enabled)
        model_s2_cfg = _cfg_get(_cfg_get(cfg, "model", {}), "sentinel2", {})
        self.sentinel2_mode = str(_cfg_get(model_s2_cfg, "s2_mode", "legacy"))
        self.use_dem = bool(cfg.modalities.dem.enabled)
        self.use_chelsa = bool(cfg.modalities.chelsa.enabled)
        self.use_lidar = bool(cfg.modalities.lidar.enabled)

        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        self.index = pd.read_csv(self.index_path)

        self.index = select_cv_rows(self.index, cfg, split).reset_index(drop=True)
        if len(self.index) == 0:
            raise ValueError(f"No rows found for split='{split}' in {self.index_path}")

        self.cv_metadata = cv_metadata(cfg)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        site_id = str(row["site_id"])
        h5_path = self.hdf5_root / f"{site_id}.h5"

        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found for site_id={site_id}: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            sample = {
                "site_id": site_id,
                "split": self.split,
                "h5_path": str(h5_path),
                "metadata": self._read_metadata(f, row),
                **self._read_labels(f),
            }

            if self.use_rgb:
                sample["rgb"] = self._read_rgb(f)

            if self.use_sentinel2:
                sample["sentinel2"] = self._read_sentinel2(f)

            if self.use_dem:
                sample["dem"] = self._read_dem(f)

            if self.use_chelsa:
                sample["chelsa"] = self._read_chelsa(f)

            if self.use_lidar:
                sample["lidar"] = self._read_lidar(f)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _read_rgb(self, f):
        if "swissimage/data" not in f:
            return None

        data = f["swissimage/data"][:]
        years = self._decode_array(f["swissimage/year"][:]) if "swissimage/year" in f else None
        seasons = self._decode_array(f["swissimage/season"][:]) if "swissimage/season" in f else None

        return {
            "data": torch.from_numpy(data),
            "year": years,
            "season": seasons,
        }

    def _read_sentinel2(self, f):
        if self.sentinel2_mode == "satellite_embedding":
            if "satellite_embedding/vector" not in f:
                return None
            out = {
                "embedding": torch.from_numpy(f["satellite_embedding/vector"][:]).float(),
                "temporal_mode": "satellite_embedding",
            }
            if "satellite_embedding/year" in f:
                out["year"] = int(f["satellite_embedding/year"][()])
            return out

        if self.sentinel2_mode == "satellite_embedding_patch":
            if "satellite_embedding_patch/vectors" not in f:
                return None
            embedding = dequantize_alphaearth_patch(
                f["satellite_embedding_patch/vectors"][:]
            )
            out = {
                "embedding": torch.from_numpy(embedding),  # (N,64), signed unit values
                "temporal_mode": "satellite_embedding_patch",
            }
            if "satellite_embedding_patch/year" in f:
                out["year"] = int(f["satellite_embedding_patch/year"][()])
            return out

        if "sentinel2/boa/data" not in f:
            return None

        boa = f["sentinel2/boa/data"][:]
        dates = self._decode_array(f["sentinel2/date"][:]) if "sentinel2/date" in f else None

        qai_valid_mask = None
        if "sentinel2/qai_valid_mask/data" in f:
            qai_valid_mask = f["sentinel2/qai_valid_mask/data"][:]

        return {
            "boa": torch.from_numpy(boa).float(),
            "date": dates,
            "qai_valid_mask": (
                torch.from_numpy(qai_valid_mask).bool()
                if qai_valid_mask is not None
                else None
            ),
        }

    def _read_dem(self, f):
        if "dem/data" not in f:
            return None

        output = {
            "data": torch.from_numpy(f["dem/data"][:]).float(),
        }

        if "dem/transform" in f:
            output["transform"] = torch.from_numpy(f["dem/transform"][:]).float()

        return output

    def _read_chelsa(self, f):
        if "chelsa/values" not in f:
            return None

        return {
            "values": torch.from_numpy(f["chelsa/values"][:]).float(),
        }

    def _read_lidar(self, f):
        if "lidar" not in f:
            return None

        lidar = {}

        if "lidar/old" in f:
            lidar["old"] = self._read_lidar_points(f["lidar/old"])

        if "lidar/new" in f:
            lidar["new"] = self._read_lidar_points(f["lidar/new"])

        if len(lidar) == 0:
            return None

        return lidar

    def _read_metadata(self, f, row):
        metadata = {
            "index": {
                key: self._to_python_value(value)
                for key, value in row.to_dict().items()
            },
            "attrs": {
                key: self._to_python_value(value)
                for key, value in f.attrs.items()
            },
        }

        if "bounds_epsg2056" in f:
            metadata["bounds_epsg2056"] = torch.from_numpy(
                f["bounds_epsg2056"][:]
            ).float()

        return metadata

    def _read_labels(self, f):
        if "labels" not in f:
            raise KeyError("Missing required group: labels")

        labels = f["labels"]
        required = [
            "stand_type",
            "leaf_type_composition",
            "leaf_type_basal_area_m2_ha",
            "species_composition",
            "species_basal_area_m2_ha",
            "total_expanded_basal_area_m2_ha",
            "nfi_record_count",
            "dominant_species_class_id",
            "dominant_species_proportion",
            "is_class_pure_80pct",
        ]
        missing = [key for key in required if key not in labels]
        if missing:
            raise KeyError(f"Missing required labels datasets: {missing}")

        schema_version = self._read_h5_string(
            labels.attrs.get("label_schema_version", f.attrs.get("label_schema_version", ""))
        )
        if schema_version != LABEL_SCHEMA_VERSION:
            raise ValueError(
                f"Expected label schema '{LABEL_SCHEMA_VERSION}', got '{schema_version}'."
            )

        species_order = decode_json_attr(labels.attrs.get("species_class_order", "[]"))
        leaf_order = decode_json_attr(labels.attrs.get("leaf_type_order", "[]"))
        if species_order != SPECIES_CLASS_ORDER:
            raise ValueError(f"Unexpected species class order: {species_order}")
        if leaf_order != LEAF_TYPE_ORDER:
            raise ValueError(f"Unexpected leaf-type order: {leaf_order}")

        stand_type = self._read_h5_string(labels["stand_type"][()])
        if stand_type not in STAND_TYPE_TO_ID:
            raise ValueError(
                f"Unknown labels/stand_type='{stand_type}'. "
                f"Expected one of {sorted(STAND_TYPE_TO_ID)}."
            )

        stand_type_id = STAND_TYPE_TO_ID[stand_type]
        leaf_target = labels["leaf_type_composition"][:].astype(np.float32)
        leaf_basal_area = labels["leaf_type_basal_area_m2_ha"][:].astype(np.float32)
        species_target = labels["species_composition"][:].astype(np.float32)
        species_basal_area = labels["species_basal_area_m2_ha"][:].astype(np.float32)
        total_basal_area = float(labels["total_expanded_basal_area_m2_ha"][()])
        nfi_record_count = int(labels["nfi_record_count"][()])
        dominant_species_class_id = int(labels["dominant_species_class_id"][()])
        dominant_species_proportion = float(labels["dominant_species_proportion"][()])
        is_pure = bool(labels["is_class_pure_80pct"][()])

        if leaf_target.shape != (LEAF_NUM_CLASSES,):
            raise ValueError(
                f"Expected leaf target dimension {LEAF_NUM_CLASSES}, "
                f"got shape {leaf_target.shape}."
            )

        if species_target.shape != (SPECIES_NUM_CLASSES,):
            raise ValueError(
                f"Expected species target dimension {SPECIES_NUM_CLASSES}, "
                f"got shape {species_target.shape}."
            )

        if leaf_basal_area.shape != (LEAF_NUM_CLASSES,):
            raise ValueError(f"Unexpected leaf basal-area shape: {leaf_basal_area.shape}.")
        if species_basal_area.shape != (SPECIES_NUM_CLASSES,):
            raise ValueError(f"Unexpected species basal-area shape: {species_basal_area.shape}.")

        for name, target in (("species", species_target), ("leaf", leaf_target)):
            if not np.isfinite(target).all() or not np.isclose(target.sum(), 1.0, atol=1e-5):
                raise ValueError(f"Invalid {name} composition: values must be finite and sum to one.")
        if total_basal_area <= 0 or nfi_record_count <= 0:
            raise ValueError("Basal area and NFI record count must both be positive.")
        for name, basal_area in (("species", species_basal_area), ("leaf", leaf_basal_area)):
            if not np.isfinite(basal_area).all() or np.any(basal_area < 0):
                raise ValueError(f"Invalid {name} basal-area vector.")
            if not np.isclose(basal_area.sum(), total_basal_area, rtol=1e-5, atol=1e-4):
                raise ValueError(f"{name} basal-area vector does not match the stored total.")
        if not np.allclose(
            species_target,
            species_basal_area / total_basal_area,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError("species_composition is inconsistent with species basal area.")
        if not np.allclose(
            leaf_target,
            leaf_basal_area / total_basal_area,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError("leaf_type_composition is inconsistent with leaf basal area.")
        if dominant_species_class_id != int(np.argmax(species_target)):
            raise ValueError("Stored dominant species class disagrees with species_composition.")
        if not np.isclose(dominant_species_proportion, species_target.max(), atol=1e-5):
            raise ValueError("Stored dominant species proportion disagrees with species_composition.")
        if is_pure != (stand_type == "pure"):
            raise ValueError("stand_type and is_class_pure_80pct disagree.")
        if is_pure != (dominant_species_proportion >= 0.80):
            raise ValueError("Purity labels do not follow the documented 0.80 threshold.")

        return {
            "stand_type_id": torch.tensor(stand_type_id, dtype=torch.long),
            "leaf_target": torch.from_numpy(leaf_target).float(),
            "species_target": torch.from_numpy(species_target).float(),
            "leaf_basal_area_m2_ha": torch.from_numpy(leaf_basal_area).float(),
            "species_basal_area_m2_ha": torch.from_numpy(species_basal_area).float(),
            "total_expanded_basal_area_m2_ha": torch.tensor(total_basal_area, dtype=torch.float32),
            "nfi_record_count": torch.tensor(nfi_record_count, dtype=torch.long),
            "dominant_species_class_id": torch.tensor(dominant_species_class_id, dtype=torch.long),
            "dominant_species_proportion": torch.tensor(dominant_species_proportion, dtype=torch.float32),
            "is_pure": torch.tensor(is_pure, dtype=torch.bool),
            "stand_type": stand_type,
        }

    @staticmethod
    def _decode_array(arr):
        return [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in arr
        ]

    @staticmethod
    def _read_h5_string(value):
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            return NFISiteDataset._read_h5_string(value.item())
        return str(value)

    @staticmethod
    def _to_python_value(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            return value.tolist()

        return value

    @staticmethod
    def _read_lidar_points(dataset):
        points = dataset[:]

        xyz = np.stack(
            [
                points["X"].astype(np.float32),
                points["Y"].astype(np.float32),
                points["Z"].astype(np.float32),
            ],
            axis=1,
        )

        classification = points["Classification"].astype(np.int64)

        return {
            "xyz": torch.from_numpy(xyz).float(),
            "classification": torch.from_numpy(classification).long(),
        }
