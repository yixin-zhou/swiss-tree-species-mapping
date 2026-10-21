import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DistributedSampler, Sampler, Subset


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)


def _dataset_h5_path(dataset, site_id):
    return Path(dataset.hdf5_root) / f"{site_id}.h5"


def _read_species_composition(h5_path):
    with h5py.File(h5_path, "r") as f:
        if "labels/species_composition" not in f:
            raise KeyError(f"Missing labels/species_composition in {h5_path}")
        return f["labels/species_composition"][:].astype(np.float32)


def compute_rare_class_weights(
    dataset,
    rare_class_ids,
    rare_weight=3.0,
    normal_weight=1.0,
    mode="presence",
    presence_threshold=0.05,
):
    """Up-weight rare-species sites for the WeightedRandomSampler.

    mode:
      - "presence": a site is rare if ANY rare species reaches the configured
        basal-area composition threshold.
      - "dominant": a site is rare only if its DOMINANT species
        (argmax of the per-site composition) is a rare class. This targets
        the sites the model most needs to get right for rare-species recall.
    """
    if isinstance(dataset, ConcatDataset):
        weights = [
            compute_rare_class_weights(
                child,
                rare_class_ids=rare_class_ids,
                rare_weight=rare_weight,
                normal_weight=normal_weight,
                mode=mode,
                presence_threshold=presence_threshold,
            )
            for child in dataset.datasets
        ]
        return torch.cat(weights, dim=0)

    if isinstance(dataset, Subset):
        base_weights = compute_rare_class_weights(
            dataset.dataset,
            rare_class_ids=rare_class_ids,
            rare_weight=rare_weight,
            normal_weight=normal_weight,
            mode=mode,
            presence_threshold=presence_threshold,
        )
        return base_weights[dataset.indices]

    rare_class_ids = np.asarray(rare_class_ids, dtype=np.int64)
    if rare_class_ids.size == 0:
        raise ValueError("sampler.rare_class_ids must contain at least one class ID.")
    rare_set = set(rare_class_ids.tolist())
    weights = []

    for _, row in dataset.index.iterrows():
        site_id = str(row["site_id"])
        h5_path = _dataset_h5_path(dataset, site_id)
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found for site_id={site_id}: {h5_path}")
        composition = _read_species_composition(h5_path)
        if rare_class_ids.min() < 0 or rare_class_ids.max() >= composition.shape[0]:
            raise ValueError(
                f"rare_class_ids={rare_class_ids.tolist()} are invalid for "
                f"a {composition.shape[0]}-class composition."
            )
        if mode == "dominant":
            is_rare = int(np.argmax(composition)) in rare_set
        else:  # "presence"
            is_rare = bool(np.any(composition[rare_class_ids] >= float(presence_threshold)))
        weights.append(rare_weight if is_rare else normal_weight)

    return torch.tensor(weights, dtype=torch.double)


def filter_dataset_for_composition(dataset, cfg):
    raise ValueError(
        "Coverage-filtered regimes are incompatible with the complete 11-class "
        "basal-area labels. Use training.regime='single_stage_allsites'."
    )


class DistributedWeightedSampler(Sampler):
    def __init__(
        self,
        dataset,
        weights,
        num_replicas=None,
        rank=None,
        replacement=True,
        num_samples=None,
        seed=0,
        drop_last=False,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank={rank}, num_replicas={num_replicas}")

        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

        if len(self.weights) != len(self.dataset):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match dataset length ({len(self.dataset)})."
            )

        if num_samples is None:
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:
                self.num_samples = math.ceil(
                    (len(self.dataset) - self.num_replicas) / self.num_replicas
                )
            else:
                self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        else:
            self.num_samples = int(num_samples)

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.replacement:
            indices = torch.multinomial(
                self.weights,
                self.total_size,
                replacement=True,
                generator=generator,
            ).tolist()
        else:
            sampled_size = min(self.total_size, len(self.dataset))
            indices = torch.multinomial(
                self.weights,
                sampled_size,
                replacement=False,
                generator=generator,
            ).tolist()

            if len(indices) < self.total_size:
                indices += indices[: self.total_size - len(indices)]

        indices = indices[self.rank : self.total_size : self.num_replicas]

        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"Sampler produced {len(indices)} samples, expected {self.num_samples}."
            )

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


def build_sampler(
    dataset,
    cfg,
    is_train=True,
    distributed=False,
    rank=None,
    world_size=None,
    seed=0,
):
    sampler_cfg = _cfg_get(cfg, "sampler", {})

    if not is_train:
        if distributed:
            return DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        return None

    # The paper always uses rare-site weighted sampling, including one-GPU runs.
    weights = compute_rare_class_weights(
        dataset=dataset,
        rare_class_ids=_cfg_get(sampler_cfg, "rare_class_ids"),
        rare_weight=_cfg_get(sampler_cfg, "rare_species_weight", 2.0),
        normal_weight=_cfg_get(sampler_cfg, "normal_weight", 1.0),
        mode=_cfg_get(sampler_cfg, "rare_mode", "presence"),
        presence_threshold=_cfg_get(sampler_cfg, "presence_threshold", 0.05),
    )

    return DistributedWeightedSampler(
        dataset=dataset,
        weights=weights,
        num_replicas=world_size,
        rank=rank,
        replacement=_cfg_get(sampler_cfg, "replacement", True),
        num_samples=_cfg_get(sampler_cfg, "num_samples", None),
        seed=seed,
        drop_last=_cfg_get(sampler_cfg, "drop_last", False),
    )
