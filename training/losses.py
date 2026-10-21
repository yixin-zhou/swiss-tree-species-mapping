import torch
import torch.nn.functional as F
from torch import nn

from src.data.label_schema import (
    BROADLEAF_SPECIES_INDICES,
    LEAF_NUM_CLASSES,
    OTHER_SPECIES_INDEX,
    SPECIES_NUM_CLASSES,
)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        return cfg.get(key, default)

    if hasattr(cfg, "get"):
        return cfg.get(key, default)

    return getattr(cfg, key, default)


def _is_enabled(cfg, default=False):
    return bool(_cfg_get(cfg, "enabled", default))


def _loss_weight(cfg, default=1.0):
    return float(_cfg_get(cfg, "weight", default))


def leaf_species_consistency_loss(species_prob, leaf_prob, reduction="mean", eps=1e-6):
    """Distance from predicted broadleaf share to its Other-aware feasible interval.

    The residual Other species class contains both broadleaf and conifer taxa.
    Consequently, the leaf-level broadleaf prediction is feasible anywhere
    between the known broadleaf mass and known broadleaf plus Other mass.
    """
    if species_prob.shape[-1] != SPECIES_NUM_CLASSES:
        raise ValueError(
            f"Expected {SPECIES_NUM_CLASSES} species classes, got {species_prob.shape[-1]}."
        )
    if leaf_prob.shape[-1] != LEAF_NUM_CLASSES:
        raise ValueError(f"Expected {LEAF_NUM_CLASSES} leaf classes, got {leaf_prob.shape[-1]}.")
    if species_prob.shape[:-1] != leaf_prob.shape[:-1]:
        raise ValueError(
            "species_prob and leaf_prob must have matching leading dimensions, "
            f"got {species_prob.shape[:-1]} and {leaf_prob.shape[:-1]}."
        )

    leaf_prob = leaf_prob.to(device=species_prob.device, dtype=species_prob.dtype)
    known_broadleaf = species_prob[..., BROADLEAF_SPECIES_INDICES].sum(dim=-1)
    other = species_prob[..., OTHER_SPECIES_INDEX]
    predicted_broadleaf = leaf_prob[..., 0]

    below_interval = F.relu(known_broadleaf - predicted_broadleaf - eps)
    above_interval = F.relu(predicted_broadleaf - known_broadleaf - other - eps)
    loss = below_interval + above_interval

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction='{reduction}'.")


def get_composition_prediction(outputs):
    if "species_prob" in outputs:
        return outputs["species_prob"]
    if "y_comp" in outputs:
        return outputs["y_comp"]
    if "composition" in outputs:
        return outputs["composition"]
    if "logits" in outputs:
        return torch.softmax(outputs["logits"], dim=-1)
    if "composition_logits" in outputs:
        return torch.softmax(outputs["composition_logits"], dim=-1)
    raise KeyError("Cannot find a composition prediction in model outputs.")


def soft_target_kl_loss(logits, target, reduction="mean", eps=1e-8):
    target = target.to(device=logits.device, dtype=logits.dtype)
    target = target.clamp_min(eps)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_prob = F.log_softmax(logits, dim=-1)
    loss = F.kl_div(log_prob, target, reduction="none").sum(dim=-1)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction='{reduction}'.")


def composition_l1_loss(prediction, target, reduction="mean"):
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    loss = torch.abs(prediction - target).sum(dim=-1)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction='{reduction}'.")


def entropy_loss(prob, eps=1e-8, reduction="mean"):
    prob = prob.clamp_min(eps)
    loss = -(prob * prob.log()).sum(dim=-1)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction='{reduction}'.")


def masked_mean(values, mask):
    mask = mask.to(device=values.device).bool()
    if not mask.any():
        return values.sum() * 0.0
    return values[mask].mean()


class HierarchicalLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = _cfg_get(cfg, "losses", cfg)
        # Label smoothing is applied ONLY to the KL targets (curbs the
        # overconfidence that makes val KL explode). It is deliberately NOT
        # applied to the L1 targets, so species_l1 stays == 2 * Bray-Curtis.
        self.label_smoothing = float(_cfg_get(self.loss_cfg, "label_smoothing", 0.0))

    def _smooth(self, target):
        eps = self.label_smoothing
        if eps <= 0.0:
            return target
        k = target.shape[-1]
        return (1.0 - eps) * target + eps / k

    def forward(self, outputs, batch):
        total_loss = None
        logs = {}

        total_loss = self._add_stand_ce(outputs, batch, total_loss, logs)
        total_loss = self._add_leaf_kl(outputs, batch, total_loss, logs)
        total_loss = self._add_leaf_l1(outputs, batch, total_loss, logs)
        total_loss = self._add_species_kl(outputs, batch, total_loss, logs)
        total_loss = self._add_species_l1(outputs, batch, total_loss, logs)
        total_loss = self._add_leaf_species_consistency(outputs, batch, total_loss, logs)
        total_loss = self._add_leaf_entropy_pure(outputs, batch, total_loss, logs)
        total_loss = self._add_species_entropy_pure(outputs, batch, total_loss, logs)

        if total_loss is None:
            raise ValueError(
                "No hierarchical loss was enabled. Check cfg.losses and set at least one enabled=true."
            )

        logs["loss"] = total_loss.detach()
        return total_loss, logs

    def _add_loss(self, total_loss, logs, name, value, weight):
        weighted_value = float(weight) * value
        logs[name] = value.detach()
        logs[f"{name}_weighted"] = weighted_value.detach()

        if total_loss is None:
            return weighted_value

        return total_loss + weighted_value

    def _add_stand_ce(self, outputs, batch, total_loss, logs):
        cfg = _cfg_get(self.loss_cfg, "stand_ce")
        if not _is_enabled(cfg):
            return total_loss

        target = batch["stand_type_id"].to(outputs["stand_logits"].device).long()
        loss = F.cross_entropy(outputs["stand_logits"], target)
        return self._add_loss(total_loss, logs, "stand_ce", loss, _loss_weight(cfg))

    def _add_leaf_kl(self, outputs, batch, total_loss, logs):
        cfg = _cfg_get(self.loss_cfg, "leaf_kl")
        if not _is_enabled(cfg):
            return total_loss

        loss = soft_target_kl_loss(outputs["leaf_logits"], self._smooth(batch["leaf_target"]))
        return self._add_loss(total_loss, logs, "leaf_kl", loss, _loss_weight(cfg))

    def _add_leaf_l1(self, outputs, batch, total_loss, logs):
        cfg = _cfg_get(self.loss_cfg, "leaf_l1")
        if not _is_enabled(cfg):
            return total_loss

        loss = composition_l1_loss(outputs["leaf_prob"], batch["leaf_target"])
        return self._add_loss(total_loss, logs, "leaf_l1", loss, _loss_weight(cfg))

    def _add_species_kl(self, outputs, batch, total_loss, logs):
        cfg = _cfg_get(self.loss_cfg, "species_kl")
        if not _is_enabled(cfg):
            return total_loss

        per_sample = soft_target_kl_loss(
            outputs["species_logits"],
            self._smooth(batch["species_target"]),
            reduction="none",
        )
        loss = per_sample.mean()
        n_used = torch.tensor(per_sample.numel(), device=per_sample.device)

        logs["species_kl_n"] = n_used.detach()
        return self._add_loss(total_loss, logs, "species_kl", loss, _loss_weight(cfg))

    def _add_species_l1(self, outputs, batch, total_loss, logs):
        cfg = _cfg_get(self.loss_cfg, "species_l1")
        if not _is_enabled(cfg):
            return total_loss

        per_sample = composition_l1_loss(
            outputs["species_prob"],
            batch["species_target"],
            reduction="none",
        )
        loss = per_sample.mean()
        n_used = torch.tensor(per_sample.numel(), device=per_sample.device)

        logs["species_l1_n"] = n_used.detach()
        return self._add_loss(total_loss, logs, "species_l1", loss, _loss_weight(cfg))

    def _add_leaf_species_consistency(self, outputs, batch, total_loss, logs):
        cfg = _cfg_get(self.loss_cfg, "leaf_species_consistency")
        if not _is_enabled(cfg):
            return total_loss

        per_sample = leaf_species_consistency_loss(
            outputs["species_prob"],
            outputs["leaf_prob"],
            reduction="none",
        )
        loss = per_sample.mean()
        logs["leaf_species_consistency_violation_rate"] = (
            (per_sample > 0).to(per_sample.dtype).mean().detach()
        )
        logs["species_other_pred_mean"] = (
            outputs["species_prob"][..., OTHER_SPECIES_INDEX].mean().detach()
        )
        if "species_target" in batch:
            logs["species_other_target_mean"] = (
                batch["species_target"]
                .to(outputs["species_prob"].device)[..., OTHER_SPECIES_INDEX]
                .mean()
                .detach()
            )
        return self._add_loss(total_loss, logs, "leaf_species_consistency", loss, _loss_weight(cfg))

    def _add_leaf_entropy_pure(self, outputs, batch, total_loss, logs):
        cfg = _cfg_get(self.loss_cfg, "leaf_entropy_pure")
        if not _is_enabled(cfg):
            return total_loss

        per_sample = entropy_loss(outputs["leaf_prob"], reduction="none")
        mask = batch["is_pure"]
        loss = masked_mean(per_sample, mask)
        logs["leaf_entropy_pure_n"] = mask.to(per_sample.device).bool().sum().detach()
        return self._add_loss(total_loss, logs, "leaf_entropy_pure", loss, _loss_weight(cfg))

    def _add_species_entropy_pure(self, outputs, batch, total_loss, logs):
        cfg = _cfg_get(self.loss_cfg, "species_entropy_pure")
        if not _is_enabled(cfg):
            return total_loss

        per_sample = entropy_loss(outputs["species_prob"], reduction="none")
        mask = batch["is_pure"]
        loss = masked_mean(per_sample, mask)
        logs["species_entropy_pure_n"] = mask.to(per_sample.device).bool().sum().detach()
        return self._add_loss(total_loss, logs, "species_entropy_pure", loss, _loss_weight(cfg))


MultiTaskLoss = HierarchicalLoss


def build_loss(cfg):
    return HierarchicalLoss(cfg)
