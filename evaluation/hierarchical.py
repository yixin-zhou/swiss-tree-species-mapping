import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from evaluation.metrics import TreeSpeciesMappingMetrics


def _as_numpy(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _nan_row(prefix, names):
    return {f"{prefix}{name}": np.nan for name in names}


def sanitize_metrics(metrics):
    clean = {}
    for key, value in metrics.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            clean[key] = None
        else:
            clean[key] = value
    return clean


def stand_metrics(target, prob, prefix="stand_"):
    target = _as_numpy(target).astype(np.int64)
    prob = _as_numpy(prob)
    pred = np.argmax(prob, axis=1)
    labels = [0, 1]

    return {
        f"{prefix}n_sites": int(target.shape[0]),
        f"{prefix}accuracy": accuracy_score(target, pred),
        f"{prefix}macro_f1": f1_score(
            target,
            pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        f"{prefix}confusion_matrix": confusion_matrix(target, pred, labels=labels),
        f"{prefix}labels": labels,
    }


def composition_metrics(
    target,
    pred,
    prefix,
    dominant_top_n=(1, 2, 3),
    other_index=None,
    exclude_other_dominant=False,
):
    target = _as_numpy(target)
    pred = _as_numpy(pred)

    if target.shape[0] == 0:
        keys = [
            "n_sites",
            "rmse_overall",
            "rmse_mean_per_site",
            "rmse_mean_per_class",
            "r2_overall",
            "r2_macro",
            "bray_curtis_mean_per_site",
            "dominant_accuracy",
            "dominant_macro_f1",
            "dominant_weighted_f1",
            "dominant_top1_accuracy",
            "dominant_top2_accuracy",
            "dominant_top3_accuracy",
        ]
        out = _nan_row(prefix, keys)
        out[f"{prefix}n_sites"] = 0
        out[f"{prefix}dominant_confusion_matrix"] = None
        out[f"{prefix}dominant_labels"] = None
        return out

    evaluator = TreeSpeciesMappingMetrics(target, pred)
    raw = evaluator.get_all_metrics(
        dominant_top_n=tuple(dominant_top_n),
        other_index=other_index,
        exclude_other_dominant=exclude_other_dominant,
    )
    dominant = raw["dominant_species"]

    out = {
        f"{prefix}n_sites": int(target.shape[0]),
        f"{prefix}rmse_overall": raw["rmse_overall"],
        f"{prefix}rmse_mean_per_site": raw["rmse_mean_per_site"],
        f"{prefix}rmse_mean_per_class": raw["rmse_mean_per_species"],
        f"{prefix}mae_overall": raw["mae_overall"],
        f"{prefix}mae_mean_per_class": raw["mae_mean_per_species"],
        f"{prefix}r2_overall": raw["r2_overall"],
        f"{prefix}r2_macro": raw["r2_macro"],
        f"{prefix}bray_curtis_mean_per_site": raw["bray_curtis_mean_per_site"],
        f"{prefix}dominant_accuracy": dominant["accuracy"],
        f"{prefix}dominant_cohen_kappa": dominant["cohen_kappa"],
        f"{prefix}dominant_macro_f1": dominant["macro_f1"],
        f"{prefix}dominant_weighted_f1": dominant["weighted_f1"],
        f"{prefix}dominant_confusion_matrix": dominant["confusion_matrix"],
        f"{prefix}dominant_labels": dominant["labels"],
    }

    for n in tuple(dominant_top_n):
        out[f"{prefix}dominant_top{n}_accuracy"] = dominant[f"top{n}_accuracy"]

    return out


def hierarchical_metrics(
    payload,
    cfg,
    prefix="",
    include_species=True,
    include_matrices=False,
):
    dominant_top_n = tuple(getattr(cfg.metrics, "dominant_top_n", [1, 2, 3]))
    metrics = {}

    stand = stand_metrics(
        payload["stand_target"],
        payload["stand_prob"],
        prefix=f"{prefix}stand_",
    )
    metrics.update(stand)

    leaf = composition_metrics(
        payload["leaf_target"],
        payload["leaf_prob"],
        prefix=f"{prefix}leaf_",
        dominant_top_n=(1,),
        other_index=None,
        exclude_other_dominant=False,
    )
    metrics.update(leaf)

    if include_species:
        species_target = _as_numpy(payload["species_target"])
        species_prob = _as_numpy(payload["species_prob"])
        species = composition_metrics(
            species_target,
            species_prob,
            prefix=f"{prefix}species_",
            dominant_top_n=dominant_top_n,
            other_index=None,
            exclude_other_dominant=False,
        )
        species[f"{prefix}species_n_total_sites"] = int(species_target.shape[0])
        metrics.update(species)

    if not include_matrices:
        metrics = {
            key: value
            for key, value in metrics.items()
            if not key.endswith("_confusion_matrix") and not key.endswith("_labels")
        }

    return sanitize_metrics(metrics)
