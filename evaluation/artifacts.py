import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data.label_schema import LABEL_SCHEMA_VERSION, LEAF_TYPE_ORDER, SPECIES_CLASS_ORDER
from training.data.cross_validation import cv_metadata
from evaluation.hierarchical import composition_metrics, stand_metrics


STAND_NAMES = ("mixed", "pure")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _validate_payload(payload):
    required = ("split", "site_id", "leaf_target", "leaf_prob", "species_target", "species_prob")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Evaluation payload is missing keys: {missing}")
    n_sites = len(payload["site_id"])
    expected = {
        "leaf_target": (n_sites, len(LEAF_TYPE_ORDER)),
        "leaf_prob": (n_sites, len(LEAF_TYPE_ORDER)),
        "species_target": (n_sites, len(SPECIES_CLASS_ORDER)),
        "species_prob": (n_sites, len(SPECIES_CLASS_ORDER)),
    }
    if "stand_target" in payload or "stand_prob" in payload:
        if "stand_target" not in payload or "stand_prob" not in payload:
            raise KeyError("stand_target and stand_prob must be supplied together.")
        expected.update({"stand_target": (n_sites,), "stand_prob": (n_sites, len(STAND_NAMES))})
    for key, shape in expected.items():
        actual = np.asarray(payload[key])
        if actual.shape != shape:
            raise ValueError(
                f"Expected {key} shape {shape} for {LABEL_SCHEMA_VERSION}, got {actual.shape}."
            )
        if key.endswith(("target", "prob")) and not np.isfinite(actual).all():
            raise ValueError(f"Non-finite values found in {key}.")
        if key.endswith(("target", "prob")) and actual.ndim == 2:
            if np.any(actual < -1e-6) or not np.allclose(actual.sum(axis=1), 1.0, atol=1e-4):
                raise ValueError(f"{key} must contain non-negative rows that sum to one.")
    if "stand_target" in payload:
        stand_target = np.asarray(payload["stand_target"])
        if not np.isin(stand_target, np.arange(len(STAND_NAMES))).all():
            raise ValueError(f"stand_target must use zero-based classes {STAND_NAMES}.")


def concat_payloads(payloads, split="overall"):
    if not payloads:
        raise ValueError("Cannot concatenate an empty payload list.")
    out = {"split": split, "site_id": [], "source_split": []}
    array_keys = (
        "stand_target",
        "stand_prob",
        "leaf_target",
        "leaf_prob",
        "species_target",
        "species_prob",
        "is_pure",
        "dominant_species_class_id",
        "dominant_species_proportion",
        "nfi_record_count",
        "total_expanded_basal_area_m2_ha",
        "outer_fold",
    )
    for payload in payloads:
        out["site_id"].extend(payload["site_id"])
        out["source_split"].extend([payload["split"]] * len(payload["site_id"]))
    for key in array_keys:
        values = [np.asarray(payload[key]) for payload in payloads if key in payload]
        if values:
            out[key] = np.concatenate(values, axis=0)
    return out


def metric_row(payload, cfg):
    metrics = {}
    if "stand_target" in payload and "stand_prob" in payload:
        metrics.update(stand_metrics(payload["stand_target"], payload["stand_prob"], prefix="stand_"))
    metrics.update(composition_metrics(
        payload["leaf_target"], payload["leaf_prob"], prefix="leaf_", dominant_top_n=(1,)
    ))
    dominant_top_n = tuple(getattr(cfg.metrics, "dominant_top_n", (1, 2, 3)))
    metrics.update(composition_metrics(
        payload["species_target"],
        payload["species_prob"],
        prefix="species_",
        dominant_top_n=dominant_top_n,
    ))
    metrics = {
        key: value for key, value in metrics.items()
        if not key.endswith("_confusion_matrix") and not key.endswith("_labels")
    }
    return {"split": payload["split"], "n_sites": len(payload["site_id"]), **metrics}


def predictions_dataframe(payloads):
    rows = []
    for payload in payloads:
        for index, site_id in enumerate(payload["site_id"]):
            row = {
                "split": payload["split"],
                "site_id": site_id,
                "leaf_true_dominant": int(np.argmax(payload["leaf_target"][index])),
                "leaf_pred_dominant": int(np.argmax(payload["leaf_prob"][index])),
                "species_true_dominant": int(np.argmax(payload["species_target"][index])),
                "species_pred_dominant": int(np.argmax(payload["species_prob"][index])),
            }
            if "stand_target" in payload and "stand_prob" in payload:
                row["stand_true"] = int(payload["stand_target"][index])
                row["stand_pred"] = int(np.argmax(payload["stand_prob"][index]))
            for key in (
                "is_pure",
                "dominant_species_class_id",
                "dominant_species_proportion",
                "nfi_record_count",
                "total_expanded_basal_area_m2_ha",
                "outer_fold",
            ):
                if key in payload:
                    value = np.asarray(payload[key])[index]
                    row[key] = value.item() if isinstance(value, np.generic) else value
            if "stand_prob" in payload:
                for class_id, name in enumerate(STAND_NAMES):
                    row[f"stand_prob_{class_id}_{_slug(name)}"] = payload["stand_prob"][index, class_id]
            for class_id, name in enumerate(LEAF_TYPE_ORDER):
                slug = _slug(name)
                row[f"leaf_true_{class_id}_{slug}"] = payload["leaf_target"][index, class_id]
                row[f"leaf_pred_{class_id}_{slug}"] = payload["leaf_prob"][index, class_id]
            for class_id, name in enumerate(SPECIES_CLASS_ORDER):
                slug = _slug(name)
                row[f"species_true_{class_id}_{slug}"] = payload["species_target"][index, class_id]
                row[f"species_pred_{class_id}_{slug}"] = payload["species_prob"][index, class_id]
            rows.append(row)
    return pd.DataFrame(rows)


def per_class_metrics(payloads):
    rows = []
    for payload in payloads:
        for task, names in (("leaf", LEAF_TYPE_ORDER), ("species", SPECIES_CLASS_ORDER)):
            target = np.asarray(payload[f"{task}_target"], dtype=np.float64)
            pred = np.asarray(payload[f"{task}_prob"], dtype=np.float64)
            for class_id, name in enumerate(names):
                true = target[:, class_id]
                estimate = pred[:, class_id]
                residual = true - estimate
                denominator = np.sum((true - true.mean()) ** 2)
                r2 = 1.0 - np.sum(residual ** 2) / denominator if denominator > 0 else np.nan
                rows.append({
                    "split": payload["split"],
                    "task": task,
                    "class_id": class_id,
                    "class_name": name,
                    "n_sites": len(true),
                    "n_present_gt_0": int(np.sum(true > 0)),
                    "n_present_ge_0_05": int(np.sum(true >= 0.05)),
                    "mean_target": float(true.mean()),
                    "mean_prediction": float(estimate.mean()),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual ** 2))),
                    "r2": float(r2),
                })
    return pd.DataFrame(rows)


def confusion_records(payloads):
    records = []
    matrices = {}
    for payload in payloads:
        split = payload["split"]
        tasks = [
            ("leaf", LEAF_TYPE_ORDER, np.asarray(payload["leaf_target"]), np.asarray(payload["leaf_prob"])),
            ("species", SPECIES_CLASS_ORDER, np.asarray(payload["species_target"]), np.asarray(payload["species_prob"])),
        ]
        if "stand_target" in payload and "stand_prob" in payload:
            tasks.insert(0, (
                "stand", STAND_NAMES, np.asarray(payload["stand_target"]), np.asarray(payload["stand_prob"])
            ))
        for task, names, target, probability in tasks:
            true_id = target.astype(np.int64) if target.ndim == 1 else np.argmax(target, axis=1)
            pred_id = np.argmax(probability, axis=1)
            labels = np.arange(len(names), dtype=np.int64)
            matrix = confusion_matrix(true_id, pred_id, labels=labels)
            matrices[f"{split}__{task}__counts"] = matrix.astype(np.int64)
            row_sum = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                row_sum,
                out=np.zeros_like(matrix, dtype=np.float64),
                where=row_sum > 0,
            )
            matrices[f"{split}__{task}__row_normalized"] = normalized
            for true_class, true_name in enumerate(names):
                for pred_class, pred_name in enumerate(names):
                    records.append({
                        "split": split,
                        "task": task,
                        "true_class_id": true_class,
                        "true_class_name": true_name,
                        "pred_class_id": pred_class,
                        "pred_class_name": pred_name,
                        "count": int(matrix[true_class, pred_class]),
                        "row_normalized": float(normalized[true_class, pred_class]),
                    })
    return pd.DataFrame(records), matrices


def _plot_confusion_matrix(matrix, normalized, names, split, task):
    import matplotlib.pyplot as plt

    size = max(5.0, 0.62 * len(names) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size, size), dpi=220)
    for axis, values, title, cmap in (
        (axes[0], matrix, "Counts", "Greens"),
        (axes[1], normalized, "Row normalized", "Blues"),
    ):
        image = axis.imshow(values, cmap=cmap, vmin=0, vmax=(1 if title.startswith("Row") else None))
        axis.set_title(title)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Observed")
        axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(names)), names)
        threshold = float(values.max()) * 0.55 if values.size else 0.0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                text = f"{values[row, col]:.2f}" if title.startswith("Row") else str(int(values[row, col]))
                axis.text(col, row, text, ha="center", va="center", fontsize=6,
                          color="white" if values[row, col] > threshold else "black")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"{task.capitalize()} confusion matrix ({split})")
    fig.tight_layout()
    return fig


def save_evaluation_artifacts(payloads, cfg, output_dir, source=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = list(payloads)
    for payload in payloads:
        _validate_payload(payload)
    report_payloads = payloads + ([concat_payloads(payloads)] if len(payloads) > 1 else [])

    metrics = [metric_row(payload, cfg) for payload in report_payloads]
    pd.DataFrame(metrics).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "source": str(source) if source else None,
                "label_schema": LABEL_SCHEMA_VERSION,
                "leaf_class_names": LEAF_TYPE_ORDER,
                "species_class_names": SPECIES_CLASS_ORDER,
                "cross_validation": cv_metadata(cfg),
                "results": _jsonable(metrics),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    predictions_dataframe(payloads).to_csv(output_dir / "predictions.csv", index=False)
    per_class_metrics(report_payloads).to_csv(output_dir / "per_class_metrics.csv", index=False)

    confusion_df, matrices = confusion_records(report_payloads)
    confusion_df.to_csv(output_dir / "confusion_matrices.csv", index=False)
    np.savez_compressed(
        output_dir / "confusion_matrices.npz",
        **matrices,
        stand_class_names=np.asarray(STAND_NAMES),
        leaf_class_names=np.asarray(LEAF_TYPE_ORDER),
        species_class_names=np.asarray(SPECIES_CLASS_ORDER),
    )
    for payload in report_payloads:
        tasks = ["leaf", "species"]
        if "stand_target" in payload and "stand_prob" in payload:
            tasks.insert(0, "stand")
        for task in tasks:
            fig = _plot_confusion_matrix(
                matrices[f"{payload['split']}__{task}__counts"],
                matrices[f"{payload['split']}__{task}__row_normalized"],
                STAND_NAMES if task == "stand" else LEAF_TYPE_ORDER if task == "leaf" else SPECIES_CLASS_ORDER,
                payload["split"],
                task,
            )
            fig.savefig(output_dir / f"confusion_matrix_{task}_{payload['split']}.png", bbox_inches="tight")
            fig.savefig(output_dir / f"confusion_matrix_{task}_{payload['split']}.pdf", bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)

    combined = concat_payloads(payloads, split="all_evaluated")
    raw = {
        "label_schema": np.asarray(LABEL_SCHEMA_VERSION),
        "site_id": np.asarray(combined["site_id"], dtype=str),
        "split": np.asarray(combined["source_split"], dtype=str),
        "leaf_target": combined["leaf_target"],
        "leaf_prob": combined["leaf_prob"],
        "species_target": combined["species_target"],
        "species_prob": combined["species_prob"],
        "stand_class_names": np.asarray(STAND_NAMES),
        "leaf_class_names": np.asarray(LEAF_TYPE_ORDER),
        "species_class_names": np.asarray(SPECIES_CLASS_ORDER),
    }
    if "stand_target" in combined and "stand_prob" in combined:
        raw["stand_target"] = combined["stand_target"]
        raw["stand_prob"] = combined["stand_prob"]
    for key in (
        "is_pure",
        "dominant_species_class_id",
        "dominant_species_proportion",
        "nfi_record_count",
        "total_expanded_basal_area_m2_ha",
        "outer_fold",
    ):
        if key in combined:
            raw[key] = combined[key]
    np.savez_compressed(output_dir / "raw_predictions.npz", **raw)
    return metrics
