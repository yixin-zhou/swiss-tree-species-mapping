"""Evaluate each outer-fold model on its held-out fold and summarize CV metrics."""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

from src.utils.devices import select_gpus
from src.config_defaults import NUM_FOLDS
from src.utils.config import config_path, default_cv_output, load_training_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_fold_jobs(root, output_dir=None, config="configs/main.yaml", overrides=None,
                    batch_size=None, num_workers=None, device=None):
    """Describe five held-out evaluations without reading data or checkpoints."""
    root = config_path(root)
    results = config_path(output_dir) if output_dir else root
    return [SimpleNamespace(
        checkpoint=str(root / f"fold_{fold}" / "checkpoints/last.pt"),
        config=config, splits=["test"], outer_fold=None,
        output_dir=str(results / f"fold_{fold}" / "evaluation"),
        overrides=list(overrides or []), batch_size=batch_size,
        num_workers=num_workers, device=device, seed=42,
    ) for fold in range(NUM_FOLDS)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Outputs: per-fold evaluation files, oof_evaluation/fold_metrics.csv, "
        "fold_metrics_mean_std.csv and pooled OOF metrics. "
        "Requires authorized training data and all five final checkpoints."
    ))
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--experiment-root", type=Path, help="Five-fold training directory; defaults to the train.py output.")
    parser.add_argument("--output-dir", type=Path, help="Optional root for evaluation outputs.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--gpus", metavar="ID", help="One visible GPU index, e.g. 1. CV folds are evaluated sequentially.")
    parser.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Show checkpoint and output paths without loading data or models.")
    args = parser.parse_args()
    try:
        indices, visibility = select_gpus(args.gpus)
        if indices is not None:
            if len(indices) != 1:
                raise ValueError("Evaluation uses one GPU; supply a single index, e.g. --gpus 1.")
            if args.device == "cpu":
                raise ValueError("--gpus cannot be combined with --device cpu.")
            args.device = "cuda"
    except ValueError as exc:
        parser.error(str(exc))
    cfg = load_training_config(config=args.config, overrides=args.overrides)
    root = config_path(args.experiment_root) if args.experiment_root else default_cv_output(cfg)
    results = config_path(args.output_dir) if args.output_dir else root
    jobs = build_fold_jobs(root, results, args.config, args.overrides,
                           args.batch_size, args.num_workers, args.device)
    if indices is not None:
        print(f"GPU: {args.gpus}", flush=True)
    for fold, job in enumerate(jobs):
        print(f"fold={fold}: {job.checkpoint} -> {job.output_dir}", flush=True)
    print(f"CV summary: {results / 'oof_evaluation'}", flush=True)
    if args.dry_run:
        return
    missing = [job.checkpoint for job in jobs if not Path(job.checkpoint).is_file()]
    if missing:
        raise FileNotFoundError("Missing final CV checkpoints. Run scripts/train.py first:\n" + "\n".join(missing))

    if visibility is not None:
        # Set visibility before importing PyTorch or initializing CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = visibility
    from evaluation.evaluate import evaluate_checkpoint
    from evaluation.aggregate_results import aggregate_results

    # Dataset and saved weight paths use the same repository root as training.
    os.chdir(PROJECT_ROOT)
    dataset_cfg = None
    for fold, job in enumerate(jobs):
        evaluated_cfg = evaluate_checkpoint(job, expected_fold=fold)
        if dataset_cfg is None:
            dataset_cfg = evaluated_cfg
        elif Path(evaluated_cfg.path.index).resolve() != Path(dataset_cfg.path.index).resolve():
            raise ValueError("All five checkpoints must evaluate the same dataset index.")
    aggregate_results(SimpleNamespace(
        experiment_root=str(results), output_dir=str(results / "oof_evaluation"),
        expected_sites=int(dataset_cfg.cross_validation.n_train) + int(dataset_cfg.cross_validation.n_test),
        config=args.config, overrides=args.overrides,
    ), dataset_cfg=dataset_cfg)


if __name__ == "__main__":
    main()
