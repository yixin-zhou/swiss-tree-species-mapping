#!/usr/bin/env python3
"""Train the five outer folds of the reference model."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from src.utils.devices import select_gpus
from src.config_defaults import NUM_FOLDS
from src.utils.config import load_training_config, config_path, default_cv_output


def build_commands(config, output_dir, num_gpus=1, overrides=None):
    """Build an explicit plan; this function does not create files or launch jobs."""
    overrides = list(overrides or [])
    cfg = load_training_config(config=config, overrides=overrides)
    if num_gpus < 1 or int(cfg.dataloader.batch_size) * num_gpus != 4:
        raise ValueError('The reference protocol uses global batch size 4; set per-GPU dataloader.batch_size accordingly.')
    config = str(config_path(config))
    output_dir = Path(output_dir).resolve()
    commands = []
    extra = ['--set', *overrides] if overrides else []
    for fold in range(NUM_FOLDS):
        fold_dir = output_dir / f'fold_{fold}'
        launch = [sys.executable]
        if num_gpus > 1:
            launch += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={num_gpus}']
        commands.append(launch + [str(PROJECT_ROOT/'training/trainer.py'), '--config', config,
                                  '--outer-fold', str(fold), '--run-dir', str(fold_dir), *extra])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/main.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--gpus', metavar='IDS', help='Visible GPU indices, e.g. 1 or 0,2. Defaults to the first visible GPU. Global batch size remains 4.')
    parser.add_argument('--set', dest='overrides', nargs='+', default=[], metavar='KEY=VALUE')
    parser.add_argument('--dry-run', action='store_true', help='Print all commands without loading data or running training.')
    args = parser.parse_args()
    try:
        indices, visibility = select_gpus(args.gpus)
        num_gpus = len(indices) if indices is not None else 1
        if 4 % num_gpus:
            raise ValueError("The reference batch size 4 supports 1, 2 or 4 GPUs.")
        per_gpu_batch = 4 // num_gpus
        explicit = OmegaConf.from_dotlist(args.overrides)
        requested_batch = OmegaConf.select(explicit, "dataloader.batch_size")
        if requested_batch is not None and int(requested_batch) != per_gpu_batch:
            raise ValueError(f"With {num_gpus} GPU(s), dataloader.batch_size must be {per_gpu_batch}.")
        if indices is not None and OmegaConf.select(explicit, "device") not in (None, "cuda"):
            raise ValueError("--gpus cannot be combined with a non-CUDA device override.")
    except ValueError as exc:
        parser.error(str(exc))
    overrides = [*args.overrides, f'dataloader.batch_size={per_gpu_batch}']
    if indices is not None:
        overrides.append('device=cuda')
    cfg = load_training_config(config=args.config, overrides=overrides)
    run_dir = config_path(args.output_dir) if args.output_dir else default_cv_output(cfg)
    commands = build_commands(args.config, run_dir, num_gpus, overrides)
    launch_env = os.environ.copy()
    if visibility is not None:
        launch_env['CUDA_VISIBLE_DEVICES'] = visibility
    print(f"GPUs: {args.gpus or 'first visible'}; per-GPU batch size: {per_gpu_batch}; global batch size: 4", flush=True)
    if not args.dry_run:
        if run_dir.exists():
            raise FileExistsError(f'Run directory already exists: {run_dir}. Choose a new --output-dir.')
        run_dir.mkdir(parents=True, exist_ok=False)
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, env=launch_env, check=True)
    if not args.dry_run:
        print(f'Five-fold training complete: {run_dir}', flush=True)
        print('Evaluate with: ' + shlex.join([sys.executable, str(PROJECT_ROOT/'scripts/evaluate.py'),
                                               '--experiment-root', str(run_dir)]), flush=True)


if __name__ == '__main__':
    main()
