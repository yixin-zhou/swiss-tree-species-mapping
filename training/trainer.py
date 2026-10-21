import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.data.collate import nfi_collate_fn
from training.data.cross_validation import configure_outer_fold, cv_enabled, outer_fold
from training.data.dataset import NFISiteDataset
from training.data.samplers import build_sampler
from training.data.utils import build_transforms
from training.losses import build_loss
from src.utils.seed import seed as seed_everything
from src.utils.config import config_from_args, save_config as save_run_config


# spconv triggers torch.fx's is_fx_tracing check, which logs a benign warning via
# the logging module (not warnings.warn), so we silence that logger directly.
logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)


from src.config_defaults import CV_EPOCHS


def parse_args():
    parser = argparse.ArgumentParser(description="Train hierarchical tree species models.")
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, help="Output directory for this fold.")
    parser.add_argument("--set", dest="overrides", nargs="+", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_merged_config(args):
    return config_from_args(args)


def setup_distributed(cfg):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    requested = str(cfg.distributed.enabled).lower()
    if requested in ["false", "0", "no"]:
        distributed = False
    elif requested in ["true", "1", "yes"] and world_size <= 1:
        raise RuntimeError("distributed.enabled=true but WORLD_SIZE is not set by torchrun.")

    if distributed:
        backend = cfg.distributed.backend
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if torch.cuda.is_available() and cfg.device == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    return distributed, rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(distributed, device):
    if not distributed:
        return

    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def is_main_process(rank):
    return rank == 0


def build_output_dir(cfg, distributed=False, rank=0):
    experiment_name = cfg_get(cfg_get(cfg, "experiment", {}), "name", "experiment")
    experiment_root = Path(cfg.output.root) / experiment_name
    if cv_enabled(cfg):
        experiment_root = experiment_root / f"fold_{outer_fold(cfg)}"

    if is_main_process(rank):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(experiment_root / timestamp)
    else:
        output_dir = None

    if distributed:
        output_holder = [output_dir]
        dist.broadcast_object_list(output_holder, src=0)
        output_dir = output_holder[0]

    return Path(output_dir)


def save_config(cfg, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, output_dir / "config.yaml")


def init_wandb(cfg, output_dir, rank):
    if not is_main_process(rank):
        return None

    if not bool(cfg.logging.use_wandb):
        return None

    wandb_cfg = cfg.logging.wandb
    if str(wandb_cfg.mode).lower() == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "logging.use_wandb=true, but wandb is not installed in this environment."
        ) from exc

    experiment_name = cfg_get(cfg_get(cfg, "experiment", {}), "name", "experiment")

    return wandb.init(
        entity=wandb_cfg.entity,
        project=wandb_cfg.project,
        name=experiment_name,
        group=wandb_cfg.group,
        tags=list(wandb_cfg.tags) if wandb_cfg.tags is not None else None,
        notes=wandb_cfg.notes,
        mode=wandb_cfg.mode,
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def make_dataset(cfg, split, transform_split=None):
    transform = build_transforms(cfg, split=transform_split or split)
    return NFISiteDataset(cfg, split=split, transform=transform)


def make_train_dataset(cfg):
    return make_dataset(cfg, "train", transform_split="train")


def make_loader(cfg, dataset, is_train, distributed, rank, world_size):
    sampler = build_sampler(
        dataset=dataset,
        cfg=cfg,
        is_train=is_train,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        seed=int(cfg.seed),
    )

    shuffle = is_train and sampler is None
    num_workers = int(cfg.dataloader.num_workers)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(cfg.dataloader.pin_memory),
        persistent_workers=bool(cfg.dataloader.persistent_workers) and num_workers > 0,
        drop_last=bool(cfg.dataloader.drop_last) if is_train else False,
        collate_fn=nfi_collate_fn,
        worker_init_fn=seed_dataloader_worker,
    )

    return loader, sampler


def seed_dataloader_worker(worker_id):
    """Seed numpy/random per worker so the transforms' np.random augmentation
    (summer/winter & old/new selection, geometric ops) is actually independent
    across workers. Without this, forked workers share the parent numpy RNG
    state and produce correlated 'random' choices."""
    import random as _random

    import numpy as _np

    base_seed = torch.initial_seed() % (2 ** 32)
    _np.random.seed(base_seed)
    _random.seed(base_seed)


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)

    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}

    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)

    return obj


def get_amp_dtype(cfg):
    dtype_name = str(cfg.mixed_precision.dtype).lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown mixed_precision.dtype='{cfg.mixed_precision.dtype}'.")


def strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def build_optimizer(cfg, model):
    name = str(cfg.optimizer.name).lower()
    if name != "adamw":
        raise ValueError(f"Unknown optimizer.name='{cfg.optimizer.name}'. Only 'adamw' is implemented.")

    params = [param for param in model.parameters() if param.requires_grad]
    if len(params) == 0:
        raise ValueError("No trainable parameters found after applying freeze policy.")

    return torch.optim.AdamW(
        params,
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
        betas=tuple(cfg.optimizer.betas),
        eps=float(cfg.optimizer.eps),
    )


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class ModelEMA:
    """Exponential moving average of the trainable weights.

    Keeps a smoothed copy of trainable parameters for the final checkpoint.
    Frozen backbone parameters and live buffers remain in the full state dict.
    """

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in _unwrap(model).named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, param in _unwrap(model).named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(d).add_(param.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def model_state_dict(self, model):
        """Full state dict with EMA params substituted (buffers kept live)."""
        state = _unwrap(model).state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name] = value.to(state[name].dtype)
        return state


def build_ema(cfg, model):
    ema_cfg = cfg_get(cfg.training, "ema", None)
    if not bool(cfg_get(ema_cfg, "enabled", False)):
        return None
    return ModelEMA(model, decay=float(cfg_get(ema_cfg, "decay", 0.999)))


def build_scheduler(cfg, optimizer):
    """Per-epoch LR schedule. 'cosine' decays base_lr -> ~min_lr over training."""
    scheduler_cfg = cfg_get(cfg, "scheduler", {})
    name = cfg_get(scheduler_cfg, "name", "constant")
    name = str(name).lower()
    if name in ("constant", "none", "null"):
        return None
    if name != "cosine":
        raise ValueError(
            f"Unknown LR schedule='{name}'. Expected 'constant', 'none', or 'cosine'.")

    import math

    epochs = int(cfg.training.epochs)
    warmup = int(cfg_get(scheduler_cfg, "warmup_epochs", 0))
    base_lr = float(cfg.optimizer.lr)
    min_lr = float(cfg_get(scheduler_cfg, "min_lr", 0.0))
    min_ratio = (min_lr / base_lr) if base_lr > 0 else 0.0

    def lr_lambda(e):
        if warmup > 0 and e < warmup:
            return float(e + 1) / float(warmup)
        prog = float(e - warmup) / float(max(epochs - warmup, 1))
        prog = min(max(prog, 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def average_loss_across_ranks(loss_sum, count, device, distributed):
    stats = torch.tensor([loss_sum, count], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    if stats[1].item() == 0:
        return float("nan")

    return (stats[0] / stats[1]).item()


def reduce_log_sums(log_sums, count, device, distributed, prefix):
    if len(log_sums) == 0:
        return {}

    keys = sorted(log_sums)
    values = [log_sums[key] for key in keys]
    tensor = torch.tensor(values + [count], dtype=torch.float64, device=device)

    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    denom = tensor[-1].item()
    if denom == 0:
        return {}

    return {
        f"{prefix}/{key}": (tensor[i] / denom).item()
        for i, key in enumerate(keys)
    }


def make_progress_bar(iterable, cfg, rank, desc):
    if not is_main_process(rank):
        return iterable

    if not bool(cfg_get(cfg.training, "progress_bar", True)):
        return iterable

    try:
        from tqdm import tqdm
    except ImportError:
        return iterable

    return tqdm(
        iterable,
        desc=desc,
        total=len(iterable),
        file=sys.stdout,
        dynamic_ncols=True,
        leave=True,
        disable=False,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    scaler,
    device,
    cfg,
    distributed,
    rank,
    epoch,
    ema=None,
):
    model.train()

    amp_enabled = bool(cfg.mixed_precision.enabled) and device.type == "cuda"
    amp_dtype = get_amp_dtype(cfg)
    use_scaler = scaler is not None and scaler.is_enabled()

    loss_sum = 0.0
    n_samples = 0
    log_sums = {}

    optimizer.zero_grad(set_to_none=True)
    progress = make_progress_bar(loader, cfg, rank, desc=f"epoch {epoch} train")

    for batch in progress:
        batch = move_to_device(batch, device)
        batch_size = int(batch["stand_type_id"].shape[0])

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            outputs = model(batch)
            loss, logs = loss_fn(outputs, batch)

        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                f"Non-finite training loss at epoch={epoch}: {loss.detach().item()}"
            )

        if use_scaler:
            scaler.scale(loss).backward()

            if cfg.training.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    float(cfg.training.max_grad_norm),
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if cfg.training.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    float(cfg.training.max_grad_norm),
                )

            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if ema is not None:
            ema.update(model)

        loss_value = float(loss.detach().item())
        loss_sum += loss_value * batch_size
        n_samples += batch_size

        for key, value in logs.items():
            if not torch.is_tensor(value) or value.numel() != 1:
                continue
            if key.endswith("_n"):
                continue
            log_sums[key] = log_sums.get(key, 0.0) + float(value.detach().item()) * batch_size

        if is_main_process(rank) and hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{loss_value:.4f}")

    train_logs = reduce_log_sums(log_sums, n_samples, device, distributed, prefix="train")
    train_logs["train/loss"] = average_loss_across_ranks(loss_sum, n_samples, device, distributed)
    return train_logs


def checkpoint_state(epoch, model, optimizer, best_metric, cfg, distributed, model_state=None):
    model_to_save = model.module if distributed else model
    return {
        "epoch": epoch,
        "model": model_state if model_state is not None else model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }


def save_checkpoint(path, epoch, model, optimizer, best_metric, cfg, distributed, model_state=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_state(epoch, model, optimizer, best_metric, cfg, distributed, model_state),
        path,
    )


def drop_constant_site_counts(log_data):
    """Strip per-epoch site-count keys before logging to wandb.

    Validation counts (val/*_n_sites, *_total_sites, *_qualified_sites) are
    constant across epochs and only clutter the wandb dashboard with flat lines.
    They stay in the returned metrics dict for one-off evaluation scripts.
    """
    return {
        key: value
        for key, value in log_data.items()
        if not key.endswith("_sites")
    }


def format_epoch_log(epoch, log_data):
    pieces = [f"epoch={epoch}"]
    keys = ["train/loss"]
    for key in keys:
        if key in log_data and log_data[key] is not None:
            value = log_data[key]
            if isinstance(value, (int, float)):
                pieces.append(f"{key}={value:.6f}")
            else:
                pieces.append(f"{key}={value}")
    return " | ".join(pieces)


def main():
    args = parse_args()
    from src.models import build_model
    cfg = load_merged_config(args)
    configure_outer_fold(cfg, args.outer_fold)
    distributed, rank, world_size, local_rank, device = setup_distributed(cfg)
    try:
        if is_main_process(rank):
            print(
                f"Training configuration (outer fold {args.outer_fold}):\n"
                f"{OmegaConf.to_yaml(cfg, resolve=True)}",
                flush=True,
            )
        seed_everything(int(cfg.seed) + rank)
        output_dir = args.run_dir or build_output_dir(cfg, distributed=distributed, rank=rank)
        checkpoint_dir = output_dir / "checkpoints"
        if is_main_process(rank):
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_config(cfg, output_dir)
        distributed_barrier(distributed, device)
        wandb_run = init_wandb(cfg, output_dir, rank)
        train_loader, train_sampler = make_loader(
            cfg, make_train_dataset(cfg), is_train=True,
            distributed=distributed, rank=rank, world_size=world_size,
        )
        model = build_model(cfg).to(device)
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=bool(cfg.distributed.find_unused_parameters),
            )
        loss_fn = build_loss(cfg)
        optimizer = build_optimizer(cfg, model)
        scheduler = build_scheduler(cfg, optimizer)
        ema = build_ema(cfg, model)
        scaler = torch.amp.GradScaler("cuda", enabled=(
            bool(cfg.mixed_precision.enabled)
            and str(cfg.mixed_precision.dtype).lower() == "float16"
            and device.type == "cuda"
        ))
        # The outer test fold is never inspected during training. Save only the
        # final epoch's EMA weights and evaluate the held-out fold afterward.
        for epoch in range(1, CV_EPOCHS + 1):
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            train_logs = train_one_epoch(
                model=model, loader=train_loader, optimizer=optimizer,
                loss_fn=loss_fn, scaler=scaler, device=device, cfg=cfg,
                distributed=distributed, rank=rank, epoch=epoch, ema=ema,
            )
            if is_main_process(rank):
                log_data = {"epoch": epoch, **train_logs, "lr": optimizer.param_groups[0]["lr"]}
                print(format_epoch_log(epoch, log_data), flush=True)
                if wandb_run is not None:
                    wandb_run.log(drop_constant_site_counts(log_data), step=epoch)
                if epoch == CV_EPOCHS:
                    ema_state = ema.model_state_dict(model) if ema is not None else None
                    save_checkpoint(
                        checkpoint_dir / "last.pt", epoch, model, optimizer, None,
                        cfg, distributed, model_state=ema_state,
                    )
            if scheduler is not None:
                scheduler.step()
            distributed_barrier(distributed, device)
        if wandb_run is not None:
            wandb_run.finish()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
