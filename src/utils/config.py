"""Load compact experiment files and materialize the complete run configuration."""
from pathlib import Path

from omegaconf import OmegaConf

from src.config_defaults import DEFAULT_CONFIG, CV_EPOCHS, NUM_FOLDS
from src.model_spec import MODEL_SPEC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_CONFIG = "configs/main.yaml"


def config_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def default_cv_output(cfg):
    """Shared default location used by the public training and evaluation entries."""
    return config_path(Path(str(cfg.output.root)) / str(cfg.experiment.name) / "cross_validation")


def load_config(path):
    """Read a YAML fragment. Use load_training_config for runnable configs."""
    path = config_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def complete_config(*fragments):
    """Apply defaults and infer data loading from active model branches.

    Explicit enabled flags remain accepted for checkpoint compatibility. Defaults
    are recreated on each call, so experiments cannot mutate each other.
    """
    overrides = OmegaConf.merge(*(f for f in fragments if f is not None))
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULT_CONFIG), overrides)
    # Saved checkpoints carry expanded architecture metadata. Verify it instead
    # of silently ignoring incompatible settings; only the weight path varies.
    def check_model(actual, expected, prefix="model"):
        for key, value in expected.items():
            path = f"{prefix}.{key}"
            if isinstance(value, dict):
                check_model(actual[key], value, path)
            elif actual[key] != value:
                raise ValueError(f"{path} differs from the fixed reference architecture.")
    check_model(cfg.model, MODEL_SPEC)
    # Older checkpoints stored the file location inside model metadata.
    if OmegaConf.select(overrides, "path.rgb_pretrained_weights") is None:
        legacy_weights = OmegaConf.select(overrides, "model.rgb.weights")
        if legacy_weights is not None:
            cfg.path.rgb_pretrained_weights = legacy_weights
    cfg.model = OmegaConf.create(MODEL_SPEC)
    # All four reference branches are active; baselines may explicitly disable
    # data loading when reusing a saved configuration.
    for modality in ("rgb", "lidar", "sentinel2", "dem", "chelsa"):
        if OmegaConf.select(overrides, f"modalities.{modality}.enabled") is None:
            cfg.modalities[modality].enabled = True
    return cfg


def load_training_config(config=MAIN_CONFIG, outer_fold=None, overrides=None):
    """Load the fixed five-fold, 80-epoch reference experiment."""
    fragments = [load_config(config)]
    if overrides:
        fragments.append(OmegaConf.from_dotlist(overrides))
    inputs = OmegaConf.merge(*fragments)
    # These are protocol/architecture facts, not public tuning parameters.
    removed = ("training.epochs", "training.validate_every_epoch", "training.regime",
               "cross_validation.enabled", "cross_validation.no_inner_validation",
               "model.fusion.mode", "model.heads", "sampler.name",
               "cross_validation.fold_column", "cross_validation.source", "cross_validation.seed", "cross_validation.normalization_file")
    for key in removed:
        if OmegaConf.select(inputs, key) is not None:
            raise ValueError(f"{key} is fixed by the reference CV implementation; remove this setting.")
    if "model" in inputs:
        raise ValueError("The model architecture is fixed; set path.rgb_pretrained_weights for the weight file.")
    cfg = complete_config(inputs)
    if int(cfg.cross_validation.num_folds) != NUM_FOLDS:
        raise ValueError("The paper protocol uses exactly five folds.")
    if set(cfg.model.fusion.active_modalities) != {"rgb", "lidar", "sentinel2", "climate"}:
        raise ValueError("The reference experiment uses all four modalities.")
    cfg.cross_validation.enabled = True
    cfg.cross_validation.outer_fold = outer_fold
    cfg.training.epochs = CV_EPOCHS
    return cfg


def config_from_args(args):
    return load_training_config(
        config=args.config,
        outer_fold=getattr(args, "outer_fold", None),
        overrides=getattr(args, "overrides", None),
    )


def apply_overrides(cfg, overrides):
    """Override a saved run, keeping a relocated dataset's index with its HDF5s."""
    if not overrides:
        return cfg
    fragment = OmegaConf.from_dotlist(overrides)
    if OmegaConf.select(fragment, "path.hdf5") is not None:
        if OmegaConf.select(fragment, "path.index") is None:
            OmegaConf.update(fragment, "path.index", "${path.hdf5}/index.csv", force_add=True)
    return complete_config(cfg, fragment)


def save_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=path, resolve=True)
