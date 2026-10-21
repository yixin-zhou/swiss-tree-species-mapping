"""Shared implementation defaults for the published reference model."""

from pathlib import Path

from src.model_spec import MODEL_SPEC

# Provenance of the existing paper fold assignments; unrelated to training RNG.
CV_FOLD_COLUMN = "cv_fold_5"
CV_FOLD_SOURCE = "10km_spatial_block_final11_balanced_v1"
CV_FOLD_SEED = 20260820
CV_NORMALIZATION_FILE = Path(__file__).resolve().parents[1] / "configs/cross_validation_normalization.json"

CV_EPOCHS = 80
NUM_FOLDS = 5

DEFAULT_CONFIG = {'path': {'hdf5': 'data/training', 'index': '${path.hdf5}/index.csv',
                         'rgb_pretrained_weights': 'checkpoints/dinov3_vith16plus.pth'},
 'modalities': {'rgb': {'enabled': True,
                        'crop_size_m': 28,
                        'resize_size': 256,
                        'multiseason': False,
                        'norm_mean': [0.485, 0.456, 0.406],
                        'norm_std': [0.229, 0.224, 0.225]},
                'sentinel2': {'enabled': True},
                'dem': {'enabled': True},
                'chelsa': {'enabled': True},
                'lidar': {'enabled': True,
                          'crop_radius_m': 14,
                          'version': 'random',
                          'height_norm': 'dem'}},
 'sampler': {'rare_class_ids': [2, 3, 4, 6, 7, 8, 9],
             'rare_mode': 'presence',
             'presence_threshold': 0.05,
             'rare_species_weight': 1.5,
             'normal_weight': 1.0,
             'replacement': True,
             'drop_last': False},
 'augmentation': {'geometric': True},
 'seed': 42,
 'device': 'cuda',
 'mixed_precision': {'enabled': False, 'dtype': 'float16'},
 'distributed': {'enabled': 'auto',
                 'backend': 'nccl',
                 'find_unused_parameters': True,
                 'sync_batchnorm': False},
 'dataloader': {'batch_size': 4,
                'num_workers': 6,
                'pin_memory': True,
                'persistent_workers': True,
                'drop_last': False},
 'optimizer': {'name': 'adamw',
               'lr': 0.0001,
               'weight_decay': 0.05,
               'betas': [0.9, 0.999],
               'eps': 1e-08},
 'scheduler': {'name': 'cosine', 'warmup_epochs': 0, 'min_lr': 1e-06},
 'training': {'epochs': 80,
              'max_grad_norm': 1.0,
              'log_every': 20,
              'progress_bar': True,
              'empty_cuda_cache_every_epoch': False,
              'ema': {'enabled': True, 'decay': 0.999}},
 'checkpoint': {'save_last': True},
 'losses': {'label_smoothing': 0.0,
            'stand_ce': {'enabled': True, 'weight': 1.0},
            'leaf_kl': {'enabled': True, 'weight': 0.2},
            'leaf_l1': {'enabled': True, 'weight': 1.0},
            'species_kl': {'enabled': True, 'weight': 0.2},
            'species_l1': {'enabled': True, 'weight': 1.0},
            'leaf_species_consistency': {'enabled': True, 'weight': 0.1},
            'leaf_entropy_pure': {'enabled': False},
            'species_entropy_pure': {'enabled': False}},
 'output': {'root': 'outputs/experiments', 'save_config': True},
 'logging': {'use_wandb': False,
             'wandb': {'entity': None,
                       'project': 'tree-species-mapping',
                       'mode': 'online',
                       'tags': [],
                       'group': None,
                       'notes': None}},
 'metrics': {'dominant_top_n': [1, 2, 3]},
 'experiment': {'name': 'reference_model'},
 'cross_validation': {'enabled': True,
                      'num_folds': 5,
                      'outer_fold': None}}

# Expanded architecture is retained in saved run/checkpoint metadata.
DEFAULT_CONFIG["model"] = MODEL_SPEC
