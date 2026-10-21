# DINOv3 runtime subset

Source: https://github.com/facebookresearch/dinov3
Copyright (c) Meta Platforms, Inc. and affiliates.
License: [DINOv3 License](LICENSE.md).

This directory retains the ViT-H+/16 backbone and the Python modules needed by
this repository. Official pretraining, datasets, downstream-task evaluation,
other backbones, and FP8 conversion entry points have been removed. Package
initializers and the backbone factory exports were narrowed accordingly.
The retained transformer and layer implementations are unchanged.

Obtain pretrained weights using [the project download instructions](../../../checkpoints/DOWNLOAD.txt).
