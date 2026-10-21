"""ResNet50Composition baseline architecture."""

import torch
from torch import nn
import torchvision.models as tvm

from src.data.label_schema import SPECIES_NUM_CLASSES


class ResNet50Composition(nn.Module):
    """ImageNet ResNet-50 trunk with independent species, leaf, and stand heads."""

    def __init__(self, pretrained=True, freeze_backbone=False, dropout=0.1):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = tvm.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.dropout = nn.Dropout(float(dropout))
        self.species_head = nn.Linear(feature_dim, SPECIES_NUM_CLASSES)
        self.leaf_head = nn.Linear(feature_dim, 2)
        self.stand_head = nn.Linear(feature_dim, 2)

    def forward(self, batch):
        images = batch["rgb"]["data"]
        if images.ndim == 5:
            images = images.mean(dim=1)
        features = self.dropout(self.backbone(images.float()))
        species_logits = self.species_head(features)
        leaf_logits = self.leaf_head(features)
        stand_logits = self.stand_head(features)
        return {
            "species_logits": species_logits,
            "species_prob": torch.softmax(species_logits, dim=-1),
            "leaf_logits": leaf_logits,
            "leaf_prob": torch.softmax(leaf_logits, dim=-1),
            "stand_logits": stand_logits,
            "stand_prob": torch.softmax(stand_logits, dim=-1),
        }

