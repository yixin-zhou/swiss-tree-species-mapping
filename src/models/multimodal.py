"""Published four-modality architecture with fixed encoders, fusion and heads."""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from src.data.label_schema import LEAF_NUM_CLASSES, SPECIES_NUM_CLASSES, STAND_NUM_CLASSES
from src.model_spec import MODEL_SPEC
from src.models.branches import (
    ClimateBranch,
    ForestFormerLidarBranch,
    SatelliteEmbeddingBranch,
    SwissImageBranch,
    build_dinov3_backbone,
)
from src.models.fusion_rgb_anchored import RGBAnchoredFusion

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class MultiModalTreeSpeciesModel(nn.Module):
    """RGB-anchored fusion with independent stand, leaf and species heads."""

    def __init__(self, cfg):
        super().__init__()
        spec = deepcopy(MODEL_SPEC)
        self.embed_dim = spec["embed_dim"]
        self.active_modalities = set(spec["fusion"]["active_modalities"])

        rgb = spec["rgb"]
        dino = build_dinov3_backbone(
            repo_path=PROJECT_ROOT / rgb.pop("repo"),
            weights_path=cfg.path.rgb_pretrained_weights,
            backbone_name=rgb.pop("backbone"),
            pretrained=rgb.pop("pretrained"),
        )
        self.rgb_branch = SwissImageBranch(dino=dino, output_mode="tokens", **rgb)

        lidar = spec["lidar"]
        lidar.pop("encoder")
        lidar_weights = lidar.pop("weights")
        unet = lidar.pop("unet")
        self.lidar_branch = ForestFormerLidarBranch(
            weights_path=lidar_weights, output_mode="tokens", **lidar, **unet,
        )

        satellite = spec["sentinel2"]
        self.sentinel2_branch = SatelliteEmbeddingBranch(
            in_dim=satellite["embedding_dim"],
            hidden_dim=satellite["embedding_hidden_dim"],
            embed_dim=self.embed_dim,
            num_tokens=satellite["embedding_num_tokens"],
            dropout=satellite["dropout"],
        )
        climate = spec["climate"]
        self.climate_branch = ClimateBranch(
            source=climate["source"], embed_dim=self.embed_dim,
            chelsa_cfg=climate["chelsa"],
        )

        # This unused parameter remains for strict loading of the published weights.
        self.month_embed = nn.Embedding(12, self.embed_dim)
        nn.init.trunc_normal_(self.month_embed.weight, std=0.02)
        fusion = spec["fusion"]
        fusion.pop("active_modalities")
        fusion.pop("return_debug")
        self.fusion = RGBAnchoredFusion(
            embed_dim=self.embed_dim, anchor_modalities=("rgb",),
            expert_modalities=("lidar", "sentinel2", "climate"), **fusion,
        )
        self.stand_num_classes = STAND_NUM_CLASSES
        self.leaf_num_classes = LEAF_NUM_CLASSES
        self.species_num_classes = SPECIES_NUM_CLASSES
        self.stand_head = nn.Linear(self.embed_dim, STAND_NUM_CLASSES)
        self.leaf_head = nn.Linear(self.embed_dim, LEAF_NUM_CLASSES)
        self.species_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, SPECIES_NUM_CLASSES),
        )

    def forward(self, batch):
        lidar_tokens = self.lidar_branch(batch)["lidar_tokens"]
        rgb = self.rgb_branch(batch["rgb"])
        rgb_tokens = rgb["rgb_tokens"]
        if rgb_tokens.ndim == 4:
            # Pool multiple eligible views using the original view mask.
            weights = rgb["rgb_view_mask"].float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            rgb_tokens = (rgb_tokens * weights[:, :, None, None]).sum(dim=1)
            rgb_mask = torch.ones(rgb_tokens.shape[:2], dtype=torch.bool, device=rgb_tokens.device)
        else:
            rgb_mask = rgb.get("rgb_token_mask")
            if rgb_mask is None:
                rgb_mask = torch.ones(rgb_tokens.shape[:2], dtype=torch.bool, device=rgb_tokens.device)
        satellite = self.sentinel2_branch(batch)
        climate_token = self.climate_branch(batch)["climate_token"]
        fused_token = self.fusion(
            {"rgb": rgb_tokens, "lidar": lidar_tokens,
             "sentinel2": satellite["s2_tokens"], "climate": climate_token.unsqueeze(1)},
            {"rgb": rgb_mask, "sentinel2": satellite["s2_valid"]},
        )
        outputs = {"fused_global_token": fused_token}
        for task in ("stand", "leaf", "species"):
            logits = getattr(self, f"{task}_head")(fused_token)
            outputs[f"{task}_logits"] = logits
            outputs[f"{task}_prob"] = torch.softmax(logits, dim=-1)
        return outputs
