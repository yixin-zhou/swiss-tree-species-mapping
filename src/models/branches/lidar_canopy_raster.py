import torch
from torch import nn


class LidarCanopyRasterBranch(nn.Module):
    """LiDAR-as-image: a small from-scratch CNN over the upper-canopy raster
    produced by SwissLidarTransform(compute_canopy_raster=True) -- 3 channels
    (P50, P90, P90-P50 of upper-canopy points per grid cell), instead of the
    sparse-conv (ForestFormer/SpConvUNet) point-cloud path.

    No pretrained weights exist for this input domain, so channel widths stay
    deliberately narrow in the (expensive, kernel^2-scaled) conv layers; the
    final projection to embed_dim happens via a small 2-layer MLP, matching the
    same "in_dim -> hidden_dim -> embed_dim" recipe used by the climate/
    satellite-embedding branches (not an abrupt single-step projection).
    """

    def __init__(
        self,
        in_channels=3,
        conv_channels=(16, 32, 64),
        hidden_dim=256,
        embed_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        c1, c2, c3 = (int(c) for c in conv_channels)

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(c3),
            nn.Linear(c3, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def forward(self, batch):
        if "lidar" not in batch or batch["lidar"] is None:
            raise KeyError("LidarCanopyRasterBranch requires batch['lidar'].")
        lidar = batch["lidar"]
        if "canopy_raster" not in lidar:
            raise KeyError(
                "LidarCanopyRasterBranch requires batch['lidar']['canopy_raster']. "
                "Enable modalities.lidar.canopy_raster.enabled in the dataset config."
            )

        raster = lidar["canopy_raster"].float()  # (B, 3, grid, grid)
        feat = self.conv(raster)                 # (B, c3, g, g) e.g. (B,64,7,7)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, h*w, c3)
        lidar_tokens = self.proj(tokens)          # (B, h*w, embed_dim)

        return {"lidar_tokens": lidar_tokens}
