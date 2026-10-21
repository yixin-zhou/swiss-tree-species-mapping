import torch
from torch import nn


class SpatialCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1, ff_mult=4):
        super().__init__()
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.lidar_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * int(ff_mult)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(embed_dim * int(ff_mult), embed_dim),
        )

    def forward(self, rgb_spatial, lidar_spatial):
        if rgb_spatial.shape != lidar_spatial.shape:
            raise ValueError(
                "SpatialCrossAttentionFusion expects matching shapes, got "
                f"rgb={tuple(rgb_spatial.shape)} and lidar={tuple(lidar_spatial.shape)}."
            )

        b, c, h, w = rgb_spatial.shape
        rgb_tokens = rgb_spatial.flatten(2).transpose(1, 2)
        lidar_tokens = lidar_spatial.flatten(2).transpose(1, 2)

        query = self.rgb_norm(rgb_tokens)
        key_value = self.lidar_norm(lidar_tokens)
        attended, attn = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=False,
        )

        fused = rgb_tokens + self.dropout(attended)
        fused = fused + self.dropout(self.ffn(fused))
        fused_spatial = fused.transpose(1, 2).reshape(b, c, h, w)

        return fused_spatial, attn


class SpatialAttentionPooling(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Linear over the channel dim (equivalent to a 1x1 conv) avoids the
        # channels-last grad layout that 1x1 Conv2d produces, which otherwise
        # triggers DDP's "grad strides do not match bucket view strides" warning.
        self.score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, spatial_features):
        if spatial_features.ndim != 4:
            raise ValueError(
                f"Expected spatial_features shape (B, C, H, W), got {tuple(spatial_features.shape)}."
            )

        b, c, h, w = spatial_features.shape
        tokens = spatial_features.flatten(2).transpose(1, 2)
        logits = self.score(tokens).squeeze(-1)
        attn = torch.softmax(logits, dim=1)
        token = torch.sum(tokens * attn.unsqueeze(-1), dim=1)
        attn = attn.reshape(b, 1, h, w)
        return token, attn


class GlobalTokenFusion(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        token_names=None,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        ff_mult=4,
        norm_first=False,
    ):
        super().__init__()
        self.token_names = token_names or [
            "rgb",
            "sentinel2",
            "climate",
            "spatial",
        ]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_embedding = nn.Parameter(
            torch.zeros(1, len(self.token_names) + 1, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=int(num_heads),
            dim_feedforward=embed_dim * int(ff_mult),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)

    def forward(self, token_dict):
        tokens = []
        for name in self.token_names:
            if name not in token_dict:
                raise KeyError(f"GlobalTokenFusion missing token '{name}'.")
            tokens.append(token_dict[name])

        x = torch.stack(tokens, dim=1)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.modality_embedding[:, : x.shape[1]]
        x = self.encoder(x)
        return self.norm(x[:, 0]), x
