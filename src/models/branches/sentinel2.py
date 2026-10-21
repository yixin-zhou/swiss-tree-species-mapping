import torch
from torch import nn


class TemporalAttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens, mask=None):
        attn_logits = self.score(tokens).squeeze(-1)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.bool(), float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)
        return torch.sum(tokens * attn[:, :, None], dim=1), attn


class Sentinel2Branch(nn.Module):
    def __init__(
        self,
        in_channels=10,
        hidden_dim=256,
        embed_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        pooling="attention",
        use_month_embedding=True,
        max_month=12,
        norm_first=False,
        return_debug=False,
    ):
        super().__init__()
        valid_pooling = ["mean", "attention"]
        if pooling not in valid_pooling:
            raise ValueError(f"Unknown pooling='{pooling}'. Expected one of {valid_pooling}.")

        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.pooling = pooling
        self.use_month_embedding = bool(use_month_embedding)
        self.max_month = int(max_month)
        self.norm_first = bool(norm_first)
        self.return_debug = bool(return_debug)

        self.patch_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        if self.use_month_embedding:
            self.month_embedding = nn.Embedding(self.max_month, self.hidden_dim)
        else:
            self.month_embedding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=self.norm_first,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(num_layers),
        )

        if self.pooling == "attention":
            self.attn_pool = TemporalAttentionPooling(self.hidden_dim)
        else:
            self.attn_pool = None

        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

    def forward(self, batch):
        if "sentinel2" not in batch or batch["sentinel2"] is None:
            raise KeyError("Sentinel2Branch requires batch['sentinel2'].")

        s2 = batch["sentinel2"]
        data = s2["data"].float()
        month = s2.get("month")

        if data.ndim != 5:
            raise ValueError(
                f"Expected Sentinel-2 data shape (B, T, C, H, W), got {tuple(data.shape)}."
            )

        b, t, c, h, w = data.shape
        if c != self.in_channels:
            raise ValueError(
                f"Expected Sentinel-2 in_channels={self.in_channels}, got {c}."
            )

        tokens = self.patch_encoder(data.reshape(b * t, c, h, w))
        tokens = tokens.reshape(b, t, self.hidden_dim)

        if self.use_month_embedding:
            if month is None:
                raise KeyError(
                    "use_month_embedding=True requires batch['sentinel2']['month']."
                )
            month_id = month.long().clamp(min=1, max=self.max_month) - 1
            tokens = tokens + self.month_embedding(month_id)

        temporal_tokens = self.temporal_encoder(tokens)

        if self.pooling == "attention":
            pooled, attn = self.attn_pool(temporal_tokens)
        elif self.pooling == "mean":
            pooled = temporal_tokens.mean(dim=1)
            attn = None
        else:
            raise ValueError(f"Unknown pooling='{self.pooling}'.")

        s2_token = self.out_proj(pooled)
        out = {"s2_token": s2_token}

        if self.return_debug:
            out.update({
                "s2_temporal_tokens": temporal_tokens,
                "s2_month_attention": attn,
            })

        return out
