import torch
from torch import nn


class SatelliteEmbeddingBranch(nn.Module):
    """AlphaEarth Foundations (AEF) embedding branch.

    num_tokens=1 (default): a single center-pixel embedding per site -> 1 token,
    same as the original design.

    num_tokens>1: a small spatial patch (e.g. 3x3=9 pixels) per site. The SAME
    per-pixel encoder is shared across all positions (they carry the same kind
    of signal, just at different spatial offsets), plus an additive learned
    positional embedding so the fusion layer's cross-attention can tell the
    center pixel (the NFI point itself) apart from its neighbors.
    """

    def __init__(self, in_dim=64, hidden_dim=256, embed_dim=512, dropout=0.1, encoder_type="mlp", num_tokens=1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.encoder_type = str(encoder_type)
        self.num_tokens = int(num_tokens)

        if self.encoder_type == "linear":
            # Linear probe: normalize the frozen embedding -> single projection -> normalize.
            # The cleanest "is the frozen embedding good?" readout; any nonlinearity is left
            # to the downstream head (avoids the redundant constant-width 2-layer MLP).
            self.encoder = nn.Sequential(
                nn.LayerNorm(self.in_dim),
                nn.Linear(self.in_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )
        else:
            self.encoder = nn.Sequential(
                nn.LayerNorm(self.in_dim),
                nn.Linear(self.in_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )

        if self.num_tokens > 1:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(self, batch):
        if "sentinel2" not in batch or batch["sentinel2"] is None:
            raise KeyError("SatelliteEmbeddingBranch requires batch['sentinel2'].")
        s2 = batch["sentinel2"]
        if "embedding" not in s2:
            raise KeyError("SatelliteEmbeddingBranch requires batch['sentinel2']['embedding'].")

        embedding = s2["embedding"].float()
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(1)  # (B,in_dim) -> (B,1,in_dim)
        if embedding.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected satellite embedding dimension {self.in_dim}, "
                f"got shape {tuple(embedding.shape)}."
            )
        if embedding.shape[1] != self.num_tokens:
            raise ValueError(
                f"Expected {self.num_tokens} satellite embedding token(s), "
                f"got shape {tuple(embedding.shape)}."
            )

        # Per-token validity from the RAW values (rare NaN edge pixels near the
        # Swiss border can occur in patch mode); sanitize afterwards so a NaN
        # token can't leak through the encoder/attention even though its gate
        # / mask already excludes it from the fusion sum.
        valid = torch.isfinite(embedding).all(dim=-1)  # (B, num_tokens)
        finite_values = embedding[torch.isfinite(embedding)]
        if finite_values.numel() and finite_values.abs().max().item() > 1.00001:
            raise ValueError(
                "AlphaEarth embeddings must be dequantized to the signed unit "
                "range before entering SatelliteEmbeddingBranch."
            )
        embedding = torch.nan_to_num(embedding, nan=0.0)

        tokens = self.encoder(embedding)  # (B, num_tokens, embed_dim)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed

        return {"s2_tokens": tokens, "s2_valid": valid}
