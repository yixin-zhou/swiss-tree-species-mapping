"""Shared building blocks for V2 token fusion."""
import torch
from torch import nn


def pad_per_sample(features, batch_idx, batch_size):
    """Scatter a flat (sumN, D) tensor (batched by batch_idx) into a padded
    (B, Nmax, D) tensor + a key-padding mask (B, Nmax) where True = PAD/ignore.
    """
    d = features.shape[1]
    counts = torch.bincount(batch_idx, minlength=batch_size)
    n_max = int(counts.max().item()) if counts.numel() else 0
    n_max = max(n_max, 1)
    out = features.new_zeros(batch_size, n_max, d)
    mask = torch.ones(batch_size, n_max, dtype=torch.bool, device=features.device)
    for b in range(batch_size):
        sel = batch_idx == b
        n = int(counts[b].item())
        if n > 0:
            out[b, :n] = features[sel]
            mask[b, :n] = False
    return out, mask


class LearnedQueryPooling(nn.Module):
    """K learnable queries cross-attend a variable token set -> K fixed tokens.

    Shared by the LiDAR voxel tokenizer and the Sentinel-2 sits_raw master-query
    pooling. Each layer: pre-norm cross-attention(queries <- tokens) + FFN.
    """

    def __init__(self, embed_dim, num_queries, num_layers=2, num_heads=8, ff_mult=4, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, int(num_queries), int(embed_dim)))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.layers.append(nn.ModuleDict({
                "q_norm": nn.LayerNorm(embed_dim),
                "kv_norm": nn.LayerNorm(embed_dim),
                "cross": nn.MultiheadAttention(embed_dim, int(num_heads), dropout=float(dropout), batch_first=True),
                "ff_norm": nn.LayerNorm(embed_dim),
                "ff": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * int(ff_mult)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(embed_dim * int(ff_mult), embed_dim),
                ),
            }))
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, key_padding_mask=None):
        """tokens (B,N,D); key_padding_mask (B,N) True=PAD. -> (B,K,D)."""
        b = tokens.shape[0]
        q = self.queries.expand(b, -1, -1)
        for layer in self.layers:
            attn, _ = layer["cross"](
                layer["q_norm"](q),
                layer["kv_norm"](tokens),
                layer["kv_norm"](tokens),
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            q = q + attn
            q = q + layer["ff"](layer["ff_norm"](q))
        return self.out_norm(q)
