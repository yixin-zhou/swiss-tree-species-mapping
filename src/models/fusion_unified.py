"""Unified token-set fusion.

All modalities are exposed as token sets (season-fused RGB patches+cls, LiDAR
K-query tokens, Sentinel-2 K_s2 master-query tokens, climate token). Each token
gets a learned modality-type embedding; a learnable [FUSE] token is prepended;
a single Transformer encoder mixes everything (self-attention), with per-sample
key-padding masks for missing/invalid modalities and optional modality dropout.
The [FUSE] output is the fused representation the hierarchical heads read.

Replaces the staged SpatialCrossAttentionFusion + SpatialAttentionPooling +
4-token GlobalTokenFusion (and the RGB multi-view gate / LiDAR BEV pooling).
"""
import torch
from torch import nn


class UnifiedTokenFusion(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        modalities=("rgb", "lidar", "sentinel2", "climate"),
        num_layers=3,
        num_heads=8,
        ff_mult=4,
        dropout=0.1,
        modality_dropout=0.1,
        norm_first=True,
        readout="fuse_token",
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.modalities = list(modalities)
        self.modality_dropout = float(modality_dropout)
        self.readout = str(readout)  # "fuse_token" | "gated_modality"

        # learned per-modality type embedding + the [FUSE] readout token
        self.modality_embed = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(1, 1, self.embed_dim)) for m in self.modalities}
        )
        self.fuse_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.input_norm = nn.LayerNorm(self.embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=self.embed_dim * int(ff_mult),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.out_norm = nn.LayerNorm(self.embed_dim)

        # modality-level gate: scores {FUSE, per-modality pooled} -> softmax weights
        if self.readout == "gated_modality":
            self.gate_score = nn.Linear(self.embed_dim, 1)
        self.last_gate = None  # cached softmax weights (for logging / interpretability)

        for p in self.modality_embed.values():
            nn.init.trunc_normal_(p, std=0.02)
        nn.init.trunc_normal_(self.fuse_token, std=0.02)

    def forward(self, token_dict, mask_dict=None, return_tokens=False):
        """
        token_dict: {modality_name: (B, N_m, D)}
        mask_dict:  {modality_name: (B, N_m) bool, True = VALID}  (optional)
        Returns fused_token (B, D); optionally the full token sequence.
        """
        mask_dict = mask_dict or {}
        any_tok = next(iter(token_dict.values()))
        b = any_tok.shape[0]
        device = any_tok.device

        seq, key_valid, spans = [], [], {}
        # [FUSE] token (always valid)
        seq.append(self.fuse_token.expand(b, -1, -1))
        key_valid.append(torch.ones(b, 1, dtype=torch.bool, device=device))
        idx = 1

        for m in self.modalities:
            if m not in token_dict or token_dict[m] is None:
                continue
            tok = token_dict[m] + self.modality_embed[m]  # add modality-type embed
            n = tok.shape[1]
            if m in mask_dict and mask_dict[m] is not None:
                valid = mask_dict[m].to(device=device, dtype=torch.bool)
            else:
                valid = torch.ones(b, n, dtype=torch.bool, device=device)

            # per-sample modality dropout (training only): drop whole modality
            if self.training and self.modality_dropout > 0.0:
                drop = torch.rand(b, device=device) < self.modality_dropout
                valid = valid & (~drop).unsqueeze(1)

            seq.append(tok)
            key_valid.append(valid)
            spans[m] = (idx, idx + n, valid)
            idx += n

        seq = torch.cat(seq, dim=1)               # (B, 1+sum N_m, D)
        key_valid = torch.cat(key_valid, dim=1)   # (B, 1+sum N_m)  True = valid
        seq = self.input_norm(seq)

        # TransformerEncoder expects True = PAD/ignore
        out = self.encoder(seq, src_key_padding_mask=~key_valid)
        out = self.out_norm(out)

        if self.readout == "gated_modality":
            cands = [out[:, 0]]                                  # [FUSE]
            cand_valid = [torch.ones(b, dtype=torch.bool, device=device)]
            for m, (s, e, valid) in spans.items():
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
                v = (out[:, s:e] * valid.unsqueeze(-1)).sum(dim=1) / denom   # masked mean (B,D)
                cands.append(v)
                cand_valid.append(valid.any(dim=1))             # fully-dropped modality -> out of gate
            cands = torch.stack(cands, dim=1)                   # (B, 1+M, D)
            cand_valid = torch.stack(cand_valid, dim=1)         # (B, 1+M)
            logits = self.gate_score(cands).squeeze(-1)         # (B, 1+M)
            logits = logits.masked_fill(~cand_valid, float("-inf"))
            weights = torch.softmax(logits, dim=1)              # (B, 1+M)
            fused = (weights.unsqueeze(-1) * cands).sum(dim=1)  # (B, D)
            self.last_gate = weights.detach()
        else:
            fused = out[:, 0]  # [FUSE] position

        if return_tokens:
            return fused, out, key_valid
        return fused
