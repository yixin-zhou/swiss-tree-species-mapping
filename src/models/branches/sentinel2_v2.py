"""Sentinel-2 monthly encoder.

    Input: cross-year monthly median composites (B,T,10,5,5) with NaN where a
    pixel had no valid observation, + per-month validity (B,T). Produces T*4 tokens
    (T months x 4 raw-band groups) via masked spatial mean -> band-group encode ->
    factorized temporal/spectral attention. Emits a per-token validity mask (month
    validity broadcast to its 4 group tokens) and the per-token calendar-month index
    (0-11) for the shared month embedding applied at fusion time.
"""
import torch
from torch import nn

DEFAULT_BAND_GROUPS = [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]


class Sentinel2Monthly12Branch(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256, band_groups=None,
                 num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.band_groups = [list(g) for g in (band_groups or DEFAULT_BAND_GROUPS)]
        self.num_groups = len(self.band_groups)

        self.group_encoders = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(len(g)), nn.Linear(len(g), self.hidden_dim),
                          nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))
            for g in self.band_groups
        ])
        self.month_pos = nn.Parameter(torch.zeros(1, 12, 1, self.hidden_dim))
        self.group_pos = nn.Parameter(torch.zeros(1, 1, self.num_groups, self.hidden_dim))

        def layer():
            return nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, nhead=int(num_heads),
                dim_feedforward=self.hidden_dim * 4, dropout=float(dropout),
                activation="gelu", batch_first=True, norm_first=True)
        self.temporal_layers = nn.ModuleList([layer() for _ in range(int(num_layers))])
        self.spectral_layers = nn.ModuleList([layer() for _ in range(int(num_layers))])

        self.out_proj = nn.Sequential(nn.LayerNorm(self.hidden_dim),
                                      nn.Linear(self.hidden_dim, self.embed_dim))
        self.group_embed_out = nn.Parameter(torch.zeros(1, self.num_groups, self.embed_dim))

        nn.init.trunc_normal_(self.month_pos, std=0.02)
        nn.init.trunc_normal_(self.group_pos, std=0.02)
        nn.init.trunc_normal_(self.group_embed_out, std=0.02)

    def forward(self, batch):
        s2 = batch["sentinel2"]
        data = s2["data"]                       # (B,T,10,5,5) with NaN
        month_valid = s2["month_valid"].bool()  # (B,T)
        months = s2["month"].long().clamp(min=1, max=12) - 1  # (B,T) or (T,)
        b = data.shape[0]
        t = data.shape[1]
        device = data.device
        if months.ndim == 2:
            months_for_pos = months[0]
        else:
            months_for_pos = months
        if months_for_pos.numel() != t:
            raise ValueError(
                f"Expected {t} Sentinel-2 month ids, got {months_for_pos.numel()}."
            )

        # masked spatial mean per band over valid (all-band-finite) pixels
        valid_pix = torch.isfinite(data).all(dim=2)          # (B,T,5,5)
        vp = valid_pix.unsqueeze(2).float()                  # (B,T,1,5,5)
        data0 = torch.nan_to_num(data, nan=0.0)
        num = (data0 * vp).sum(dim=(-1, -2))                 # (B,T,10)
        den = vp.sum(dim=(-1, -2)).clamp_min(1.0)            # (B,T,1)
        band_mean = num / den                                # (B,T,10)

        # per band-group -> list of (B,T,hidden)
        group_feats = [
            self.group_encoders[g](band_mean[..., self.band_groups[g]])
            for g in range(len(self.band_groups))
        ]

        grid = torch.stack(group_feats, dim=2)               # (B,T,num_groups,hidden)
        month_pos = self.month_pos[:, months_for_pos, :, :]
        grid = grid + month_pos + self.group_pos

        # factorized attention: temporal (over selected months, per group), spectral (over raw-band groups, per month)
        # temporal key-padding mask from month_valid; guard fully-invalid rows to avoid NaN
        tmask = ~month_valid                                  # (B,T) True = PAD
        tmask_g = tmask.unsqueeze(1).expand(b, self.num_groups, t).reshape(b * self.num_groups, t).clone()
        tmask_g[tmask_g.all(dim=1)] = False                  # fully-invalid -> unmask (output masked later)
        for tl, sl in zip(self.temporal_layers, self.spectral_layers):
            x = grid.permute(0, 2, 1, 3).reshape(b * self.num_groups, t, self.hidden_dim)
            x = tl(x, src_key_padding_mask=tmask_g)
            grid = x.reshape(b, self.num_groups, t, self.hidden_dim).permute(0, 2, 1, 3)
            y = grid.reshape(b * t, self.num_groups, self.hidden_dim)
            y = sl(y)
            grid = y.reshape(b, t, self.num_groups, self.hidden_dim)

        tokens = self.out_proj(grid).reshape(b, t * self.num_groups, self.embed_dim)  # (B,T*G,D)
        group_idx = torch.arange(self.num_groups, device=device).repeat(t)            # (T*G,)
        tokens = tokens + self.group_embed_out[:, group_idx % self.num_groups, :]

        s2_valid = month_valid.unsqueeze(-1).expand(b, t, self.num_groups).reshape(b, t * self.num_groups)
        s2_month_idx = months_for_pos.to(device).repeat_interleave(self.num_groups)  # (T*G,) 0..11

        return {"s2_tokens": tokens, "s2_valid": s2_valid, "s2_month_idx": s2_month_idx}
