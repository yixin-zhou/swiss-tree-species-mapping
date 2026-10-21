"""RGB-anchored gated residual fusion.

RGB is the trunk: its tokens are attention-pooled into an anchor
vector h0 (the RGB-only representation the heads would otherwise use). Each other
modality (LiDAR / Sentinel-2 / climate) proposes a correction Δ_m by cross-attending
from h0 to its tokens, gated by a learned g_m in [0,1]:

    fused = h0 + Σ_m g_m · Δ_m

Gates are initialised ~0 (gate_bias_init negative + zero-init final weight), so at
the start fused ≈ h0 == RGB-only (the strongest baseline). The optimiser can only
open a gate where a modality demonstrably helps -> fusion cannot hurt RGB by
construction. Missing / fully-invalid / modality-dropped modalities get g_m = 0.

Three independent, config-driven gate-design axes (each defaults to the original
behaviour):
  - gate_type: "scalar" (one number per modality) | "per_channel" (one gate value
    per embed_dim channel, i.e. Δ_m can be trusted on some channels and not others).
  - gate_sees_delta: the gate MLP's input is normally [h0, masked-mean(tokens)];
    if true, the actual computed correction Δ_m is concatenated in too, so the gate
    can react to what the cross-attention concluded, not just the raw modality.
  - task_conditioned: Δ_m is computed once per modality (shared, expensive part
    unchanged) but each downstream task (stand/leaf/species) gets its own learned
    gate, producing a task-specific fused token instead of one shared token for
    all three heads.

Replaces the symmetric UnifiedTokenFusion (where strong RGB tokens were diluted by
noisy weak-modality tokens before the late readout gate could act).
"""
import torch
from torch import nn


def _guard_full_pad(key_padding_mask):
    """Unmask fully-padded rows (all True) so MultiheadAttention stays finite.

    Those rows are zeroed out downstream by the gate / validity flag anyway.
    """
    kpm = key_padding_mask.clone()
    kpm[kpm.all(dim=1)] = False
    return kpm


class _CrossAttnExpert(nn.Module):
    """Cross-attend from the RGB anchor query to one modality's tokens -> correction Δ."""

    def __init__(self, embed_dim, num_heads, ff_mult, dropout):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, int(num_heads), dropout=float(dropout), batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * int(ff_mult)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(embed_dim * int(ff_mult), embed_dim),
        )

    def forward(self, anchor, tokens, key_padding_mask):
        # anchor: (B,D); tokens: (B,N,D); key_padding_mask: (B,N) True = PAD
        q = self.norm_q(self.q_proj(anchor)).unsqueeze(1)   # (B,1,D)
        kv = self.norm_kv(tokens)                           # (B,N,D)
        ctx, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        ctx = ctx.squeeze(1)                                # (B,D)
        return ctx + self.ff(ctx)                           # (B,D)


class RGBAnchoredFusion(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        anchor_modalities=("rgb",),
        expert_modalities=("lidar", "sentinel2", "climate"),
        num_heads=8,
        ff_mult=4,
        dropout=0.1,
        modality_dropout=0.1,
        gate_bias_init=-4.0,
        gate_type="scalar",
        gate_sees_delta=False,
        task_conditioned=False,
        tasks=("stand", "leaf", "species"),
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.anchor_modalities = list(anchor_modalities)
        self.expert_modalities = list(expert_modalities)
        self.modality_dropout = float(modality_dropout)

        self.gate_type = str(gate_type)
        if self.gate_type not in ("scalar", "per_channel"):
            raise ValueError(f"Unknown gate_type='{self.gate_type}'. Expected 'scalar' or 'per_channel'.")
        self.gate_sees_delta = bool(gate_sees_delta)
        self.task_conditioned = bool(task_conditioned)
        self.tasks = list(tasks) if self.task_conditioned else None

        # anchor attention pool: a learned query attends to all RGB tokens -> h0
        self.anchor_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.anchor_pool = nn.MultiheadAttention(
            self.embed_dim, int(num_heads), dropout=float(dropout), batch_first=True)
        self.anchor_norm = nn.LayerNorm(self.embed_dim)

        # per-expert cross-attention correction (computed once, shared across
        # every gate-design variant and across tasks if task_conditioned)
        self.experts = nn.ModuleDict({
            m: _CrossAttnExpert(self.embed_dim, num_heads, ff_mult, dropout)
            for m in self.expert_modalities
        })

        gate_in_dim = self.embed_dim * (3 if self.gate_sees_delta else 2)
        gate_out_dim = self.embed_dim if self.gate_type == "per_channel" else 1

        def make_gate():
            gate = nn.Sequential(
                nn.Linear(gate_in_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, gate_out_dim),
            )
            # init gate ~ sigmoid(gate_bias_init) ~ 0 for every sample/channel
            # -> fused ≈ anchor at start, regardless of gate_type/task_conditioned.
            nn.init.zeros_(gate[-1].weight)
            nn.init.constant_(gate[-1].bias, float(gate_bias_init))
            return gate

        if self.task_conditioned:
            self.gates = nn.ModuleDict({
                f"{task}__{m}": make_gate()
                for task in self.tasks
                for m in self.expert_modalities
            })
        else:
            self.gates = nn.ModuleDict({m: make_gate() for m in self.expert_modalities})

        self.out_norm = nn.LayerNorm(self.embed_dim)
        nn.init.trunc_normal_(self.anchor_query, std=0.02)
        self.last_gates = None  # {modality: (B,)} or {task: {modality: (B,)}} for logging

    def _anchor(self, token_dict, mask_dict, device):
        toks, valids = [], []
        for m in self.anchor_modalities:
            if m not in token_dict or token_dict[m] is None:
                continue
            t = token_dict[m]
            toks.append(t)
            if m in mask_dict and mask_dict[m] is not None:
                valids.append(mask_dict[m].to(device=device, dtype=torch.bool))
            else:
                valids.append(torch.ones(t.shape[:2], dtype=torch.bool, device=device))
        if not toks:
            raise ValueError("RGBAnchoredFusion requires at least one anchor (RGB) modality.")
        anc = torch.cat(toks, dim=1)                 # (B, Na, D)
        anc_valid = torch.cat(valids, dim=1)         # (B, Na) True = valid
        b = anc.shape[0]
        q = self.anchor_query.expand(b, -1, -1)      # (B,1,D)
        kpm = _guard_full_pad(~anc_valid)
        h0, _ = self.anchor_pool(q, anc, anc, key_padding_mask=kpm, need_weights=False)
        return self.anchor_norm(h0.squeeze(1))       # (B,D)

    def _compute_gate(self, key, h0, summary, delta):
        gate_input = [h0, summary]
        if self.gate_sees_delta:
            gate_input.append(delta)
        raw = self.gates[key](torch.cat(gate_input, dim=-1))  # (B,1) or (B,D)
        g = torch.sigmoid(raw)
        if self.gate_type == "scalar":
            g = g.squeeze(-1)  # (B,)
        return g  # (B,) if scalar, (B,D) if per_channel

    def forward(self, token_dict, mask_dict=None, return_gates=False):
        mask_dict = mask_dict or {}
        any_tok = next(iter(token_dict.values()))
        b, device = any_tok.shape[0], any_tok.device

        h0 = self._anchor(token_dict, mask_dict, device)   # (B,D)

        # Δ_m, its masked-mean token summary, and modality-presence flag: computed
        # ONCE per expert modality, shared across every gate variant and every task.
        deltas, summaries, any_valids = {}, {}, {}
        for m in self.expert_modalities:
            if m not in token_dict or token_dict[m] is None:
                deltas[m] = None
                continue
            tok = token_dict[m]                            # (B,N,D)
            n = tok.shape[1]
            if m in mask_dict and mask_dict[m] is not None:
                valid = mask_dict[m].to(device=device, dtype=torch.bool)
            else:
                valid = torch.ones(b, n, dtype=torch.bool, device=device)

            # per-sample modality dropout (training only): drop the whole modality
            if self.training and self.modality_dropout > 0.0:
                drop = torch.rand(b, device=device) < self.modality_dropout
                valid = valid & (~drop).unsqueeze(1)

            any_valids[m] = valid.any(dim=1)                # (B,)
            deltas[m] = self.experts[m](h0, tok, _guard_full_pad(~valid))   # (B,D)

            denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
            summaries[m] = (tok * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B,D) masked mean

        def fuse_with(gate_key_fn):
            fused = h0
            gates_out = {}
            for m in self.expert_modalities:
                if deltas[m] is None:
                    gates_out[m] = torch.zeros(b, device=device)
                    continue
                g = self._compute_gate(gate_key_fn(m), h0, summaries[m], deltas[m])
                any_valid = any_valids[m].to(g.dtype)
                if self.gate_type == "per_channel":
                    g = g * any_valid.unsqueeze(-1)         # force 0 if fully invalid / dropped
                    fused = fused + g * deltas[m]
                else:
                    g = g * any_valid
                    fused = fused + g.unsqueeze(-1) * deltas[m]
                gates_out[m] = g.detach()
            return self.out_norm(fused), gates_out

        if self.task_conditioned:
            fused_by_task, gates_by_task = {}, {}
            for task in self.tasks:
                fused_t, gates_t = fuse_with(lambda m, task=task: f"{task}__{m}")
                fused_by_task[task] = fused_t
                gates_by_task[task] = gates_t
            self.last_gates = gates_by_task
            if return_gates:
                return fused_by_task, gates_by_task
            return fused_by_task

        fused, gates_out = fuse_with(lambda m: m)
        self.last_gates = gates_out
        if return_gates:
            return fused, gates_out
        return fused
