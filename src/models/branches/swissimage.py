import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# SwissImage default normalization (must match SwissimageTransform.DEFAULT_*);
# used to un-normalize the input before re-normalizing for the side CNN.
SWISSIMAGE_DEFAULT_MEAN = (0.430, 0.411, 0.296)
SWISSIMAGE_DEFAULT_STD = (0.213, 0.156, 0.143)


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, tokens, mask=None):
        # tokens: (B, N, C)
        attn_logits = self.score(tokens).squeeze(-1)

        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.bool(), float("-inf"))

        attn = torch.softmax(attn_logits, dim=-1)
        return torch.sum(tokens * attn[:, :, None], dim=1)


class SwissImageBranch(nn.Module):
    def __init__(
        self,
        dino,
        dino_dim=1024,
        embed_dim=256,
        season_dim=16,
        freeze_backbone=True,
        patch_size=16,
        use_season_film=True,
        pooling="attention",
        fusion="gated",
        return_debug=False,
        backbone_dtype=None,
        output_mode="fused",
        patch_pool_grid=8,
        side_tuning="none",
        side_input_mean=None,
        side_input_std=None,
        multilayer_blocks=None,
        multilayer_fusion="weighted_sum",
    ):
        super().__init__()
        self.output_mode = str(output_mode)  # "tokens" (V2) | "fused" (legacy)
        self.patch_pool_grid = int(patch_pool_grid)
        self.side_tuning = str(side_tuning)  # "none" | timm model name (e.g. "edgenext_small")
        # Multi-layer DINOv3 feature fusion: instead of only the last transformer
        # block's output (forward_features), read several intermediate blocks'
        # outputs (already computed for free in the same forward pass -> ~zero
        # extra compute) and fuse them into one (cls, patch) pair BEFORE the rest
        # of the pipeline (season fusion, pooling, tokenizer) which is unchanged.
        # None/[] -> exact original single-last-layer behaviour (backward compat).
        self.multilayer_blocks = list(multilayer_blocks) if multilayer_blocks else None
        self.multilayer_fusion = str(multilayer_fusion)
        if self.multilayer_blocks and self.multilayer_fusion not in ("weighted_sum", "attention"):
            raise ValueError(
                f"Unknown multilayer_fusion='{self.multilayer_fusion}'. Expected 'weighted_sum' or 'attention'."
            )
        valid_pooling = ["mean", "attention"]
        valid_fusion = ["summer_only", "mean", "gated"]
        if pooling not in valid_pooling:
            raise ValueError(f"Unknown pooling='{pooling}'. Expected one of {valid_pooling}.")
        if fusion not in valid_fusion:
            raise ValueError(f"Unknown fusion='{fusion}'. Expected one of {valid_fusion}.")

        self.dino = dino
        self.dino_dim = int(dino_dim)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.use_season_film = bool(use_season_film)
        self.pooling = pooling
        self.fusion = fusion
        self.return_debug = bool(return_debug)

        if freeze_backbone:
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False

        self.freeze_backbone = bool(freeze_backbone)

        # Optionally store the (frozen) backbone in a lower-precision dtype to fit
        # very large checkpoints (e.g. DINOv3 ViT-7B ~27GB fp32 -> ~13GB bf16).
        # Only sensible when frozen; the adapter/rest of the model stays fp32 and
        # features are cast back to fp32 in _dino_forward.
        self._dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.backbone_dtype = None
        if backbone_dtype is not None and str(backbone_dtype) not in ("", "float32", "None"):
            if not self.freeze_backbone:
                raise ValueError("backbone_dtype casting is only supported for a frozen backbone.")
            self.backbone_dtype = self._dtype_map[str(backbone_dtype)]
            self.dino.to(self.backbone_dtype)

        self.token_adapter = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.cls_adapter = nn.Sequential(
            nn.LayerNorm(self.dino_dim),
            nn.Linear(self.dino_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        if self.use_season_film:
            self.season_embed = nn.Embedding(2, int(season_dim))
            self.film = nn.Linear(int(season_dim), self.embed_dim * 2)
        else:
            self.season_embed = None
            self.film = None

        if self.pooling == "attention":
            self.attn_pool = AttentionPooling(self.embed_dim)
        else:
            self.attn_pool = None

        self.view_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        if self.fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        # --- V2 token mode: shared DINO projection + explicit seasonal fusion ---
        # Each view first becomes cls(1) + avgpool(patch_pool_grid^2) patch tokens.
        # Summer/winter are then fused inside the RGB branch into one RGB token set
        # so the multimodal fusion sees RGB as one anchor modality.
        if self.output_mode == "tokens":
            if self.multilayer_blocks:
                depth = len(self.dino.blocks)
                bad = [i for i in self.multilayer_blocks if i < 0 or i >= depth]
                if bad:
                    raise ValueError(
                        f"multilayer_blocks {bad} out of range for backbone with {depth} blocks "
                        f"(valid indices: 0..{depth - 1})."
                    )
                n_layers = len(self.multilayer_blocks)
                if self.multilayer_fusion == "weighted_sum":
                    # learnable per-layer softmax weight, shared token_proj projects
                    # each layer to embed_dim first -> convex combination in embed_dim
                    # space. Init at 0 -> uniform average at the start of training.
                    self.multilayer_patch_weights = nn.Parameter(torch.zeros(n_layers))
                    self.multilayer_cls_weights = nn.Parameter(torch.zeros(n_layers))
                else:  # "attention"
                    # content-based attention pooling OVER THE LAYER AXIS (shared
                    # across all patch positions); reuses the same AttentionPooling
                    # block used elsewhere in this file for season/view pooling.
                    self.multilayer_patch_attn = AttentionPooling(self.embed_dim)
                    self.multilayer_cls_attn = AttentionPooling(self.embed_dim)

            self.token_proj = nn.Sequential(
                nn.LayerNorm(self.dino_dim),
                nn.Linear(self.dino_dim, self.embed_dim),
            )
            self.n_rgb_tokens = 1 + self.patch_pool_grid * self.patch_pool_grid
            self.rgb_pos = nn.Parameter(torch.zeros(1, self.n_rgb_tokens, self.embed_dim))
            nn.init.trunc_normal_(self.rgb_pos, std=0.02)
            self.rgb_patch_fuse_in = nn.Conv2d(self.embed_dim * 3, self.embed_dim, kernel_size=1)
            self.rgb_patch_fuse_dw = nn.Conv2d(
                self.embed_dim,
                self.embed_dim,
                kernel_size=3,
                padding=1,
                groups=self.embed_dim,
            )
            self.rgb_patch_fuse_out = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)
            self.rgb_cls_fuse = nn.Sequential(
                nn.LayerNorm(self.embed_dim * 3),
                nn.Linear(self.embed_dim * 3, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )

            # --- optional side-tuning CNN (ImageNet-pretrained, trainable) ---
            # The frozen DINOv3 patch tokens are enriched additively at the same
            # 16x16 grid (token count unchanged). The side CNN needs ImageNet
            # normalization regardless of the DINOv3 norm, so we un-normalize the
            # input (with side_input_mean/std) and re-normalize to ImageNet.
            if self.side_tuning != "none":
                import timm
                self.side_cnn = timm.create_model(self.side_tuning, pretrained=True, features_only=True)
                reds = self.side_cnn.feature_info.reduction()
                if 16 not in reds:
                    raise ValueError(f"side_tuning model '{self.side_tuning}' has no stride-16 stage; strides={reds}")
                self.side_idx = reds.index(16)
                side_ch = self.side_cnn.feature_info.channels()[self.side_idx]
                self.side_proj = nn.Linear(side_ch, self.embed_dim)
                self.side_alpha = nn.Parameter(torch.zeros(1))  # gated, init 0 -> starts as pure DINOv3
                in_mean = side_input_mean if side_input_mean is not None else SWISSIMAGE_DEFAULT_MEAN
                in_std = side_input_std if side_input_std is not None else SWISSIMAGE_DEFAULT_STD
                self.register_buffer("side_in_mean", torch.tensor(in_mean).view(1, 3, 1, 1))
                self.register_buffer("side_in_std", torch.tensor(in_std).view(1, 3, 1, 1))
                self.register_buffer("side_imnet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
                self.register_buffer("side_imnet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, rgb):
        if self.output_mode == "tokens":
            return self._forward_tokens(rgb)

        images, season_id, view_mask = self._prepare_inputs(rgb)
        b, v, c, h, w = images.shape

        flat_images = images.reshape(b * v, c, h, w)
        flat_season_id = season_id.reshape(b * v)

        cls_token, patch_tokens, grid_hw = self._dino_forward(flat_images)

        patch_tokens = self.token_adapter(patch_tokens)
        cls_token = self.cls_adapter(cls_token)

        patch_tokens, cls_token = self._apply_season_film(
            patch_tokens,
            cls_token,
            flat_season_id,
        )

        view_tokens = self._make_view_tokens(cls_token, patch_tokens)

        ht, wt = grid_hw
        spatial = patch_tokens.transpose(1, 2).reshape(
            b * v,
            self.embed_dim,
            ht,
            wt,
        )

        view_spatial = spatial.reshape(b, v, self.embed_dim, ht, wt)
        view_tokens = view_tokens.reshape(b, v, self.embed_dim)

        rgb_spatial, rgb_token, alpha = self._fuse_views(
            view_spatial,
            view_tokens,
            view_mask,
        )

        out = {
            "rgb_spatial": rgb_spatial,
            "rgb_token": rgb_token,
        }

        if self.return_debug:
            out.update({
                "rgb_view_spatial": view_spatial,
                "rgb_view_tokens": view_tokens,
                "rgb_view_mask": view_mask,
                "rgb_fusion_gate": alpha,
            })

        return out

    def _forward_tokens(self, rgb):
        images, season_id, view_mask = self._prepare_inputs(rgb)
        b, v, c, h, w = images.shape
        flat = images.reshape(b * v, c, h, w)

        if self.multilayer_blocks:
            patch_list, cls_list, grid_hw = self._dino_forward_multilayer(flat)
            patch_list = [self.token_proj(p) for p in patch_list]  # each (B*V, N, D)
            cls_list = [self.token_proj(c) for c in cls_list]      # each (B*V, D)
            patch_tokens, cls_token = self._fuse_layers(patch_list, cls_list)
        else:
            cls_token, patch_tokens, grid_hw = self._dino_forward(flat)
            cls_token = self.token_proj(cls_token)        # (B*V, D)
            patch_tokens = self.token_proj(patch_tokens)  # (B*V, N, D)

        # side-tuning: enrich the 16x16 patch tokens additively (token count unchanged)
        if self.side_tuning != "none":
            raw = flat * self.side_in_std + self.side_in_mean             # un-normalize input
            side_in = (raw - self.side_imnet_mean) / self.side_imnet_std  # -> ImageNet norm for the side CNN
            side_feat = self.side_cnn(side_in)[self.side_idx]             # (B*V, side_ch, 16, 16)
            side = side_feat.flatten(2).transpose(1, 2)                   # (B*V, 256, side_ch)
            patch_tokens = patch_tokens + torch.tanh(self.side_alpha) * self.side_proj(side)

        ht, wt = grid_hw
        g = self.patch_pool_grid
        spatial = patch_tokens.transpose(1, 2).reshape(b * v, self.embed_dim, ht, wt)
        pooled = F.adaptive_avg_pool2d(spatial, (g, g))            # (B*V, D, g, g)
        pooled = pooled.flatten(2).transpose(1, 2)                 # (B*V, g*g, D)

        tokens = torch.cat([cls_token.unsqueeze(1), pooled], dim=1)  # (B*V, 1+g*g, D)
        tokens = tokens.reshape(b, v, self.n_rgb_tokens, self.embed_dim)
        fused_tokens = self._fuse_season_tokens(tokens, view_mask) + self.rgb_pos

        return {
            "rgb_tokens": fused_tokens,      # (B, 65, D)
            "rgb_token_mask": torch.ones(b, self.n_rgb_tokens, dtype=torch.bool, device=tokens.device),
            "rgb_view_tokens": tokens,       # (B, V, 65, D), kept for debugging/inspection
            "rgb_view_mask": view_mask,      # (B, V) True = valid
            "rgb_season_id": season_id,      # (B, V) 0=summer, 1=winter
        }

    def _fuse_season_tokens(self, tokens, view_mask):
        b, v, n, d = tokens.shape
        if n != self.n_rgb_tokens:
            raise ValueError(f"Expected {self.n_rgb_tokens} RGB tokens, got {n}.")

        if v == 1:
            return tokens[:, 0]

        g = self.patch_pool_grid
        if n != 1 + g * g:
            raise ValueError(f"patch_pool_grid={g} is inconsistent with token count {n}.")

        summer = tokens[:, 0]
        winter = tokens[:, 1]
        has_summer = view_mask[:, 0].bool()
        has_winter = view_mask[:, 1].bool()
        has_both = has_summer & has_winter

        cls_input = torch.cat(
            [summer[:, 0], winter[:, 0], winter[:, 0] - summer[:, 0]],
            dim=-1,
        )
        fused_cls = self.rgb_cls_fuse(cls_input)

        summer_grid = summer[:, 1:].reshape(b, g, g, d).permute(0, 3, 1, 2)
        winter_grid = winter[:, 1:].reshape(b, g, g, d).permute(0, 3, 1, 2)
        patch_input = torch.cat([summer_grid, winter_grid, winter_grid - summer_grid], dim=1)
        x = F.gelu(self.rgb_patch_fuse_in(patch_input))
        x = x + self.rgb_patch_fuse_dw(x)
        x = F.gelu(x)
        fused_grid = self.rgb_patch_fuse_out(x)
        fused_patches = fused_grid.flatten(2).transpose(1, 2)
        fused = torch.cat([fused_cls.unsqueeze(1), fused_patches], dim=1)

        fallback = torch.where(has_summer[:, None, None], summer, winter)
        return torch.where(has_both[:, None, None], fused, fallback)

    def _dino_forward(self, images):
        adapter_dtype = self.token_adapter[1].weight.dtype
        if self.backbone_dtype is not None:
            images = images.to(self.backbone_dtype)

        if self.freeze_backbone:
            with torch.no_grad():
                features = self.dino.forward_features(images)
        else:
            features = self.dino.forward_features(images)

        # Cast features back to the (fp32) adapter dtype so the rest of the model
        # is unaffected by a low-precision backbone.
        cls_token = features["x_norm_clstoken"].to(adapter_dtype)
        patch_tokens = features["x_norm_patchtokens"].to(adapter_dtype)

        n_tokens = patch_tokens.shape[1]
        grid_size = int(n_tokens ** 0.5)
        if grid_size * grid_size != n_tokens:
            raise ValueError(
                f"DINO patch token count must form a square grid, got {n_tokens}."
            )

        return cls_token, patch_tokens, (grid_size, grid_size)

    def _dino_forward_multilayer(self, images):
        """Like _dino_forward, but returns EVERY requested block's (patch, cls)
        output instead of only the last block's. get_intermediate_layers runs the
        exact same sequential block-by-block forward pass as forward_features (all
        blocks are computed regardless, to reach the last one) -- it just also
        retains the intermediate activations we ask for, so this costs ~zero extra
        compute vs a normal single-layer forward, only a bit of transient memory.
        """
        adapter_dtype = self.token_adapter[1].weight.dtype
        if self.backbone_dtype is not None:
            images = images.to(self.backbone_dtype)

        ctx = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with ctx:
            layers = self.dino.get_intermediate_layers(
                images,
                n=self.multilayer_blocks,
                return_class_token=True,
                reshape=False,
                norm=True,
            )

        patch_list = [patch.to(adapter_dtype) for patch, _ in layers]
        cls_list = [cls.to(adapter_dtype) for _, cls in layers]

        n_tokens = patch_list[0].shape[1]
        grid_size = int(n_tokens ** 0.5)
        if grid_size * grid_size != n_tokens:
            raise ValueError(
                f"DINO patch token count must form a square grid, got {n_tokens}."
            )

        return patch_list, cls_list, (grid_size, grid_size)

    def _fuse_layers(self, patch_list, cls_list):
        """Fuse L per-layer (patch, cls) pairs (each already projected to
        embed_dim) into one (patch, cls) pair, same shape as the single-layer path.
        """
        if self.multilayer_fusion == "weighted_sum":
            pw = torch.softmax(self.multilayer_patch_weights, dim=0)
            cw = torch.softmax(self.multilayer_cls_weights, dim=0)
            patch_tokens = sum(pw[i] * patch_list[i] for i in range(len(patch_list)))
            cls_token = sum(cw[i] * cls_list[i] for i in range(len(cls_list)))
            return patch_tokens, cls_token

        # "attention": content-based attention pooling over the layer axis,
        # shared across all patch positions.
        bv, n, d = patch_list[0].shape
        stacked_patch = torch.stack(patch_list, dim=2)          # (B*V, N, L, D)
        flat_patch = stacked_patch.reshape(bv * n, len(patch_list), d)
        patch_tokens = self.multilayer_patch_attn(flat_patch).reshape(bv, n, d)

        stacked_cls = torch.stack(cls_list, dim=1)              # (B*V, L, D)
        cls_token = self.multilayer_cls_attn(stacked_cls)

        return patch_tokens, cls_token

    def _apply_season_film(self, patch_tokens, cls_token, season_id):
        if not self.use_season_film:
            return patch_tokens, cls_token

        season_id = season_id.clamp(min=0, max=1)
        season = self.season_embed(season_id)
        gamma_beta = self.film(season)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = 1.0 + gamma

        patch_tokens = gamma[:, None, :] * patch_tokens + beta[:, None, :]
        cls_token = gamma * cls_token + beta

        return patch_tokens, cls_token

    def _make_view_tokens(self, cls_token, patch_tokens):
        if self.pooling == "attention":
            pooled = self.attn_pool(patch_tokens)
        elif self.pooling == "mean":
            pooled = patch_tokens.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling='{self.pooling}'.")

        return self.view_mlp(torch.cat([cls_token, pooled], dim=-1))

    def _fuse_views(self, view_spatial, view_tokens, view_mask):
        b, v, c, h, w = view_spatial.shape

        # dataset.yaml controls whether one or two RGB views are returned.
        # With multiseason=false, v == 1 and fusion naturally becomes summer-only.
        if v == 1:
            alpha = torch.ones(b, c, device=view_tokens.device, dtype=view_tokens.dtype)
            return view_spatial[:, 0], view_tokens[:, 0], alpha

        summer_spatial = view_spatial[:, 0]
        winter_spatial = view_spatial[:, 1]
        summer_token = view_tokens[:, 0]
        winter_token = view_tokens[:, 1]

        has_summer = view_mask[:, 0].bool()
        has_winter = view_mask[:, 1].bool()
        has_both = has_summer & has_winter

        if self.fusion == "summer_only":
            alpha = torch.ones(b, c, device=view_tokens.device, dtype=view_tokens.dtype)
            return summer_spatial, summer_token, alpha

        if self.fusion == "mean":
            weights = view_mask.float()
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            rgb_token = torch.sum(view_tokens * weights[:, :, None], dim=1)
            rgb_spatial = torch.sum(
                view_spatial * weights[:, :, None, None, None],
                dim=1,
            )
            alpha = weights[:, 0:1].expand(b, c)
            return rgb_spatial, rgb_token, alpha

        if self.fusion != "gated":
            raise ValueError(f"Unknown fusion='{self.fusion}'.")

        alpha = self.gate(torch.cat([summer_token, winter_token], dim=-1))
        alpha_spatial = alpha[:, :, None, None]

        fused_token = alpha * summer_token + (1.0 - alpha) * winter_token
        fused_spatial = (
            alpha_spatial * summer_spatial
            + (1.0 - alpha_spatial) * winter_spatial
        )

        token_without_both = torch.where(
            has_summer[:, None],
            summer_token,
            winter_token,
        )
        spatial_without_both = torch.where(
            has_summer[:, None, None, None],
            summer_spatial,
            winter_spatial,
        )

        rgb_token = torch.where(has_both[:, None], fused_token, token_without_both)
        rgb_spatial = torch.where(
            has_both[:, None, None, None],
            fused_spatial,
            spatial_without_both,
        )

        return rgb_spatial, rgb_token, alpha

    def _prepare_inputs(self, rgb):
        if isinstance(rgb, dict):
            images = rgb["data"]
            season_id = rgb.get("season_id")
            view_mask = rgb.get("view_mask")
            is_fallback = rgb.get("is_fallback")
        else:
            images = rgb
            season_id = None
            view_mask = None
            is_fallback = None

        if images.ndim == 4:
            images = images[:, None, :, :, :]
        elif images.ndim != 5:
            raise ValueError(
                f"Expected RGB shape (B, 3, H, W) or (B, V, 3, H, W), "
                f"got {tuple(images.shape)}."
            )

        b, v = images.shape[:2]
        device = images.device

        default_season = torch.arange(v, device=device).clamp(max=1)
        default_season = default_season[None, :].expand(b, v)

        if season_id is None:
            season_id = default_season
        else:
            season_id = season_id.to(device)
            if season_id.ndim == 1:
                season_id = season_id[:, None]
            season_id = torch.where(
                (season_id >= 0) & (season_id <= 1),
                season_id,
                default_season,
            )

        if view_mask is None:
            if is_fallback is not None:
                is_fallback = is_fallback.to(device)
                if is_fallback.ndim == 1:
                    is_fallback = is_fallback[:, None]
                view_mask = ~is_fallback.bool()
                view_mask[:, 0] = True
            else:
                view_mask = torch.ones(b, v, dtype=torch.bool, device=device)
        else:
            view_mask = view_mask.to(device).bool()
            if view_mask.ndim == 1:
                view_mask = view_mask[:, None]

        if not view_mask.any(dim=1).all():
            raise ValueError("Each sample must have at least one available SwissImage view.")

        return images.float(), season_id.long(), view_mask


def build_dinov3_backbone(
    repo_path,
    weights_path,
    backbone_name="dinov3_vith16plus",
    pretrained=False,
):
    repo_path = Path(repo_path).expanduser().resolve()
    weights_path = Path(weights_path).expanduser().resolve() if weights_path else None

    if not repo_path.exists():
        raise FileNotFoundError(f"DINOv3 repo not found: {repo_path}")

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    from dinov3.hub import backbones

    if not hasattr(backbones, backbone_name):
        raise ValueError(f"Unknown DINOv3 backbone: {backbone_name}")

    builder = getattr(backbones, backbone_name)
    model = builder(pretrained=bool(pretrained), weights=str(weights_path) if weights_path else None)

    if weights_path is not None and not bool(pretrained):
        if not weights_path.exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        # LVD1689M (web) checkpoints carry training-only keys (e.g. local_cls_norm)
        # absent from the inference backbone. Allow extra keys, but require that
        # every backbone parameter is actually loaded (no missing keys).
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"DINOv3 weights missing {len(missing)} keys for {weights_path}: {missing[:5]}"
            )

    return model
