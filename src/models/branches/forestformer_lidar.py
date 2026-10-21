"""LiDAR token encoder adapted from ForestFormer3D.

ResidualBlock and SpConvUNet were adapted from the original
oneformer3d/spconv_unet.py, with the mmdet3d registry dependency removed.
Source: https://bxiang233.github.io/FF3D/
The original license is retained in LICENSE.ForestFormer3D in this directory.

The reference model trains the sparse U-Net from random initialization, uses
0.1 m voxels, and pools voxel features into 64 learned query tokens.
"""
import functools
import logging
from collections import OrderedDict

import torch
from torch import nn

logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)

import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule

from src.models.layers import LearnedQueryPooling, pad_per_sample


class ResidualBlock(SparseModule):
    """Verbatim from ForestFormer3D spconv_unet.ResidualBlock (names preserved)."""

    def __init__(self, in_channels, out_channels,
                 norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key=None, normalize_before=True):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False))

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels), nn.ReLU(),
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1,
                                  bias=False, indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1,
                                  bias=False, indice_key=indice_key))
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1,
                                  bias=False, indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1,
                                  bias=False, indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU())

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices,
                                           input.spatial_shape, input.batch_size)
        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        
        return output


class SpConvUNet(nn.Module):
    """Verbatim from ForestFormer3D spconv_unet.SpConvUNet (names preserved)."""

    def __init__(self, num_planes,
                 norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 block_reps=2, block=ResidualBlock, indice_key_id=1,
                 normalize_before=True, return_blocks=False):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes

        blocks = {
            f'block{i}': block(num_planes[0], num_planes[0], norm_fn,
                               normalize_before=normalize_before,
                               indice_key=f'subm{indice_key_id}')
            for i in range(block_reps)
        }
        self.blocks = spconv.SparseSequential(OrderedDict(blocks))

        if len(num_planes) > 1:
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(num_planes[0]), nn.ReLU(),
                    spconv.SparseConv3d(num_planes[0], num_planes[1], kernel_size=2,
                                        stride=2, bias=False, indice_key=f'spconv{indice_key_id}'))
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(num_planes[0], num_planes[1], kernel_size=2,
                                        stride=2, bias=False, indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[1]), nn.ReLU())

            self.u = SpConvUNet(num_planes[1:], norm_fn, block_reps, block,
                                indice_key_id=indice_key_id + 1,
                                normalize_before=normalize_before, return_blocks=return_blocks)

            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(num_planes[1]), nn.ReLU(),
                    spconv.SparseInverseConv3d(num_planes[1], num_planes[0], kernel_size=2,
                                               bias=False, indice_key=f'spconv{indice_key_id}'))
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(num_planes[1], num_planes[0], kernel_size=2,
                                               bias=False, indice_key=f'spconv{indice_key_id}'),
                    norm_fn(num_planes[0]), nn.ReLU())

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f'block{i}'] = block(num_planes[0] * (2 - i), num_planes[0], norm_fn,
                                                 indice_key=f'subm{indice_key_id}',
                                                 normalize_before=normalize_before)
            self.blocks_tail = spconv.SparseSequential(OrderedDict(blocks_tail))

    def forward(self, input, previous_outputs=None):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices,
                                           output.spatial_shape, output.batch_size)

        if len(self.num_planes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)
            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)

        if self.return_blocks:
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        return output


class ForestFormerLidarBranch(nn.Module):
    """Sparse U-Net encoder with configurable freezing and site-level token pooling."""

    # Valid stage names for `freeze_stages`, from shallowest to deepest. "levelK" is
    # the SpConvUNet recursion depth K's OWN blocks/conv/deconv/blocks_tail params
    # (K=0 is the outermost/shallowest U-Net level, channel-narrowest; K=len(num_planes)-1
    # is the bottleneck, channel-widest). input_conv/output_layer are the outer stem/head.
    STAGE_NAMES = ("input_conv", "level0", "level1", "level2", "level3", "level4", "output_layer")

    def __init__(self, weights_path, embed_dim=256, in_channels=3, num_planes=None,
                 block_reps=2, voxel_size=0.2, crop_radius_m=14.0, min_spatial_shape=128,
                 bev_grid_size=16, freeze=True, freeze_stages=None, return_debug=False,
                 output_mode="tokens", num_query_tokens=64, tokenizer_layers=2, num_heads=8,
                 ground_mask=False, ground_class=2):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.in_channels = int(in_channels)
        self.voxel_size = float(voxel_size)
        # ISA-inspired ground masking: drop LAS-class `ground_class` (2=ground)
        # voxels from the learnable-query pooling so LiDAR tokens summarise canopy,
        # not terrain. Uses the classification already loaded in batch["lidar"].
        self.ground_mask = bool(ground_mask)
        self.ground_class = int(ground_class)
        self.crop_radius_m = float(crop_radius_m)
        self.min_spatial_shape = int(min_spatial_shape)
        self.bev_grid_size = int(bev_grid_size)
        self.freeze = bool(freeze)
        # Partial fine-tuning: freeze only specific stages (by name, see STAGE_NAMES),
        # leaving the rest trainable. Ignored if `freeze=True` (that already freezes
        # everything). Mutually exclusive with `freeze` in practice, not enforced.
        self.freeze_stages = [str(s) for s in freeze_stages] if freeze_stages else []
        for s in self.freeze_stages:
            if s not in self.STAGE_NAMES:
                raise ValueError(f"Unknown freeze_stages entry '{s}'. Expected one of {self.STAGE_NAMES}")
        self.return_debug = bool(return_debug)
        self.output_mode = str(output_mode)  # "tokens" (V2) | "bev" (legacy)

        if num_planes is None:
            num_planes = [32 * (i + 1) for i in range(5)]  # ForestFormer: num_channels=32
        self.num_planes = [int(c) for c in num_planes]

        # --- encoder (frozen; names MUST match the checkpoint) ---
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(self.in_channels, self.num_planes[0], kernel_size=3,
                              padding=1, bias=False, indice_key="subm1"))
        self.unet = SpConvUNet(self.num_planes, block_reps=int(block_reps), return_blocks=True)
        self.output_layer = spconv.SparseSequential(
            nn.BatchNorm1d(self.num_planes[0], eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True))

        # weights_path=None -> same architecture, random init, trained from scratch
        # (better matches airborne LiDAR density than the ForAINetV2-pretrained frozen one)
        if weights_path is not None:
            self._load_pretrained(weights_path)

        if self.freeze:
            for m in (self.input_conv, self.unet, self.output_layer):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        elif self.freeze_stages:
            self._apply_partial_freeze()

        # --- trainable head ---
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_planes[0]),
            nn.Linear(self.num_planes[0], self.embed_dim))

        if self.output_mode == "tokens":
            self.num_query_tokens = int(num_query_tokens)
            self.tokenizer = LearnedQueryPooling(
                embed_dim=self.embed_dim,
                num_queries=self.num_query_tokens,
                num_layers=int(tokenizer_layers),
                num_heads=int(num_heads),
            )

    @staticmethod
    def _stage_of(param_name):
        """Map a `self`-relative param/module name to its STAGE_NAMES entry.

        "unet.blocks...." / "unet.conv...." / "unet.deconv...." / "unet.blocks_tail...."
        -> level0 (that SpConvUNet's OWN params). Each ".u." segment right after "unet"
        descends one recursion level (level0 -> level1 -> ... -> the bottleneck), mirroring
        SpConvUNet's `self.u = SpConvUNet(...)` nesting.
        """
        parts = param_name.split(".")
        if parts[0] in ("input_conv", "output_layer"):
            return parts[0]
        if parts[0] != "unet":
            return None
        depth, i = 0, 1
        while i < len(parts) and parts[i] == "u":
            depth += 1
            i += 1
        return f"level{depth}"

    def _apply_partial_freeze(self):
        frozen = set(self.freeze_stages)
        n_frozen = n_total = 0
        for name, p in self.named_parameters():
            n_total += p.numel()
            if self._stage_of(name) in frozen:
                p.requires_grad = False
                n_frozen += p.numel()
        # Keep frozen stages' BatchNorm running stats frozen (eval mode) even in train().
        self._frozen_bn_modules = [
            m for name, m in self.named_modules()
            if isinstance(m, nn.BatchNorm1d) and self._stage_of(name) in frozen
        ]
        for m in self._frozen_bn_modules:
            m.eval()
        logging.getLogger(__name__).info(
            "ForestFormerLidarBranch partial freeze: stages=%s -> %d/%d encoder params frozen (%.1f%%)",
            self.freeze_stages, n_frozen, n_total, 100.0 * n_frozen / max(n_total, 1),
        )

    def _load_pretrained(self, weights_path):
        sd = torch.load(weights_path, map_location="cpu")
        # The epoch_3000_fix.pth conv weights are in (k,k,k,in,out) layout (the
        # spconv version it was 'fixed' for). spconv 2.3.8 SubMConv/SparseConv3d
        # expect (out,k,k,k,in), so undo it: 5D weight -> permute(4,0,1,2,3).
        sd = {k: (v.permute(4, 0, 1, 2, 3).contiguous() if v.dim() == 5 else v)
              for k, v in sd.items()}
        target = {}
        target.update({f"input_conv.{k}": v for k, v in self._sub(sd, "input_conv").items()})
        target.update({f"unet.{k}": v for k, v in self._sub(sd, "unet").items()})
        target.update({f"output_layer.{k}": v for k, v in self._sub(sd, "output_layer").items()})
        own = {**{f"input_conv.{k}": None for k in self.input_conv.state_dict()},
               **{f"unet.{k}": None for k in self.unet.state_dict()},
               **{f"output_layer.{k}": None for k in self.output_layer.state_dict()}}
        missing = [k for k in own if k not in target]
        unexpected = [k for k in target if k not in own]
        if missing or unexpected:
            raise RuntimeError(
                f"ForestFormer encoder weight mismatch. missing={len(missing)} "
                f"unexpected={len(unexpected)}\n  missing[:5]={missing[:5]}\n  unexpected[:5]={unexpected[:5]}")
        self.input_conv.load_state_dict(self._sub(sd, "input_conv"), strict=True)
        self.unet.load_state_dict(self._sub(sd, "unet"), strict=True)
        self.output_layer.load_state_dict(self._sub(sd, "output_layer"), strict=True)

    @staticmethod
    def _sub(sd, prefix):
        p = prefix + "."
        return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:  # keep frozen BN/encoder in eval so running stats are used
            for m in (self.input_conv, self.unet, self.output_layer):
                m.eval()
        elif self.freeze_stages:
            for m in getattr(self, "_frozen_bn_modules", []):
                m.eval()
        return self

    def forward(self, batch):
        if "lidar" not in batch or batch["lidar"] is None:
            raise KeyError("ForestFormerLidarBranch requires batch['lidar'].")
        points = batch["lidar"]["points"]
        classification = batch["lidar"].get("classification") if self.ground_mask else None
        sparse_tensor, voxel_xy, voxel_ground = self._points_to_sparse_tensor(points, classification)

        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            x = self.input_conv(sparse_tensor)
            x, _ = self.unet(x)
            x = self.output_layer(x)
        feats = x.features
        if self.freeze:
            feats = feats.detach()

        features = self.proj(feats)          # (sumM, embed_dim)
        batch_idx = x.indices[:, 0].long()

        if self.output_mode == "tokens":
            padded, pad_mask = pad_per_sample(features, batch_idx, len(points))  # (B,Mmax,D),(B,Mmax) True=PAD
            key_padding_mask = pad_mask
            if self.ground_mask:
                # scatter the per-voxel ground flag into the same padded layout,
                # then also mask ground voxels out of the pooling attention.
                gpad, _ = pad_per_sample(voxel_ground.to(features.dtype).unsqueeze(1), batch_idx, len(points))
                ground = gpad.squeeze(-1) > 0.5                                  # (B,Mmax) True=ground
                masked = pad_mask | ground
                # guard: never fully mask a sample (all-ground crop) -> attn NaN.
                # for rows left empty by ground-masking, fall back to pad_mask only.
                fully_masked = masked.all(dim=1) & (~pad_mask).any(dim=1)
                masked[fully_masked] = pad_mask[fully_masked]
                key_padding_mask = masked
            lidar_tokens = self.tokenizer(padded, key_padding_mask=key_padding_mask)  # (B,K,D)
            out = {"lidar_tokens": lidar_tokens}
            if self.return_debug:
                out["lidar_sparse_features"] = features
            return out

        # (bev path does not use ground masking; voxel_ground unused here)
        lidar_spatial, bev_count = self._sparse_to_bev(
            features=features, voxel_xy=voxel_xy,
            batch_idx=batch_idx, batch_size=len(points))
        out = {"lidar_spatial": lidar_spatial}
        if self.return_debug:
            out.update({"lidar_bev_count": bev_count, "lidar_sparse_features": features})
        return out

    def _points_to_sparse_tensor(self, points_list, classification_list=None):
        voxel_features, voxel_coords, voxel_xy, voxel_ground = [], [], [], []
        max_coord = None
        for batch_idx, points in enumerate(points_list):
            points = points[:, :3].float()
            if points.numel() == 0:
                raise ValueError(f"Empty LiDAR point tensor at batch index {batch_idx}.")
            # ForestFormer feature = cloud-mean-centered XYZ
            centered = points - points.mean(dim=0, keepdim=True)
            # ForestFormer sparse coordinates = cloud-min-shifted XYZ / voxel_size.
            coord_points = points - points.min(dim=0, keepdim=True).values
            voxel = torch.floor(coord_points / self.voxel_size).long()

            unique_voxel, inverse, counts = torch.unique(
                voxel, dim=0, return_inverse=True, return_counts=True)
            feat_sum = centered.new_zeros((unique_voxel.shape[0], self.in_channels))
            feat_sum.index_add_(0, inverse, centered[:, :self.in_channels])
            feat_mean = feat_sum / counts[:, None].to(centered.dtype)

            # Per-voxel ground fraction (LAS class 2 = ground). A voxel is "ground"
            # if the majority of its points are ground -> later masked out of pooling
            # so the learnable LiDAR queries attend to canopy/vegetation, not terrain.
            if classification_list is not None:
                cls = classification_list[batch_idx].to(points.device).long()
                is_ground = (cls == self.ground_class).to(centered.dtype)     # (N,)
                gsum = centered.new_zeros((unique_voxel.shape[0],))
                gsum.index_add_(0, inverse, is_ground)
                ground_frac = gsum / counts.to(centered.dtype)
                voxel_ground.append(ground_frac > 0.5)
            else:
                voxel_ground.append(torch.zeros(unique_voxel.shape[0], dtype=torch.bool, device=points.device))

            batch_col = torch.full((unique_voxel.shape[0], 1), batch_idx,
                                   dtype=torch.long, device=points.device)
            voxel_features.append(feat_mean)
            voxel_coords.append(torch.cat([batch_col, unique_voxel], dim=1))
            voxel_xy.append(feat_mean[:, :2])
            cmax = unique_voxel.max(dim=0).values
            max_coord = cmax if max_coord is None else torch.maximum(max_coord, cmax)

        features = torch.cat(voxel_features, dim=0)
        coords = torch.cat(voxel_coords, dim=0).to(torch.int32)
        voxel_xy = torch.cat(voxel_xy, dim=0)
        voxel_ground = torch.cat(voxel_ground, dim=0)
        spatial_shape = torch.clamp(max_coord + 1, min=self.min_spatial_shape)
        spatial_shape = [int(v) for v in spatial_shape.tolist()]
        return (spconv.SparseConvTensor(features=features, indices=coords,
                                        spatial_shape=spatial_shape,
                                        batch_size=len(points_list)),
                voxel_xy, voxel_ground)

    def _sparse_to_bev(self, features, voxel_xy, batch_idx, batch_size):
        grid_size, radius = self.bev_grid_size, self.crop_radius_m
        x, y = voxel_xy[:, 0], voxel_xy[:, 1]
        x_bin = torch.floor((x + radius) / (2.0 * radius) * grid_size).long().clamp(0, grid_size - 1)
        y_bin = torch.floor((radius - y) / (2.0 * radius) * grid_size).long().clamp(0, grid_size - 1)
        flat_idx = batch_idx * (grid_size * grid_size) + y_bin * grid_size + x_bin
        bev = features.new_zeros((batch_size * grid_size * grid_size, features.shape[1]))
        count = features.new_zeros((batch_size * grid_size * grid_size, 1))
        bev.index_add_(0, flat_idx, features)
        count.index_add_(0, flat_idx, torch.ones_like(features[:, :1]))
        bev = (bev / count.clamp_min(1.0)).view(batch_size, grid_size, grid_size, features.shape[1])
        bev = bev.permute(0, 3, 1, 2).contiguous()
        count = count.view(batch_size, grid_size, grid_size, 1).permute(0, 3, 1, 2).contiguous()
        return bev, count
