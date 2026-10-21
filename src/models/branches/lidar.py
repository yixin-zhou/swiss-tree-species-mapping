import functools
import logging
from collections import OrderedDict

import torch
from torch import nn

# spconv triggers torch.fx's is_fx_tracing check, which logs a benign warning via
# the logging module (not warnings.warn), so we silence that logger directly.
logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)

try:
    import spconv.pytorch as spconv
    from spconv.pytorch.modules import SparseModule
except ImportError as exc:
    raise ImportError(
        "LidarSparseUNetBranch requires spconv. Install a compatible "
        "spconv-cuXXX package before using the LiDAR sparse U-Net branch."
    ) from exc


class SparseResidualBlock(SparseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
        indice_key=None,
        normalize_before=True,
    ):
        super().__init__()

        if in_channels == out_channels:
            self.identity_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.identity_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels),
                nn.ReLU(),
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                ),
                norm_fn(out_channels),
                nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                ),
            )
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                ),
                norm_fn(out_channels),
                nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                ),
                norm_fn(out_channels),
                nn.ReLU(),
            )

    def forward(self, input_tensor):
        identity = spconv.SparseConvTensor(
            input_tensor.features,
            input_tensor.indices,
            input_tensor.spatial_shape,
            input_tensor.batch_size,
        )
        output = self.conv_branch(input_tensor)
        output = output.replace_feature(
            output.features + self.identity_branch(identity).features
        )
        return output


class SparseUNet(nn.Module):
    """Sparse 3D U-Net backbone adapted from ForestFormer3D's SpConvUNet."""

    def __init__(
        self,
        num_planes,
        norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
        block_reps=2,
        block=SparseResidualBlock,
        indice_key_id=1,
        normalize_before=True,
    ):
        super().__init__()
        self.num_planes = list(num_planes)

        blocks = OrderedDict()
        for i in range(int(block_reps)):
            blocks[f"block{i}"] = block(
                self.num_planes[0],
                self.num_planes[0],
                norm_fn,
                indice_key=f"subm{indice_key_id}",
                normalize_before=normalize_before,
            )
        self.blocks = spconv.SparseSequential(blocks)

        if len(self.num_planes) > 1:
            if normalize_before:
                self.down = spconv.SparseSequential(
                    norm_fn(self.num_planes[0]),
                    nn.ReLU(),
                    spconv.SparseConv3d(
                        self.num_planes[0],
                        self.num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}",
                    ),
                )
            else:
                self.down = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        self.num_planes[0],
                        self.num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}",
                    ),
                    norm_fn(self.num_planes[1]),
                    nn.ReLU(),
                )

            self.inner_unet = SparseUNet(
                self.num_planes[1:],
                norm_fn=norm_fn,
                block_reps=block_reps,
                block=block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
            )

            if normalize_before:
                self.up = spconv.SparseSequential(
                    norm_fn(self.num_planes[1]),
                    nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        self.num_planes[1],
                        self.num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}",
                    ),
                )
            else:
                self.up = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        self.num_planes[1],
                        self.num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}",
                    ),
                    norm_fn(self.num_planes[0]),
                    nn.ReLU(),
                )

            tail_blocks = OrderedDict()
            for i in range(int(block_reps)):
                tail_blocks[f"block{i}"] = block(
                    self.num_planes[0] * (2 - i),
                    self.num_planes[0],
                    norm_fn,
                    indice_key=f"subm{indice_key_id}",
                    normalize_before=normalize_before,
                )
            self.tail_blocks = spconv.SparseSequential(tail_blocks)

    def forward(self, input_tensor):
        output = self.blocks(input_tensor)
        identity = spconv.SparseConvTensor(
            output.features,
            output.indices,
            output.spatial_shape,
            output.batch_size,
        )

        if len(self.num_planes) > 1:
            decoded = self.down(output)
            decoded = self.inner_unet(decoded)
            decoded = self.up(decoded)
            output = output.replace_feature(
                torch.cat([identity.features, decoded.features], dim=1)
            )
            output = self.tail_blocks(output)

        return output


class LidarSparseUNetBranch(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=32,
        embed_dim=256,
        voxel_size=0.2,
        crop_radius_m=14.0,
        min_spatial_shape=32,
        bev_grid_size=16,
        num_planes=None,
        block_reps=2,
        bev_pooling="mean",
        return_debug=False,
    ):
        super().__init__()
        if bev_pooling != "mean":
            raise ValueError("Only bev_pooling='mean' is implemented for now.")

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.embed_dim = int(embed_dim)
        self.voxel_size = float(voxel_size)
        self.crop_radius_m = float(crop_radius_m)
        self.min_spatial_shape = int(min_spatial_shape)
        self.bev_grid_size = int(bev_grid_size)
        self.bev_pooling = bev_pooling
        self.return_debug = bool(return_debug)

        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive.")
        if self.crop_radius_m <= 0:
            raise ValueError("crop_radius_m must be positive.")
        if self.bev_grid_size <= 0:
            raise ValueError("bev_grid_size must be positive.")

        if num_planes is None:
            num_planes = [self.base_channels * (i + 1) for i in range(5)]
        self.num_planes = [int(ch) for ch in num_planes]

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.in_channels,
                self.num_planes[0],
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        self.unet = SparseUNet(self.num_planes, block_reps=int(block_reps))
        self.output_layer = spconv.SparseSequential(
            nn.BatchNorm1d(self.num_planes[0], eps=1e-4, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(self.num_planes[0]),
            nn.Linear(self.num_planes[0], self.embed_dim),
        )

    def forward(self, batch):
        if "lidar" not in batch or batch["lidar"] is None:
            raise KeyError("LidarSparseUNetBranch requires batch['lidar'].")

        points = batch["lidar"]["points"]
        sparse_tensor, voxel_xy = self._points_to_sparse_tensor(points)

        sparse_tensor = self.input_conv(sparse_tensor)
        sparse_tensor = self.unet(sparse_tensor)
        sparse_tensor = self.output_layer(sparse_tensor)

        features = self.proj(sparse_tensor.features)
        lidar_spatial, bev_count = self._sparse_to_bev(
            features=features,
            voxel_xy=voxel_xy,
            batch_idx=sparse_tensor.indices[:, 0].long(),
            batch_size=len(points),
        )

        out = {"lidar_spatial": lidar_spatial}

        if self.return_debug:
            out.update({
                "lidar_bev_count": bev_count,
                "lidar_sparse_features": features,
                "lidar_sparse_coords": sparse_tensor.indices,
            })

        return out

    def _points_to_sparse_tensor(self, points_list):
        voxel_features = []
        voxel_coords = []
        voxel_xy = []
        max_coord = None
        device = None

        for batch_idx, points in enumerate(points_list):
            points = points[:, :3].float()
            if points.numel() == 0:
                raise ValueError(f"Empty LiDAR point tensor in batch index {batch_idx}.")

            if device is None:
                device = points.device

            voxel = torch.floor(points / self.voxel_size).long()
            voxel = voxel - voxel.min(dim=0).values

            unique_voxel, inverse, counts = torch.unique(
                voxel,
                dim=0,
                return_inverse=True,
                return_counts=True,
            )

            feature_sum = points.new_zeros((unique_voxel.shape[0], self.in_channels))
            feature_sum.index_add_(0, inverse, points[:, :self.in_channels])
            feature_mean = feature_sum / counts[:, None].to(points.dtype)

            batch_col = torch.full(
                (unique_voxel.shape[0], 1),
                batch_idx,
                dtype=torch.long,
                device=points.device,
            )
            coords = torch.cat([batch_col, unique_voxel], dim=1)

            voxel_features.append(feature_mean)
            voxel_coords.append(coords)
            voxel_xy.append(feature_mean[:, :2])

            coord_max = unique_voxel.max(dim=0).values
            max_coord = coord_max if max_coord is None else torch.maximum(max_coord, coord_max)

        features = torch.cat(voxel_features, dim=0)
        coords = torch.cat(voxel_coords, dim=0).to(torch.int32)
        voxel_xy = torch.cat(voxel_xy, dim=0)

        spatial_shape = torch.clamp(
            max_coord + 1,
            min=self.min_spatial_shape,
        )
        spatial_shape = [int(v) for v in spatial_shape.tolist()]

        return (
            spconv.SparseConvTensor(
                features=features,
                indices=coords,
                spatial_shape=spatial_shape,
                batch_size=len(points_list),
            ),
            voxel_xy,
        )

    def _sparse_to_bev(self, features, voxel_xy, batch_idx, batch_size):
        grid_size = self.bev_grid_size
        radius = self.crop_radius_m

        x = voxel_xy[:, 0]
        y = voxel_xy[:, 1]

        x_bin = torch.floor((x + radius) / (2.0 * radius) * grid_size).long()
        y_bin = torch.floor((radius - y) / (2.0 * radius) * grid_size).long()
        x_bin = x_bin.clamp(0, grid_size - 1)
        y_bin = y_bin.clamp(0, grid_size - 1)

        flat_idx = batch_idx * (grid_size * grid_size) + y_bin * grid_size + x_bin

        bev = features.new_zeros((batch_size * grid_size * grid_size, features.shape[1]))
        count = features.new_zeros((batch_size * grid_size * grid_size, 1))

        bev.index_add_(0, flat_idx, features)
        count.index_add_(0, flat_idx, torch.ones_like(features[:, :1]))
        bev = bev / count.clamp_min(1.0)

        bev = bev.view(batch_size, grid_size, grid_size, features.shape[1])
        bev = bev.permute(0, 3, 1, 2).contiguous()

        count = count.view(batch_size, grid_size, grid_size, 1)
        count = count.permute(0, 3, 1, 2).contiguous()

        return bev, count
