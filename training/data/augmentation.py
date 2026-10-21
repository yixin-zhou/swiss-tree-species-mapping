import numpy as np
import torch


class GeometricAugmentation:
    """Cross-modal-consistent D4 augmentation (flips + 90-degree rotations).

    Applies the SAME random element of the dihedral group D4 to the RGB image,
    the LiDAR point cloud (XY), Sentinel-2 imagery, and the spatial AlphaEarth
    embedding patch, so all spatial modalities stay aligned. Train-only.

    Correspondence (image array dims (-2,-1) = (row=N->S, col=W->E), LiDAR x=E, y=N):
      rot90 k times (CCW):  image torch.rot90(img, k, (-2,-1))  <->  points (x,y) -> (-y, x) applied k times
      hflip (E<->W):        image torch.flip(img, [-1])          <->  points x -> -x
    """

    def __init__(self, p=1.0, rotations=True, flips=True, apply_sentinel2=True):
        self.p = float(p)
        self.rotations = bool(rotations)
        self.flips = bool(flips)
        self.apply_sentinel2 = bool(apply_sentinel2)

    def __call__(self, sample):
        if np.random.rand() >= self.p:
            return sample

        k = int(np.random.randint(0, 4)) if self.rotations else 0
        hflip = bool(np.random.rand() < 0.5) if self.flips else False
        if k == 0 and not hflip:
            return sample

        if "rgb" in sample and sample["rgb"] is not None and "data" in sample["rgb"]:
            sample["rgb"]["data"] = self._img(sample["rgb"]["data"], k, hflip)

        if self.apply_sentinel2 and "sentinel2" in sample and sample["sentinel2"] is not None:
            sentinel2 = sample["sentinel2"]
            if "data" in sentinel2:
                sentinel2["data"] = self._img(sentinel2["data"], k, hflip)
            if (
                sentinel2.get("temporal_mode") == "satellite_embedding_patch"
                and "embedding" in sentinel2
            ):
                sentinel2["embedding"] = self._embedding_patch(
                    sentinel2["embedding"], k, hflip
                )

        if "lidar" in sample and sample["lidar"] is not None:
            if "points" in sample["lidar"]:
                sample["lidar"]["points"] = self._points(
                    sample["lidar"]["points"], k, hflip
                )
            if "canopy_raster" in sample["lidar"]:
                sample["lidar"]["canopy_raster"] = self._img(
                    sample["lidar"]["canopy_raster"], k, hflip
                )

        return sample

    @staticmethod
    def _img(x, k, hflip):
        # x: (..., H, W). Apply hflip first, then rot90 k times (CCW), on the last two dims.
        if hflip:
            x = torch.flip(x, dims=[-1])
        if k:
            x = torch.rot90(x, k, dims=(-2, -1))
        return x.contiguous()

    @staticmethod
    def _points(p, k, hflip):
        # p: (N, 3) = (x, y, z). Apply the matching transform to XY only.
        p = p.clone()
        x, y = p[:, 0].clone(), p[:, 1].clone()
        if hflip:
            x = -x
        for _ in range(k):  # (x, y) -> (-y, x)
            x, y = -y, x
        p[:, 0], p[:, 1] = x, y
        return p

    @classmethod
    def _embedding_patch(cls, embedding, k, hflip):
        # embedding: (H*W, D), stored row-major in the HDF5 file.
        if embedding.ndim != 2:
            raise ValueError(
                "Expected spatial AlphaEarth embedding shape (num_tokens, dim), "
                f"got {tuple(embedding.shape)}."
            )
        num_tokens, embedding_dim = embedding.shape
        grid_size = int(np.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            raise ValueError(
                "Spatial AlphaEarth embedding must contain a square token grid, "
                f"got {num_tokens} tokens."
            )

        grid = embedding.reshape(grid_size, grid_size, embedding_dim)
        grid = grid.permute(2, 0, 1)
        grid = cls._img(grid, k, hflip)
        return grid.permute(1, 2, 0).reshape(num_tokens, embedding_dim).contiguous()
