import torch


class SiteLocationTransform:
    def __init__(self, src_crs="EPSG:2056", dst_crs="EPSG:4326"):
        self.src_crs = src_crs
        self.dst_crs = dst_crs
        self.transformer = None

    def __call__(self, sample):
        metadata = sample.get("metadata", {})
        bounds = metadata.get("bounds_epsg2056")

        if bounds is None:
            return sample

        bounds = bounds.float()
        left, bottom, right, top = bounds
        x_center = ((left + right) * 0.5).item()
        y_center = ((bottom + top) * 0.5).item()

        transformer = self._get_transformer()
        lon, lat = transformer.transform(x_center, y_center)
        sample["location_wgs84"] = torch.tensor([lon, lat], dtype=torch.float32)

        return sample

    def _get_transformer(self):
        if self.transformer is None:
            from pyproj import Transformer

            self.transformer = Transformer.from_crs(
                self.src_crs,
                self.dst_crs,
                always_xy=True,
            )

        return self.transformer
