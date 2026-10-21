import torch


class ChelsaTransform:
    MEAN = torch.tensor([
        6.44953437976,
        8.64090258720,
        31.9952721780,
        667.614613574,
        19.9400787938,
        -7.03825215068,
        26.9783309285,
        13.1069484184,
        -0.181733523402,
        15.0242478509,
        -2.16690544512,
        1547.99570201,
        171.625787913,
        89.2243911954,
        20.5307664711,
        488.970666120,
        295.627220490,
        476.427686221,
        309.930945708,
    ], dtype=torch.float32)

    STD = torch.tensor([
        3.05190351889,
        0.841170403171,
        1.55512575411,
        47.2846997194,
        2.64640342596,
        4.06833351897,
        2.17070382999,
        4.34921547104,
        3.84194112902,
        2.80011631226,
        3.51497013587,
        398.223542690,
        51.6269777885,
        24.6208640653,
        7.48821105998,
        144.133118802,
        76.1682558137,
        142.195915221,
        78.1656777126,
    ], dtype=torch.float32)

    # Terrain features derived from the DEM, appended after the 19 bioclim vars
    # when add_terrain=True. Order: [elevation(m), slope(deg), sin(aspect), cos(aspect)].
    # elevation/slope stats computed over the 2778 training sites; sin/cos are
    # already in [-1,1] with mean~0 so they pass through unchanged (mean 0, std 1)
    # to preserve the circular aspect encoding.
    TERRAIN_MEAN = torch.tensor([1062.4243, 22.7858, 0.0, 0.0], dtype=torch.float32)
    TERRAIN_STD = torch.tensor([472.8872, 12.8445, 1.0, 1.0], dtype=torch.float32)

    def __init__(self, add_terrain=False, eps=1e-6, mean=None, std=None):
        self.add_terrain = bool(add_terrain)
        self.eps = float(eps)
        climate_mean = self.MEAN if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        climate_std = self.STD if std is None else torch.as_tensor(std, dtype=torch.float32)
        if climate_mean.shape != (19,) or climate_std.shape != (19,):
            raise ValueError(
                "CHELSA normalization mean/std must each contain exactly 19 values."
            )
        if self.add_terrain:
            self.mean = torch.cat([climate_mean, self.TERRAIN_MEAN])
            self.std = torch.cat([climate_std, self.TERRAIN_STD]).clamp_min(self.eps)
        else:
            self.mean = climate_mean
            self.std = climate_std.clamp_min(self.eps)

    @staticmethod
    def _terrain_features(sample):
        """Compute [elevation(center), slope(deg), sin(aspect), cos(aspect)] from
        the site DEM via a least-squares plane fit over the whole tile."""
        if "dem" not in sample or sample["dem"] is None or "data" not in sample["dem"]:
            raise KeyError(
                "ChelsaTransform(add_terrain=True) requires sample['dem']['data']; "
                "enable modalities.dem in the dataset config."
            )
        dem = sample["dem"]["data"].float()
        if dem.ndim == 3:
            dem = dem[0]
        h, w = dem.shape

        transform = sample["dem"].get("transform", None)
        px = abs(float(transform[0])) if transform is not None else 0.5

        elev = dem[h // 2, w // 2]

        ys, xs = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        X = (xs * px).reshape(-1)
        Y = (-ys * px).reshape(-1)  # image rows go down -> flip so north is +
        Z = dem.reshape(-1)
        ok = torch.isfinite(Z)
        if ok.sum() < 10:
            # degenerate DEM -> flat, north-facing placeholder
            return torch.tensor([float(elev), 0.0, 0.0, 1.0], dtype=torch.float32)
        A = torch.stack([X[ok], Y[ok], torch.ones(int(ok.sum()))], dim=1)
        coef = torch.linalg.lstsq(A, Z[ok].unsqueeze(-1)).solution.squeeze(-1)
        a, b = coef[0], coef[1]  # dz/dx, dz/dy
        g = torch.hypot(a, b)
        slope_deg = torch.rad2deg(torch.atan(g))
        aspect = torch.atan2(b, a)
        return torch.stack([
            elev.float(),
            slope_deg.float(),
            torch.sin(aspect).float(),
            torch.cos(aspect).float(),
        ])

    def __call__(self, sample):
        if "chelsa" not in sample or sample["chelsa"] is None:
            return sample

        values = sample["chelsa"]["values"].float()

        if self.add_terrain:
            terrain = self._terrain_features(sample)  # (4,)
            values = torch.cat([values, terrain])

        if values.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"Expected CHELSA feature dimension {self.mean.shape[0]}, "
                f"got shape {tuple(values.shape)}."
            )

        sample["chelsa"]["values"] = (values - self.mean) / self.std
        sample["chelsa"]["normalized"] = True

        return sample
