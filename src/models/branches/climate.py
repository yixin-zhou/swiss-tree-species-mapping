import torch
from torch import nn


def _as_float_tensor(values, expected_dim, name):
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() != int(expected_dim):
        raise ValueError(f"{name} must have {expected_dim} values, got {tensor.numel()}.")
    return tensor.reshape(int(expected_dim))


class _TwoLayerProjector(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        in_dim = int(in_dim)
        hidden_dim = int(hidden_dim)
        output_dim = int(output_dim)
        if in_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("in_dim, hidden_dim, and output_dim must be positive.")

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x):
        return self.net(x.float())


class ChelsaMLPEncoder(nn.Module):
    def __init__(
        self,
        in_dim=19,
        hidden_dim=256,
        output_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        in_dim = int(in_dim)
        hidden_dim = int(hidden_dim)
        output_dim = int(output_dim)
        if in_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("in_dim, hidden_dim, and output_dim must be positive.")

        self.encoder = _TwoLayerProjector(in_dim, hidden_dim, output_dim, dropout)
        self.output_dim = output_dim

    def forward(self, chelsa):
        if isinstance(chelsa, dict):
            chelsa = chelsa["values"]

        return self.encoder(chelsa.float())


CHELSA_BIOCLIM_GROUPS = {
    "temperature": list(range(0, 11)),    # bio1-bio11 (0-indexed 0-10)
    "precipitation": list(range(11, 19)),  # bio12-bio19 (0-indexed 11-18)
}

# Group presets over the CHELSA(+terrain) feature vector. Indices 0-10 =
# temperature (bio1-11), 11-18 = precipitation (bio12-19), 19-22 = terrain
# (elevation, slope, sin/cos aspect) when ChelsaTransform(add_terrain=True).
CHELSA_GROUP_PRESETS = {
    # 19-dim, no terrain (original chelsa_grouped experiment)
    "temp_precip": {
        "temperature": list(range(0, 11)),
        "precipitation": list(range(11, 19)),
    },
    # 23-dim, 2 tokens: terrain folded into temperature (elevation drives the
    # temperature lapse rate, so terrain is most correlated with the temp block)
    "tempterrain_precip": {
        "temperature_terrain": list(range(0, 11)) + list(range(19, 23)),
        "precipitation": list(range(11, 19)),
    },
    # 23-dim, 3 tokens: temperature | terrain | precipitation
    "temp_terrain_precip": {
        "temperature": list(range(0, 11)),
        "terrain": list(range(19, 23)),
        "precipitation": list(range(11, 19)),
    },
}


class ChelsaGroupedMLPEncoder(nn.Module):
    """Splits the 19 CHELSA bioclim variables into semantic groups (temperature vs
    precipitation) instead of collapsing all 19 into one token via a single MLP.

    Each group gets its own encoder -- unlike the AEF spatial-patch tokens (which
    share one encoder since every position carries the same kind of signal),
    temperature and precipitation are physically different quantities, so their
    encoders should NOT share weights -- plus a learned group-identity embedding
    so the fusion layer's cross-attention can tell the two tokens apart.
    """

    def __init__(self, groups=None, hidden_dim=256, output_dim=256, dropout=0.1):
        super().__init__()
        self.groups = {k: list(v) for k, v in (groups or CHELSA_BIOCLIM_GROUPS).items()}
        self.group_names = list(self.groups.keys())
        if len(self.group_names) == 0:
            raise ValueError("ChelsaGroupedMLPEncoder requires at least one group.")

        self.encoders = nn.ModuleDict({
            name: _TwoLayerProjector(len(idx), hidden_dim, output_dim, dropout)
            for name, idx in self.groups.items()
        })
        self.group_embed = nn.Parameter(torch.zeros(1, len(self.group_names), output_dim))
        nn.init.trunc_normal_(self.group_embed, std=0.02)
        self.output_dim = output_dim

    def forward(self, chelsa):
        if isinstance(chelsa, dict):
            chelsa = chelsa["values"]
        chelsa = chelsa.float()  # (B, 19)

        tokens = torch.stack(
            [self.encoders[name](chelsa[:, idx]) for name, idx in self.groups.items()],
            dim=1,
        )  # (B, num_groups, output_dim)
        return tokens + self.group_embed


class LocationMLPEncoder(nn.Module):
    def __init__(
        self,
        in_dim=2,
        hidden_dim=256,
        output_dim=256,
        dropout=0.1,
        mean=(8.180277827, 46.85262284),
        std=(0.9639575219, 0.4270623019),
        eps=1e-6,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.register_buffer("mean", _as_float_tensor(mean, self.in_dim, "location mean"))
        self.register_buffer(
            "std",
            _as_float_tensor(std, self.in_dim, "location std").clamp_min(float(eps)),
        )
        self.encoder = _TwoLayerProjector(self.in_dim, hidden_dim, output_dim, dropout)
        self.output_dim = self.encoder.output_dim

    def forward(self, location_wgs84):
        location_wgs84 = location_wgs84.float()
        return self.encoder((location_wgs84 - self.mean) / self.std)


class ClimplicitEncoder(nn.Module):
    VALID_TEMPORAL_MODES = ["without_month", "four_month_concat"]

    def __init__(
        self,
        model_name="Jobedo/climplicit",
        temporal_mode="without_month",
        months=(3, 6, 9, 12),
        freeze=True,
        return_chelsa=False,
        raw_dim=1024,
        embed_dim=256,
        projection_hidden_dim=256,
        dropout=0.1,
        project_to_embed_dim=True,
    ):
        super().__init__()
        if temporal_mode not in self.VALID_TEMPORAL_MODES:
            raise ValueError(
                f"Unknown temporal_mode='{temporal_mode}'. "
                f"Expected one of {self.VALID_TEMPORAL_MODES}."
            )

        try:
            from rshf.climplicit import Climplicit
        except ImportError as exc:
            raise ImportError(
                "ClimplicitEncoder requires the 'rshf' package. "
                "Install the official climplicit dependencies before using "
                "model.climate.source='climplicit'."
            ) from exc

        self.temporal_mode = temporal_mode
        self.months = tuple(int(month) for month in months)
        self.freeze = bool(freeze)
        self.raw_dim = int(raw_dim)
        self.embed_dim = int(embed_dim)
        self.project_to_embed_dim = bool(project_to_embed_dim)
        self.projection_hidden_dim = int(projection_hidden_dim)

        self.clim_model = Climplicit.from_pretrained(
            model_name,
            config={"return_chelsa": bool(return_chelsa)},
        )

        if self.freeze:
            for param in self.clim_model.parameters():
                param.requires_grad = False
            self.clim_model.eval()

        if self.project_to_embed_dim:
            if self.raw_dim == self.embed_dim:
                self.out_proj = nn.Identity()
            else:
                self.out_proj = _TwoLayerProjector(
                    self.raw_dim,
                    self.projection_hidden_dim,
                    self.embed_dim,
                    dropout=dropout,
                )
            self.output_dim = self.embed_dim
        else:
            self.out_proj = nn.Identity()
            self.output_dim = self.raw_dim

    def forward(self, location_wgs84):
        location_wgs84 = location_wgs84.float()

        if self.freeze:
            with torch.no_grad():
                raw = self._forward_raw(location_wgs84)
        else:
            raw = self._forward_raw(location_wgs84)

        return self.out_proj(raw)

    def _forward_raw(self, location_wgs84):
        if self.temporal_mode == "without_month":
            return self.clim_model(location_wgs84)

        month_embeddings = []
        batch_size = location_wgs84.shape[0]
        for month in self.months:
            month_tensor = torch.full(
                (batch_size,),
                float(month),
                dtype=torch.float32,
                device=location_wgs84.device,
            )
            month_embeddings.append(self.clim_model(location_wgs84, month_tensor))

        return torch.cat(month_embeddings, dim=-1)


class ClimateBranch(nn.Module):
    VALID_SOURCES = ["chelsa_mlp", "climplicit", "location_mlp"]

    def __init__(
        self,
        source="chelsa_mlp",
        embed_dim=256,
        chelsa_cfg=None,
        climplicit_cfg=None,
        location_cfg=None,
    ):
        super().__init__()
        if source not in self.VALID_SOURCES:
            raise ValueError(
                f"Unknown climate source='{source}'. "
                f"Expected one of {self.VALID_SOURCES}."
            )

        self.source = source
        self.embed_dim = int(embed_dim)
        chelsa_cfg = chelsa_cfg or {}
        climplicit_cfg = climplicit_cfg or {}
        location_cfg = location_cfg or {}

        if self.source == "chelsa_mlp":
            if bool(chelsa_cfg.get("grouped", False)):
                preset = chelsa_cfg.get("groups_preset", None)
                groups = chelsa_cfg.get("groups", None)
                if groups is None and preset is not None:
                    if preset not in CHELSA_GROUP_PRESETS:
                        raise ValueError(
                            f"Unknown chelsa.groups_preset='{preset}'. "
                            f"Expected one of {sorted(CHELSA_GROUP_PRESETS)}."
                        )
                    groups = CHELSA_GROUP_PRESETS[preset]
                self.encoder = ChelsaGroupedMLPEncoder(
                    groups=groups,
                    hidden_dim=chelsa_cfg.get("hidden_dim", self.embed_dim),
                    output_dim=chelsa_cfg.get("output_dim", self.embed_dim),
                    dropout=chelsa_cfg.get("dropout", 0.1),
                )
            else:
                self.encoder = ChelsaMLPEncoder(
                    in_dim=chelsa_cfg.get("in_dim", 19),
                    hidden_dim=chelsa_cfg.get("hidden_dim", self.embed_dim),
                    output_dim=chelsa_cfg.get("output_dim", self.embed_dim),
                    dropout=chelsa_cfg.get("dropout", 0.1),
                )
        elif self.source == "climplicit":
            self.encoder = ClimplicitEncoder(
                model_name=climplicit_cfg.get("model_name", "Jobedo/climplicit"),
                temporal_mode=climplicit_cfg.get("temporal_mode", "without_month"),
                months=climplicit_cfg.get("months", [3, 6, 9, 12]),
                freeze=climplicit_cfg.get("freeze", True),
                return_chelsa=climplicit_cfg.get("return_chelsa", False),
                raw_dim=climplicit_cfg.get("raw_dim", 1024),
                embed_dim=climplicit_cfg.get("embed_dim", self.embed_dim),
                projection_hidden_dim=climplicit_cfg.get("projection_hidden_dim", 256),
                dropout=climplicit_cfg.get("dropout", 0.1),
                project_to_embed_dim=climplicit_cfg.get("project_to_embed_dim", True),
            )
        else:
            self.encoder = LocationMLPEncoder(
                in_dim=location_cfg.get("in_dim", 2),
                hidden_dim=location_cfg.get("hidden_dim", 256),
                output_dim=location_cfg.get("output_dim", self.embed_dim),
                dropout=location_cfg.get("dropout", 0.1),
                mean=location_cfg.get("mean", [8.180277827, 46.85262284]),
                std=location_cfg.get("std", [0.9639575219, 0.4270623019]),
            )

        self.output_dim = self.encoder.output_dim

    def forward(self, batch):
        if self.source == "chelsa_mlp":
            if "chelsa" not in batch or batch["chelsa"] is None:
                raise KeyError("ClimateBranch source='chelsa_mlp' requires batch['chelsa'].")
            climate_token = self.encoder(batch["chelsa"])
        elif self.source in ("climplicit", "location_mlp"):
            if "location_wgs84" not in batch:
                raise KeyError(
                    f"ClimateBranch source='{self.source}' requires "
                    "batch['location_wgs84']."
                )
            climate_token = self.encoder(batch["location_wgs84"])
        else:
            raise RuntimeError(f"Unhandled climate source='{self.source}'.")

        return {"climate_token": climate_token}
