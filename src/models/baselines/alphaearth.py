"""AlphaEarthCompositionMLP baseline architecture."""

import torch
from torch import nn

from src.data.label_schema import SPECIES_NUM_CLASSES


class AlphaEarthCompositionMLP(nn.Module):
    """Eight paper-style Linear/BatchNorm/GELU/Dropout blocks and two heads."""

    def __init__(self, input_dim=64, hidden_dim=128, hidden_layers=8, dropout=0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.extend((
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.species_head = nn.Linear(hidden_dim, SPECIES_NUM_CLASSES)
        self.leaf_head = nn.Linear(hidden_dim, 2)

    def forward(self, embedding):
        if embedding.ndim != 2 or embedding.shape[-1] != 64:
            raise ValueError(f"Expected (B,64) AEF input, got {tuple(embedding.shape)}.")
        features = self.trunk(torch.nan_to_num(embedding.float()))
        species_logits = self.species_head(features)
        leaf_logits = self.leaf_head(features)
        return {
            "species_logits": species_logits,
            "species_prob": torch.softmax(species_logits, dim=-1),
            "leaf_logits": leaf_logits,
            "leaf_prob": torch.softmax(leaf_logits, dim=-1),
        }

