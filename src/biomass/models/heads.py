"""Regression heads for biomass prediction.

After the backbone produces a feature vector, the head maps it to a single
scalar AGBD value (in Mg/ha). For single-encoder variants the input is 256-dim;
for late fusion (which concatenates two encoders' features) it's 512-dim.

The head is intentionally simple: a 2-layer MLP with dropout. The published
GEDI biomass papers (Lang 2023, Sialelli 2024) all use comparable head sizes.
The contribution of this paper is the fusion comparison, not the head design.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Two-layer MLP regression head.

    Args:
        in_features: dimension of input feature vector (256 for single encoder,
                     512 for late fusion).
        hidden_dim: width of the hidden layer (default 64).
        dropout: dropout probability before the final linear (default 0.2).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, in_features) -> (B,) AGBD predictions in Mg/ha.

        Output is unconstrained (no sigmoid / softplus). We rely on the
        Huber loss + augmented data distribution to keep predictions in range.
        Negative predictions are clipped at evaluation time only.
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)            # (B, 1)
        return x.squeeze(-1)       # (B,)