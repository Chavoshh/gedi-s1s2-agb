"""Loss functions for biomass regression.

Currently provides Huber loss (smooth L1) with configurable delta. Wrapped
in a simple class for consistency with the rest of the training stack.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """Huber loss with configurable delta.

    Quadratic for |error| < delta, linear beyond. The transition point
    matters for biomass regression because the AGBD distribution is
    right-skewed: low-biomass shots are common, high-biomass shots are rare
    but contain real signal that we don't want the loss to discount entirely.

    Delta is in the same units as the labels (Mg/ha).

    Args:
        delta: transition point between quadratic and linear regimes.
               Default 30 Mg/ha is roughly the std of the training labels,
               a sensible starting point.
        reduction: 'mean' (default), 'sum', or 'none'.
    """

    def __init__(self, delta: float = 30.0, reduction: str = "mean"):
        super().__init__()
        self.delta = float(delta)
        self.reduction = reduction
        self._loss = nn.HuberLoss(delta=self.delta, reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred, target)

    def extra_repr(self) -> str:
        return f"delta={self.delta}, reduction={self.reduction!r}"