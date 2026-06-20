"""Four model variants for the fusion comparison.

All four share the same ResNet18Adapted backbone and the same RegressionHead.
Only the input channel routing and the number of encoders differ:

  S1Only:    1 encoder takes (S1 + DEM) = 5 channels
  S2Only:    1 encoder takes (S2 + DEM) = 12 channels
  EarlyFusion: 1 encoder takes (S2 + S1 + DEM) = 15 channels
  LateFusion:  2 encoders run in parallel; concat features then MLP head
"""
from __future__ import annotations

import torch
import torch.nn as nn

from biomass.models.backbone import ResNet18Adapted, count_parameters
from biomass.models.heads import RegressionHead
from biomass.training.dataset import (
    DEM_CHANNELS, S1_CHANNELS, S2_CHANNELS, Variant,
)


class SingleEncoderModel(nn.Module):
    """A backbone + head pair for single-encoder variants (S1, S2, early)."""

    def __init__(self, in_channels: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = ResNet18Adapted(in_channels=in_channels)
        self.head = RegressionHead(
            in_features=self.encoder.feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, 25, 25)
        feats = self.encoder(x)          # (B, 256)
        pred = self.head(feats)          # (B,)
        return pred


class LateFusionModel(nn.Module):
    """Two parallel encoders (one per sensor) + concat + MLP head.

    The DEM is duplicated across both branches because (a) it's small (2 channels),
    (b) terrain context is informative for both modalities, and (c) this keeps
    each branch self-contained and the comparison to single-sensor variants clean.

    The dataset always returns the full 15-channel patch with the locked channel
    order; the model splits internally using the channel-index constants.
    """

    # Channel groups as routed to each branch. Slice positions reflect the
    # locked channel order in the dataset.
    # S1 branch input: VV_dB, VH_dB, LIA_deg, elevation_m, slope_deg
    # S2 branch input: B02..B12, B8A, elevation_m, slope_deg
    S1_BRANCH_CHANNELS = S1_CHANNELS + DEM_CHANNELS    # indices into the 15-ch patch
    S2_BRANCH_CHANNELS = S2_CHANNELS + DEM_CHANNELS

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.s1_encoder = ResNet18Adapted(in_channels=len(self.S1_BRANCH_CHANNELS))  # 5
        self.s2_encoder = ResNet18Adapted(in_channels=len(self.S2_BRANCH_CHANNELS))  # 12

        # Register the channel indices as buffers so they move with .cuda()
        self.register_buffer(
            "_s1_idx", torch.tensor(self.S1_BRANCH_CHANNELS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_s2_idx", torch.tensor(self.S2_BRANCH_CHANNELS, dtype=torch.long),
            persistent=False,
        )

        self.head = RegressionHead(
            in_features=self.s1_encoder.feature_dim + self.s2_encoder.feature_dim,  # 512
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 15, 25, 25) — full patch from the dataset
        x_s1 = x.index_select(dim=1, index=self._s1_idx)   # (B, 5, 25, 25)
        x_s2 = x.index_select(dim=1, index=self._s2_idx)   # (B, 12, 25, 25)
        f_s1 = self.s1_encoder(x_s1)                       # (B, 256)
        f_s2 = self.s2_encoder(x_s2)                       # (B, 256)
        f = torch.cat([f_s1, f_s2], dim=1)                 # (B, 512)
        pred = self.head(f)                                # (B,)
        return pred


def build_model(variant: Variant, hidden_dim: int = 64, dropout: float = 0.2) -> nn.Module:
    """Factory: construct the model for a given variant.

    The dataset variant determines how many channels are exposed; the model
    here matches that contract.

    Variant       | Dataset channels                       | Model
    S1_ONLY       | 5  (S1 + DEM)                          | SingleEncoderModel(5)
    S2_ONLY       | 12 (S2 + DEM)                          | SingleEncoderModel(12)
    EARLY         | 15 (S2 + S1 + DEM, single encoder)     | SingleEncoderModel(15)
    LATE          | 15 (S2 + S1 + DEM, split in the model) | LateFusionModel()
    """
    if variant == Variant.S1_ONLY:
        return SingleEncoderModel(in_channels=5, hidden_dim=hidden_dim, dropout=dropout)
    if variant == Variant.S2_ONLY:
        return SingleEncoderModel(in_channels=12, hidden_dim=hidden_dim, dropout=dropout)
    if variant == Variant.EARLY:
        return SingleEncoderModel(in_channels=15, hidden_dim=hidden_dim, dropout=dropout)
    if variant == Variant.LATE:
        return LateFusionModel(hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown variant: {variant}")


def summarize_model(variant: Variant, model: nn.Module) -> str:
    """Pretty summary of a model: variant, param count, and head/encoder split."""
    total = count_parameters(model)
    if isinstance(model, LateFusionModel):
        enc = count_parameters(model.s1_encoder) + count_parameters(model.s2_encoder)
    else:
        enc = count_parameters(model.encoder)
    head = count_parameters(model.head)
    return (
        f"  {variant.value:10s}  total={total/1e6:5.2f}M  "
        f"encoder(s)={enc/1e6:5.2f}M  head={head/1e3:5.1f}K"
    )