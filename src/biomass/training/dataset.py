"""Zarr-backed PyTorch Dataset for GEDI-supervised biomass regression.

Wraps the patches Zarr store produced by scripts/20_extract_patches.py.
Handles split filtering (train / val / test), variant-specific channel
selection, per-channel normalization, and label-invariant augmentation.

Usage:
    from biomass.training.dataset import BiomassPatchDataset, Variant

    ds_train = BiomassPatchDataset(
        zarr_path=PROCESSED_DIR / "patches_dev.zarr",
        meta_path=PROCESSED_DIR / "patches_dev_meta.json",
        split="train",
        variant=Variant.EARLY,
    )
    patch, label = ds_train[0]  # patch: (C, 25, 25) float32; label: scalar float32
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


# Locked channel order from patches.py / patches_dev_meta.json
# Index 0..9   -> S2 spectral bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)
# Index 10..12 -> S1 SAR (VV_dB, VH_dB, LIA_deg)
# Index 13..14 -> DEM (elevation_m, slope_deg)
S2_CHANNELS = list(range(0, 10))      # 10 bands
S1_CHANNELS = list(range(10, 13))     # 3 bands
DEM_CHANNELS = list(range(13, 15))    # 2 bands

SPLIT_TO_INT = {"train": 0, "val": 1, "test": 2}


class Variant(str, Enum):
    """Which input channels each model variant sees.

    All variants include DEM (elevation, slope) as auxiliary input.
    """
    S1_ONLY = "s1_only"     # 3 S1 + 2 DEM = 5 channels
    S2_ONLY = "s2_only"     # 10 S2 + 2 DEM = 12 channels
    EARLY = "early"         # 10 S2 + 3 S1 + 2 DEM = 15 channels (all)
    LATE = "late"           # Same as EARLY at dataset level; the model splits internally


# Map variant to channel indices into the 15-channel raw patch.
VARIANT_CHANNELS: dict[Variant, list[int]] = {
    Variant.S1_ONLY: S1_CHANNELS + DEM_CHANNELS,
    Variant.S2_ONLY: S2_CHANNELS + DEM_CHANNELS,
    Variant.EARLY: S2_CHANNELS + S1_CHANNELS + DEM_CHANNELS,
    Variant.LATE: S2_CHANNELS + S1_CHANNELS + DEM_CHANNELS,
}


@dataclass
class NormStats:
    """Per-channel mean and std, indexed by channel position in the *selected*
    channel subset for a given variant (NOT the raw 15-channel array)."""
    mean: np.ndarray  # shape (C,)
    std: np.ndarray   # shape (C,)


def load_norm_stats(meta_path: Path, channel_indices: list[int]) -> NormStats:
    """Load per-channel normalization stats for a subset of channels.

    Reads patches_dev_meta.json and selects the mean/std entries for the
    channels actually used by the variant.
    """
    with open(meta_path) as f:
        meta = json.load(f)

    channels_all = meta["channels"]                  # list of 15 channel names
    norm = meta["normalization"]                     # dict[name] -> {mean, std}

    selected_names = [channels_all[i] for i in channel_indices]
    means = np.array([norm[name]["mean"] for name in selected_names], dtype=np.float32)
    stds = np.array([norm[name]["std"] for name in selected_names], dtype=np.float32)

    # Defensive: a zero std would produce inf after division
    if np.any(stds < 1e-8):
        raise ValueError(
            f"Near-zero std for channels: "
            f"{[selected_names[i] for i, s in enumerate(stds) if s < 1e-8]}"
        )

    return NormStats(mean=means, std=stds)


class BiomassPatchDataset(Dataset):
    """PyTorch Dataset for one split of the patches Zarr store.

    Args:
        zarr_path: path to patches_dev.zarr directory
        meta_path: path to patches_dev_meta.json
        split: one of "train", "val", "test"
        variant: which channel subset to expose to the model
        augment: whether to apply label-invariant augmentations (rotations, flips).
                 Forced to False for val and test regardless of this argument.

    __getitem__ returns:
        patch: torch.Tensor of shape (C, 25, 25), float32, normalized
        label: torch.Tensor scalar, float32 (AGBD in Mg/ha, raw)
    """

    def __init__(
        self,
        zarr_path: Path | str,
        meta_path: Path | str,
        split: str,
        variant: Variant,
        augment: bool = True,
    ):
        if split not in SPLIT_TO_INT:
            raise ValueError(f"Unknown split: {split!r}. Use one of {list(SPLIT_TO_INT)}")
        self.split = split
        self.variant = Variant(variant)
        self.augment = augment and (split == "train")

        # Open Zarr store (read-only)
        self.zarr_path = Path(zarr_path)
        self.meta_path = Path(meta_path)
        self.root = zarr.open_group(str(self.zarr_path), mode="r")

        # Determine indices belonging to this split
        # splits array is shape (N,), int8: 0=train, 1=val, 2=test
        all_splits = self.root["splits"][:]
        target = SPLIT_TO_INT[split]
        self.indices = np.where(all_splits == target)[0]
        if len(self.indices) == 0:
            raise RuntimeError(
                f"No patches found for split={split!r} in {zarr_path}"
            )

        # Channel selection for this variant
        self.channel_indices = VARIANT_CHANNELS[self.variant]
        self.n_channels = len(self.channel_indices)

        # Load normalization stats for the selected channels
        self.norm = load_norm_stats(self.meta_path, self.channel_indices)
        # Reshape for broadcasting against (C, H, W)
        self._mean_chw = self.norm.mean.reshape(-1, 1, 1).astype(np.float32)
        self._std_chw = self.norm.std.reshape(-1, 1, 1).astype(np.float32)

        log.info(
            f"BiomassPatchDataset[{split}, {self.variant.value}]: "
            f"{len(self.indices):,} patches, {self.n_channels} channels, "
            f"augment={self.augment}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Translate split-local index into the underlying Zarr index
        zarr_idx = int(self.indices[i])

        # Read raw patch (15, 25, 25) and select the variant's channels
        # Zarr's vindex / oindex would be slower; selecting all 15 channels and
        # slicing is faster because chunks already contain all channels.
        raw = self.root["patches"][zarr_idx]                  # (15, 25, 25) float32
        patch = raw[self.channel_indices, :, :]               # (C, 25, 25) float32

        # Normalize: per-channel z-score using training-set stats
        patch = (patch - self._mean_chw) / self._std_chw

        # Augmentation (train-only)
        if self.augment:
            patch = self._augment(patch)

        # Label
        label = float(self.root["labels"][zarr_idx])

        return (
            torch.from_numpy(np.ascontiguousarray(patch)),
            torch.tensor(label, dtype=torch.float32),
        )

    @staticmethod
    def _augment(patch: np.ndarray) -> np.ndarray:
        """Apply random 90-degree rotation and horizontal/vertical flips.

        These are label-invariant: AGBD is rotation- and flip-invariant.
        """
        # Random 90-degree rotation: k in {0, 1, 2, 3}
        k = np.random.randint(0, 4)
        if k > 0:
            patch = np.rot90(patch, k=k, axes=(-2, -1))

        # Random horizontal flip (50%)
        if np.random.rand() < 0.5:
            patch = patch[..., ::-1]

        # Random vertical flip (50%)
        if np.random.rand() < 0.5:
            patch = patch[..., ::-1, :]

        return patch