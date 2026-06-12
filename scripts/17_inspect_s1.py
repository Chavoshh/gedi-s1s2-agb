"""Visual sanity check for an S1 annual composite.

Plots VV (dB), VH (dB), LIA (degrees), and VV-VH ratio. Logs per-band
statistics. Mirrors scripts/10_inspect_composite.py but for SAR.

Usage:
    uv run python scripts/17_inspect_s1.py year=2020
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR, PROCESSED_DIR
from biomass.data.aoi import get_aoi
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

# Band order matches script 16's output
BAND_NAMES = ["VV_dB", "VH_dB", "LIA_deg"]


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    year = int(cfg.get("year", 2020))
    aoi = get_aoi(cfg.aoi.name)

    setup_logging(cfg.log_level)

    composite_path = PROCESSED_DIR / f"s1_composite_{aoi.name}_{year}.tif"
    out_dir = INTERIM_DIR / "composite_previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {composite_path}...")
    da = rioxarray.open_rasterio(composite_path)
    log.info(f"Shape: {da.shape}")
    log.info(f"CRS: {da.rio.crs}")

    # Downsample for plotting (full resolution would be huge)
    factor = 10
    arr = da[:, ::factor, ::factor].values.astype(np.float32)
    log.info(f"Downsampled to {arr.shape} for visualization")

    # Bands
    vv  = arr[0]
    vh  = arr[1]
    lia = arr[2]
    ratio = vv - vh  # dB ratio, biomass-sensitive

    # Per-band stats (over valid pixels)
    log.info("\nPer-band statistics:")
    for name, b in [("VV_dB", vv), ("VH_dB", vh), ("LIA_deg", lia),
                    ("VV-VH (dB)", ratio)]:
        v = b[~np.isnan(b)]
        if len(v) == 0:
            log.warning(f"  {name}: all NaN")
            continue
        log.info(
            f"  {name}: min={v.min():.2f}, max={v.max():.2f}, "
            f"mean={v.mean():.2f}, std={v.std():.2f}, "
            f"valid={len(v):,} ({len(v)/b.size*100:.1f}%)"
        )

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(22, 6.5))

    im0 = axes[0].imshow(vv, cmap="gray", vmin=-25, vmax=0)
    axes[0].set_title("VV (dB)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(vh, cmap="gray", vmin=-30, vmax=-5)
    axes[1].set_title("VH (dB)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(lia, cmap="viridis", vmin=20, vmax=60)
    axes[2].set_title("Local Incidence Angle (°)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(ratio, cmap="RdYlGn", vmin=0, vmax=12)
    axes[3].set_title("VV − VH (dB)\n(biomass-sensitive ratio)")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"S1 annual composite — {aoi.name} AOI, {year}",
                 fontsize=14, y=1.02)
    out_path = out_dir / f"s1_preview_{aoi.name}_{year}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    log.info(f"\nPreview saved to {out_path}")


if __name__ == "__main__":
    main()