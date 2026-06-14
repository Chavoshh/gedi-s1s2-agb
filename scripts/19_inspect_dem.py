"""Visual sanity check for the Copernicus DEM composite.

Plots elevation (m) and slope (degrees) side by side, logs per-band stats.
Mirrors scripts/10_inspect_composite.py and scripts/17_inspect_s1.py.

Usage:
    uv run python scripts/19_inspect_dem.py
"""
from __future__ import annotations

import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR, PROCESSED_DIR
from biomass.data.aoi import get_aoi
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    aoi = get_aoi(cfg.aoi.name)
    setup_logging(cfg.log_level)

    dem_path = PROCESSED_DIR / f"dem_{aoi.name}.tif"
    out_dir = INTERIM_DIR / "composite_previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {dem_path}...")
    da = rioxarray.open_rasterio(dem_path)
    log.info(f"Shape: {da.shape}, CRS: {da.rio.crs}")

    # Downsample for plotting
    factor = 10
    arr = da[:, ::factor, ::factor].values.astype(np.float32)
    log.info(f"Downsampled to {arr.shape} for visualization")

    elev = arr[0]
    slope = arr[1]

    # Per-band stats over valid pixels
    log.info("\nPer-band statistics (over valid pixels):")
    for name, b, unit in [("elevation", elev, "m"), ("slope", slope, "deg")]:
        v = b[~np.isnan(b)]
        if len(v) == 0:
            log.warning(f"  {name}: all NaN")
            continue
        log.info(
            f"  {name} ({unit}): min={v.min():.2f}, max={v.max():.2f}, "
            f"mean={v.mean():.2f}, std={v.std():.2f}, "
            f"valid={len(v):,} ({len(v)/b.size*100:.1f}%)"
        )

    # Plot — terrain colormap for elevation, plasma for slope
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    im0 = axes[0].imshow(elev, cmap="terrain", vmin=0, vmax=1200)
    axes[0].set_title("Elevation (m)")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(slope, cmap="plasma", vmin=0, vmax=40)
    axes[1].set_title("Slope (degrees)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Copernicus DEM GLO-30 - {aoi.name} AOI",
                 fontsize=14, y=1.02)
    out_path = out_dir / f"dem_preview_{aoi.name}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    log.info(f"\nPreview saved to {out_path}")


if __name__ == "__main__":
    main()