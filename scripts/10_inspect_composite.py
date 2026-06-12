"""Quick visual sanity check on a built composite.

Plots a true-color preview, false-color (NIR-R-G) preview, and NDVI.
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

# Band index in the multi-band TIFF (1-indexed for rioxarray .sel, 0-indexed for .isel)
# Order matches S2_BANDS in script 09: B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12
BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def stretch(arr: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Percentile stretch for visualization, ignoring NaN."""
    lo, hi = np.nanpercentile(arr, [p_low, p_high])
    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    year = int(cfg.get("year", 2020))
    aoi = get_aoi(cfg.aoi.name)

    setup_logging(cfg.log_level)

    composite_path = PROCESSED_DIR / f"s2_composite_{aoi.name}_{year}.tif"
    out_dir = INTERIM_DIR / "composite_previews"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {composite_path}...")
    da = rioxarray.open_rasterio(composite_path)
    log.info(f"Shape: {da.shape}")

    # Downsample for plotting (full resolution would be unwieldy)
    factor = 10
    da_small = da[:, ::factor, ::factor].values.astype(np.float32)
    log.info(f"Downsampled to {da_small.shape} for visualization")

    # Per-band valid-pixel counts
    log.info("\nValid pixel counts per band (after cloud masking):")
    for i, name in enumerate(BAND_NAMES):
        b = da_small[i]
        n_valid = (~np.isnan(b) & (b > 0)).sum()
        total = b.size
        log.info(f"  {name}: {n_valid:>8d} / {total} ({n_valid/total*100:.1f}%)")

    # Extract bands by name (index lookup)
    def band(name): return da_small[BAND_NAMES.index(name)]
    b02, b03, b04 = band("B02"), band("B03"), band("B04")
    b08 = band("B08")

    # True color (B04 R, B03 G, B02 B)
    rgb_true = np.stack([stretch(b04), stretch(b03), stretch(b02)], axis=-1)

    # False color (B08 R, B04 G, B03 B)
    rgb_false = np.stack([stretch(b08), stretch(b04), stretch(b03)], axis=-1)

    # NDVI
    ndvi = (b08 - b04) / (b08 + b04 + 1e-9)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    axes[0].imshow(rgb_true)
    axes[0].set_title(f"True color (RGB = B04, B03, B02)")
    axes[0].axis("off")

    axes[1].imshow(rgb_false)
    axes[1].set_title(f"False color (RGB = B08, B04, B03)")
    axes[1].axis("off")

    im = axes[2].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    axes[2].set_title("NDVI")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"S2 annual composite - {aoi.name} AOI, {year}",
                 fontsize=13, y=1.02)
    out_path = out_dir / f"preview_{aoi.name}_{year}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    log.info(f"\nPreview saved to {out_path}")

    # NDVI stats over valid pixels
    valid = ~np.isnan(ndvi)
    log.info(f"\nNDVI summary (over {valid.sum()} valid pixels):")
    log.info(f"  min:    {np.nanmin(ndvi):.3f}")
    log.info(f"  max:    {np.nanmax(ndvi):.3f}")
    log.info(f"  mean:   {np.nanmean(ndvi):.3f}")
    log.info(f"  median: {np.nanmedian(ndvi):.3f}")
    log.info(f"  std:    {np.nanstd(ndvi):.3f}")


if __name__ == "__main__":
    main()