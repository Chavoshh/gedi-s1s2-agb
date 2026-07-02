"""Phase 5 step 4: publication figures for the wall-to-wall mapping results.

Produces four figures in data/processed/figures/:
  1. biomass_map_ensemble.png     - published late-fusion ensemble mean map
  2. uncertainty_map.png          - per-pixel ensemble std (epistemic uncertainty)
  3. variant_maps_comparison.png  - S1-only / S2-only / late fusion side by side
  4. cci_agreement.png            - our map vs CCI hexbin scatter over forest pixels

All map figures crop to the valid-data bounding box and mask nodata so there are no
edge gaps or stray fill pixels in the rendered output.

Usage:
    uv run python scripts/31_make_phase5_figures.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm
from omegaconf import DictConfig
from rasterio.enums import Resampling

from biomass.config import PROCESSED_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

MAPS_DIR = PROCESSED_DIR / "maps"
COMP_DIR = PROCESSED_DIR / "comparison"
FIG_DIR = PROCESSED_DIR / "figures"
NODATA = -9999.0


def _valid_mask(data: np.ndarray, nod, treat_zero_nodata: bool = False,
                clip_max: float | None = None) -> np.ndarray:
    """True where pixel is valid data (finite, not nodata, optional zero/clip masking)."""
    mask = np.isfinite(data) & (data != nod)
    if treat_zero_nodata:
        mask &= (data > 0)
    if clip_max is not None:
        # Mask the exact clip-ceiling pixels (rendering artifacts at boundaries)
        mask &= (data < clip_max)
    return mask


def _crop_to_valid(data: np.ndarray, valid: np.ndarray, pad: int = 2):
    """Crop arrays to the bounding box of valid pixels (removes nodata edge strips)."""
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0 = max(r0 - pad, 0); c0 = max(c0 - pad, 0)
    r1 = min(r1 + pad + 1, data.shape[0]); c1 = min(c1 + pad + 1, data.shape[1])
    return data[r0:r1, c0:c1], valid[r0:r1, c0:c1]


def load_map(path: Path, treat_zero_nodata: bool = False,
             clip_max: float | None = None, factor: int = 1) -> np.ma.MaskedArray:
    """Load a map (optionally downsampled), crop to valid bbox, return masked array."""
    with rasterio.open(path) as src:
        nod = src.nodata if src.nodata is not None else NODATA
        if factor > 1:
            out_h, out_w = src.height // factor, src.width // factor
            data = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.average)
        else:
            data = src.read(1)
    if data.ndim == 3:
        data = data[0]
    valid = _valid_mask(data, nod, treat_zero_nodata=treat_zero_nodata, clip_max=clip_max)
    data, valid = _crop_to_valid(data, valid)
    return np.ma.masked_array(data, mask=~valid)


def _style_map_axis(ax):
    ax.set_facecolor("white")   # nodata (masked) renders as white, clean
    ax.axis("off")


def _render_map(m, cmap_name, vmin, vmax, title, cbar_label, out_name,
                figsize=(9, 10)):
    """Render a single map with neutral background, margin, and clean frame."""
    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad("#f0f0f0")  # light gray for nodata -> reads as background

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f0f0f0")
    im = ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=13)

    # Add a margin so the footprint doesn't touch the frame
    h, w = m.shape
    margin_x = w * 0.03
    margin_y = h * 0.03
    ax.set_xlim(-margin_x, w + margin_x)
    ax.set_ylim(h + margin_y, -margin_y)  # inverted y for image coords

    # Clean frame: thin border, no ticks
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.8); spine.set_color("#888888")

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, label=cbar_label)
    cbar.outline.set_linewidth(0.5)
    fig.tight_layout()
    out = FIG_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Wrote {out.name}")


def fig_biomass_map():
    m = load_map(MAPS_DIR / "biomass_late_fusion_mean.tif")
    _render_map(m, "viridis", 0, 200,
                "Above-ground biomass, NW Iberia (late-fusion ensemble, 2021)",
                "AGBD (Mg/ha)", "biomass_map_ensemble.png")


def fig_uncertainty_map():
    m = load_map(MAPS_DIR / "biomass_late_fusion_std.tif")
    _render_map(m, "magma", 0, 30,
                "Per-pixel uncertainty (std across 3 seeds)",
                "Ensemble std (Mg/ha)", "uncertainty_map.png")


def fig_variant_comparison():
    variants = [
        ("biomass_s1_only_seed7.tif", "S1-only"),
        ("biomass_s2_only_seed7.tif", "S2-only"),
        ("biomass_late_fusion_mean.tif", "Late fusion (ensemble)"),
    ]
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("white")
    for ax, (fname, label) in zip(axes, variants):
        m = load_map(MAPS_DIR / fname, factor=3)
        ax.set_facecolor("#f0f0f0")
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=200, interpolation="nearest")
        ax.set_title(label, fontsize=12)
        h, w = m.shape
        ax.set_xlim(-w*0.03, w*1.03); ax.set_ylim(h*1.03, -h*0.03)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.8); spine.set_color("#888888")
        plt.colorbar(im, ax=ax, shrink=0.6, label="AGBD (Mg/ha)")
    fig.suptitle("Wall-to-wall biomass by model variant (2021)", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "variant_maps_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Wrote {out.name}")


def fig_cci_agreement():
    # Load without cropping alignment concerns: both are on the same grid already
    with rasterio.open(MAPS_DIR / "biomass_late_fusion_mean.tif") as s:
        our = s.read(1); our_nod = s.nodata if s.nodata is not None else NODATA
    with rasterio.open(COMP_DIR / "cci_on_grid.tif") as s:
        cci = s.read(1); cci_nod = s.nodata if s.nodata is not None else NODATA

    both = _valid_mask(our, our_nod, clip_max=499.0) & _valid_mask(cci, cci_nod, treat_zero_nodata=True)
    x = cci[both]; y = our[both]

    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(x, y, gridsize=60, extent=(0, 300, 0, 300),
                   cmap="viridis", mincnt=1, norm=LogNorm())
    ax.plot([0, 300], [0, 300], "k--", lw=1, alpha=0.6, label="1:1")
    coef = np.polyfit(x, y, 1)
    xs = np.array([0, 300])
    ax.plot(xs, coef[0] * xs + coef[1], "r-", lw=1.5,
            label=f"fit: y = {coef[0]:.2f}x + {coef[1]:.1f}")
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    bias = float(np.mean(y - x))
    corr = float(np.corrcoef(x, y)[0, 1])
    ax.text(0.04, 0.96,
            f"n = {both.sum():,}\nRMSE = {rmse:.1f} Mg/ha\nbias = {bias:+.1f}\nr = {corr:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    ax.set_xlabel("ESA CCI Biomass v5 (Mg/ha)")
    ax.set_ylabel("This study, late-fusion ensemble (Mg/ha)")
    ax.set_title("Agreement with ESA CCI Biomass over forest pixels", fontsize=13)
    ax.set_xlim(0, 300); ax.set_ylim(0, 300); ax.set_aspect("equal")
    ax.legend(loc="lower right", framealpha=0.9)
    plt.colorbar(hb, ax=ax, shrink=0.7, label="Pixel count (log)")
    fig.tight_layout()
    out = FIG_DIR / "cci_agreement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  Wrote {out.name}")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("Phase 5 step 4: generating publication figures (edge-cropped, nodata-masked)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("  Figure 1: biomass map")
    fig_biomass_map()
    log.info("  Figure 2: uncertainty map")
    fig_uncertainty_map()
    log.info("  Figure 3: variant comparison")
    fig_variant_comparison()
    log.info("  Figure 4: CCI agreement scatter")
    fig_cci_agreement()
    log.info("")
    log.info("All Phase 5 figures regenerated.")


if __name__ == "__main__":
    main()