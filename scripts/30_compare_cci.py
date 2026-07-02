"""Phase 5 step 3: compare the wall-to-wall biomass map against ESA CCI Biomass v5.

CCI Biomass v5 (2021, 100 m) maps *woody* above-ground biomass: ~25% of AOI pixels
are 0, corresponding to non-forest (grassland, cropland) with no woody biomass, not
nodata. Our GEDI-calibrated model predicts continuous biomass everywhere. A fair
comparison therefore restricts to forest pixels (CCI > 0) where both products claim
biomass. We report:
  - Comparison A (primary): metrics over forest pixels (CCI > 0 AND our map valid)
  - Comparison B (qualitative): characterization of where the products differ,
    cross-referenced with WorldCover land cover.

CCI (EPSG:4326) is reprojected onto our map's UTM 29N grid (bilinear) so pixels align.

Outputs (data/processed/comparison/):
  - cci_on_grid.tif                 CCI reprojected to our grid
  - comparison_vs_cci_metrics.csv   agreement metrics, overall + per AGBD bin
  - comparison_vs_cci_landcover.csv WorldCover breakdown of CCI-zero vs nonzero
  - (figures produced in script 32)

Usage:
    uv run python scripts/30_compare_cci.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
from omegaconf import DictConfig
from rasterio.enums import Resampling
from rasterio.warp import reproject

from biomass.config import PROCESSED_DIR, RAW_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

MAPS_DIR = PROCESSED_DIR / "maps"
OUT_DIR = PROCESSED_DIR / "comparison"
CCI_DIR = RAW_DIR / "cci_biomass"
CCI_FILE = "N50W010_ESACCI-BIOMASS-L4-AGB-MERGED-100m-2021-fv5.0.tif"

# Our published biomass map (late-fusion ensemble mean)
OUR_MAP = "biomass_late_fusion_mean.tif"
WORLDCOVER = "worldcover_dev.tif"   # 10 m; we sample it onto the 100 m grid

NODATA = -9999.0

AGBD_BINS = [0, 50, 100, 150, 200, 300]
AGBD_BIN_LABELS = ["0-50", "50-100", "100-150", "150-200", "200-300"]

WORLDCOVER_NAMES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse", 70: "snow_ice", 80: "water",
    90: "wetland", 95: "mangroves", 100: "moss_lichen",
}


def reproject_cci_to_grid(our_map_path: Path) -> tuple[np.ndarray, dict]:
    """Reproject the CCI tile onto our map's exact grid (bilinear). Returns
    (cci_on_grid, our_profile) where cci_on_grid is float32 with NaN for CCI zeros
    kept as 0 (we handle the zero semantics downstream)."""
    with rasterio.open(our_map_path) as ref:
        ref_profile = ref.profile.copy()
        dst_shape = (ref.height, ref.width)
        dst_transform = ref.transform
        dst_crs = ref.crs

    cci_path = CCI_DIR / CCI_FILE
    with rasterio.open(cci_path) as cci:
        dst = np.zeros(dst_shape, dtype=np.float32)
        reproject(
            source=rasterio.band(cci, 1),
            destination=dst,
            src_transform=cci.transform,
            src_crs=cci.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return dst, ref_profile


def sample_worldcover_to_grid(our_map_path: Path) -> np.ndarray:
    """Reproject WorldCover (10 m, categorical) onto our 100 m grid with nearest
    neighbor, returning the majority-ish class per 100 m cell (nearest is adequate
    for a coarse cross-tab)."""
    with rasterio.open(our_map_path) as ref:
        dst_shape = (ref.height, ref.width)
        dst_transform = ref.transform
        dst_crs = ref.crs

    wc_path = PROCESSED_DIR / WORLDCOVER
    with rasterio.open(wc_path) as wc:
        dst = np.zeros(dst_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(wc, 1),
            destination=dst,
            src_transform=wc.transform,
            src_crs=wc.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    return dst


def compute_metrics(our: np.ndarray, cci: np.ndarray) -> dict:
    """Standard agreement metrics between our map and CCI over a set of pixels."""
    if len(our) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan,
                "corr": np.nan, "our_mean": np.nan, "cci_mean": np.nan}
    err = our - cci
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))  # positive = our map higher than CCI
    corr = float(np.corrcoef(our, cci)[0, 1]) if len(our) > 1 else np.nan
    return {"n": len(our), "rmse": rmse, "mae": mae, "bias": bias, "corr": corr,
            "our_mean": float(our.mean()), "cci_mean": float(cci.mean())}


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("Phase 5 step 3: compare wall-to-wall map against ESA CCI Biomass v5")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    our_map_path = MAPS_DIR / OUR_MAP

    # 1. Load our map
    with rasterio.open(our_map_path) as src:
        our_map = src.read(1)
        our_profile = src.profile.copy()
    log.info(f"  Our map: {our_map.shape}, "
             f"valid {(our_map != NODATA).sum():,}")

    # 2. Reproject CCI onto our grid
    log.info("  Reprojecting CCI to our grid (bilinear)")
    cci_on_grid, _ = reproject_cci_to_grid(our_map_path)

    # Save the reprojected CCI for figures/inspection
    cci_out = OUT_DIR / "cci_on_grid.tif"
    cci_profile = our_profile.copy()
    cci_profile.update(dtype="float32", nodata=NODATA, compress="deflate")
    # Mark CCI zeros distinctly from our nodata: keep as 0 here, semantics handled below
    with rasterio.open(cci_out, "w", **cci_profile) as dst:
        dst.write(cci_on_grid.astype(np.float32), 1)
    log.info(f"  Wrote {cci_out.name}")

    # 3. Sample WorldCover onto the grid for the qualitative cross-tab
    log.info("  Sampling WorldCover onto grid (nearest)")
    wc_on_grid = sample_worldcover_to_grid(our_map_path)

    # 4. Build masks
    our_valid = our_map != NODATA
    cci_forest = cci_on_grid > 0            # CCI claims woody biomass
    cci_zero = (cci_on_grid == 0)

    # Comparison A: forest pixels (both valid, CCI > 0)
    mask_A = our_valid & cci_forest
    log.info("")
    log.info(f"  Comparison A (forest, CCI>0 & our valid): {mask_A.sum():,} pixels")

    our_A = our_map[mask_A]
    cci_A = cci_on_grid[mask_A]

    # Overall metrics
    overall = compute_metrics(our_A, cci_A)
    log.info("")
    log.info("  === Agreement over forest pixels (Comparison A) ===")
    log.info(f"    n={overall['n']:,}")
    log.info(f"    our mean={overall['our_mean']:.1f}, CCI mean={overall['cci_mean']:.1f} Mg/ha")
    log.info(f"    RMSE={overall['rmse']:.1f}, MAE={overall['mae']:.1f}, "
             f"bias={overall['bias']:+.1f} (our - CCI)")
    log.info(f"    correlation={overall['corr']:.3f}")

    # Per-AGBD-bin agreement (binned on CCI value)
    rows = [{"stratum": "overall", **overall}]
    for i in range(len(AGBD_BINS) - 1):
        lo, hi = AGBD_BINS[i], AGBD_BINS[i + 1]
        b = (cci_A >= lo) & (cci_A < hi)
        m = compute_metrics(our_A[b], cci_A[b])
        m["stratum"] = f"cci_{AGBD_BIN_LABELS[i]}"
        rows.append(m)
        log.info(f"    CCI bin {AGBD_BIN_LABELS[i]:>8}: n={m['n']:>8,}  "
                 f"RMSE={m['rmse']:6.1f}  bias={m['bias']:+6.1f}  corr={m['corr']:.3f}")

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "comparison_vs_cci_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    log.info(f"  Wrote {metrics_path.name}")

    # 5. Comparison B: characterize CCI zeros by WorldCover class
    log.info("")
    log.info("  === Where CCI is zero, what does WorldCover say? (Comparison B) ===")
    zero_and_valid = cci_zero & our_valid
    lc_rows = []
    for cls in sorted(np.unique(wc_on_grid[zero_and_valid])):
        if cls == 0:
            continue
        sel = zero_and_valid & (wc_on_grid == cls)
        n = int(sel.sum())
        our_mean_here = float(our_map[sel].mean()) if n > 0 else np.nan
        name = WORLDCOVER_NAMES.get(int(cls), f"class_{cls}")
        pct = n / zero_and_valid.sum() * 100
        lc_rows.append({"lc_class": int(cls), "lc_name": name, "n": n,
                        "pct_of_cci_zero": pct, "our_mean_agbd": our_mean_here})
        log.info(f"    {name:14s}: {n:>9,} ({pct:4.1f}%)  our map predicts "
                 f"mean {our_mean_here:.1f} Mg/ha here")

    lc_df = pd.DataFrame(lc_rows)
    lc_path = OUT_DIR / "comparison_vs_cci_landcover.csv"
    lc_df.to_csv(lc_path, index=False)
    log.info(f"  Wrote {lc_path.name}")

    log.info("")
    log.info("Done. Forest-pixel agreement is the primary CCI comparison;")
    log.info("the CCI-zero / WorldCover cross-tab characterizes the definitional difference.")


if __name__ == "__main__":
    main()