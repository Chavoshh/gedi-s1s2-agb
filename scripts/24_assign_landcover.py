"""Phase 4 prep: assign a majority land cover class to each patch.

Reads the patches Zarr and the WorldCover raster from script 23, looks up the
25 x 25 window of WorldCover values at each patch's location, computes the
majority class, and writes a parquet mapping patch_id -> lc_class.

The output (data/processed/patch_landcover_dev.parquet) is consumed by
script 25 (evaluation) for stratified metrics by land cover.

Usage:
    uv run python scripts/24_assign_landcover.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
import zarr
from omegaconf import DictConfig
from rasterio.windows import Window

from biomass.config import PROCESSED_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

PATCH_SIZE = 25  # must match the patch extraction
WORLDCOVER_PATH = PROCESSED_DIR / "worldcover_dev.tif"

CLASS_NAMES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built_up", 60: "bare_sparse", 70: "snow_ice", 80: "water",
    90: "wetland", 95: "mangroves", 100: "moss_lichen",
}


def majority_class(window_values: np.ndarray) -> tuple[int, float]:
    """Return (majority_class, fraction) for a window of WorldCover values."""
    if window_values.size == 0:
        return 0, 0.0
    classes, counts = np.unique(window_values, return_counts=True)
    idx = np.argmax(counts)
    return int(classes[idx]), float(counts[idx]) / window_values.size


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("Phase 4: assigning majority land cover class to each patch")

    zarr_path = PROCESSED_DIR / f"patches_{cfg.aoi.name}.zarr"
    out_path = PROCESSED_DIR / f"patch_landcover_{cfg.aoi.name}.parquet"
    log.info(f"Patches:    {zarr_path}")
    log.info(f"WorldCover: {WORLDCOVER_PATH}")
    log.info(f"Output:     {out_path}")

    # Read patch positions as UTM coordinates from the Zarr.
    root = zarr.open_group(str(zarr_path), mode="r")
    eastings = root["eastings"][:]    # shape (N,) UTM 29N easting in meters
    northings = root["northings"][:]  # shape (N,) UTM 29N northing in meters
    patch_ids = np.arange(len(eastings))
    n_patches = len(patch_ids)
    log.info(f"Loaded {n_patches:,} patch positions from Zarr")

    # Open WorldCover and convert UTM coords -> raster (row, col) using the transform.
    classes = np.zeros(n_patches, dtype=np.uint8)
    fractions = np.zeros(n_patches, dtype=np.float32)
    log.info("Computing majority class for each patch (this may take a few minutes)")

    with rasterio.open(WORLDCOVER_PATH) as src:
        # rasterio's inverse transform: (easting, northing) -> (col, row).
        inv = ~src.transform
        half = PATCH_SIZE // 2  # 12

        for i in range(n_patches):
            col_f, row_f = inv * (float(eastings[i]), float(northings[i]))
            col = int(round(col_f))
            row = int(round(row_f))
            window = Window(col - half, row - half, PATCH_SIZE, PATCH_SIZE)
            window_data = src.read(1, window=window)
            cls, frac = majority_class(window_data)
            classes[i] = cls
            fractions[i] = frac

            if (i + 1) % 50000 == 0:
                log.info(f"  Processed {i + 1:,} / {n_patches:,} patches")

    df = pd.DataFrame({
        "patch_id": patch_ids,
        "lc_class": classes,
        "lc_name": [CLASS_NAMES.get(int(c), "unknown") for c in classes],
        "lc_purity": fractions,
    })
    df.to_parquet(out_path, index=False)
    log.info(f"Wrote {out_path} ({len(df):,} rows)")

    splits = root["splits"][:]
    test_mask = splits == 2

    log.info("")
    log.info("Land cover distribution over ALL patches:")
    _report_distribution(df)

    log.info("")
    log.info(f"Land cover distribution over TEST patches ({test_mask.sum():,}):")
    _report_distribution(df[test_mask].reset_index(drop=True))


def _report_distribution(df: pd.DataFrame) -> None:
    n = len(df)
    counts = df["lc_class"].value_counts().sort_index()
    for cls, count in counts.items():
        name = CLASS_NAMES.get(int(cls), "unknown")
        pct = count / n * 100
        log.info(f"  {int(cls):3d} ({name:18s}): {count:>10,} patches ({pct:5.1f}%)")
    log.info(f"  Mean purity: {df['lc_purity'].mean():.3f}  median: {df['lc_purity'].median():.3f}")


if __name__ == "__main__":
    main()