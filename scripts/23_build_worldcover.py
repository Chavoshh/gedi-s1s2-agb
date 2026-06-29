"""Phase 4 prep: download ESA WorldCover 2021 v200 and reproject to the dev AOI grid.

The output is data/processed/worldcover_dev.tif, a single GeoTIFF on the same
10 m UTM 29N grid as the Sentinel and DEM composites. This raster is the
source of land cover labels for patch-level stratified evaluation.

WorldCover classes (kept as integer codes; semantics in the docstring below):
    10  Tree cover
    20  Shrubland
    30  Grassland
    40  Cropland
    50  Built-up
    60  Bare / sparse vegetation
    70  Snow and ice
    80  Permanent water
    90  Herbaceous wetland
    95  Mangroves
    100 Moss and lichen

Usage:
    uv run python scripts/24_build_worldcover.py
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject

from biomass.config import PROCESSED_DIR, RAW_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

# Single tile covers our dev AOI (lon -8.5 to -7.3, lat 42.6 to 43.6).
# Tile N42W009 covers lon -9 to -6, lat 42 to 45 -- fully contains the AOI.
WORLDCOVER_URL = (
    "https://esa-worldcover.s3.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_N42W009_Map.tif"
)
RAW_TILE_PATH = RAW_DIR / "worldcover" / "ESA_WorldCover_10m_2021_v200_N42W009_Map.tif"
OUTPUT_PATH = PROCESSED_DIR / "worldcover_dev.tif"

# Target grid parameters (matches all Phase 2 rasters).
TARGET_CRS = "EPSG:32629"
TARGET_RESOLUTION = 10.0  # meters


def download_worldcover_tile() -> None:
    """Download the WorldCover tile from ESA's S3 bucket if not already present."""
    if RAW_TILE_PATH.exists():
        log.info(f"WorldCover tile already exists at {RAW_TILE_PATH}, skipping download.")
        return

    RAW_TILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading WorldCover tile from {WORLDCOVER_URL}")
    log.info(f"  Expected size: ~62 MB. Saving to {RAW_TILE_PATH}")
    urllib.request.urlretrieve(WORLDCOVER_URL, RAW_TILE_PATH)
    size_mb = RAW_TILE_PATH.stat().st_size / 1e6
    log.info(f"  Downloaded {size_mb:.1f} MB")


def reproject_to_aoi(cfg: DictConfig) -> None:
    """Reproject the WorldCover tile to the dev AOI grid (10 m UTM 29N).

    The S2 composite is the reference for grid alignment - we read its transform
    and shape and reproject WorldCover to match exactly.
    """
    # Load the S2 composite as the reference grid.
    # All Phase 2 rasters share the same grid; we pick S2-2020 arbitrarily.
    reference_path = PROCESSED_DIR / f"s2_composite_{cfg.aoi.name}_2020.tif"
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference raster not found at {reference_path}. "
            "Phase 2 outputs are required before running this script."
        )

    with rasterio.open(reference_path) as ref:
        target_transform = ref.transform
        target_width = ref.width
        target_height = ref.height
        target_crs = ref.crs
        log.info(
            f"Target grid (from {reference_path.name}): "
            f"{target_width} x {target_height} pixels at {target_crs}, "
            f"10 m resolution"
        )

    # Open the source WorldCover tile and reproject.
    with rasterio.open(RAW_TILE_PATH) as src:
        log.info(
            f"Source WorldCover: {src.width} x {src.height} pixels at {src.crs}, "
            f"dtype={src.dtypes[0]}"
        )

        # Allocate the output array. WorldCover is uint8 (values 10-100).
        dst_array = np.zeros((target_height, target_width), dtype=np.uint8)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,  # categorical data, never interpolate
        )

    # Write the reprojected raster.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        OUTPUT_PATH,
        "w",
        driver="GTiff",
        height=target_height,
        width=target_width,
        count=1,
        dtype=np.uint8,
        crs=target_crs,
        transform=target_transform,
        nodata=0,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(dst_array, 1)

    log.info(f"Wrote reprojected WorldCover to {OUTPUT_PATH}")


def sanity_check() -> None:
    """Read back the reprojected raster, report class distribution."""
    with rasterio.open(OUTPUT_PATH) as src:
        data = src.read(1)
        log.info(f"Reprojected raster: {src.width} x {src.height} pixels")
        log.info(f"  CRS: {src.crs}")
        log.info(f"  Resolution: {src.transform.a} m")

    valid = data[data > 0]
    log.info(f"Class distribution ({len(valid):,} valid pixels):")
    classes, counts = np.unique(valid, return_counts=True)
    total = counts.sum()
    class_names = {
        10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
        50: "Built-up", 60: "Bare / sparse veg", 70: "Snow / ice",
        80: "Water", 90: "Herbaceous wetland", 95: "Mangroves",
        100: "Moss and lichen",
    }
    for cls, count in zip(classes, counts):
        name = class_names.get(int(cls), "Unknown")
        pct = count / total * 100
        log.info(f"  {int(cls):3d} ({name:20s}): {count:>12,} pixels ({pct:5.1f}%)")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("Phase 4: building WorldCover raster for dev AOI")

    download_worldcover_tile()
    reproject_to_aoi(cfg)
    sanity_check()

    log.info("Done. Output ready for patch-level land cover assignment.")


if __name__ == "__main__":
    main()