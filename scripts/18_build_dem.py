"""Phase 2, step 9: build the Copernicus DEM GLO-30 reference layer.

Downloads (anonymously, via HTTPS) the four Copernicus DEM GLO-30 tiles
covering the dev AOI from the public AWS S3 bucket, mosaics them, reprojects
to match the S2 composite grid (10 m, EPSG:32629), computes slope, and writes
a 2-band GeoTIFF (elevation_m, slope_deg) at data/processed/dem_<aoi>.tif.

No openEO or Hyp3 credits used.

Usage:
    uv run python scripts/18_build_dem.py
"""
from __future__ import annotations

import logging
import math
import shutil
import urllib.request
from pathlib import Path

import hydra
import numpy as np
import rasterio
import rioxarray
from omegaconf import DictConfig
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject

from biomass.config import INTERIM_DIR, PROCESSED_DIR
from biomass.data.aoi import get_aoi
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

COP_DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"


def tiles_for_bbox(west: float, south: float, east: float, north: float) -> list[str]:
    """Determine which Copernicus DEM tiles cover a bbox.

    Tile naming: Copernicus_DSM_COG_10_<lat>_00_<lon>_00_DEM
    where <lat> = N42 / S07 / etc, <lon> = W008 / E115 / etc.
    Each tile covers a 1deg x 1deg cell, named by its southwest corner.
    """
    lat_min = math.floor(south)
    lat_max = math.floor(north - 1e-9)  # exclusive upper bound
    lon_min = math.floor(west)
    lon_max = math.floor(east - 1e-9)

    names = []
    for lat in range(lat_min, lat_max + 1):
        lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
        for lon in range(lon_min, lon_max + 1):
            lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
            names.append(f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM")
    return names


def download_tile(tile_name: str, cache_dir: Path) -> Path:
    """Download a Copernicus DEM tile to local cache."""
    url = f"{COP_DEM_BUCKET}/{tile_name}/{tile_name}.tif"
    local_path = cache_dir / f"{tile_name}.tif"
    if local_path.exists() and local_path.stat().st_size > 1_000_000:
        log.info(f"  Cached: {tile_name}")
        return local_path
    log.info(f"  Downloading {tile_name}...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, local_path)
    log.info(f"    -> {local_path.stat().st_size / 1e6:.1f} MB")
    return local_path


def compute_slope(elevation: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Compute slope in degrees from elevation using central-difference gradients.

    Both axes of `elevation` must be in meters (i.e., a projected CRS).
    Returns slope in degrees, same shape as input. NaN where elevation is NaN.
    """
    # np.gradient with explicit spacing returns derivatives in (units of elevation) / (units of pixel)
    # i.e., m/m, dimensionless, which is what we want for slope.
    dy, dx = np.gradient(elevation, pixel_size_m, pixel_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad).astype(np.float32)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    aoi = get_aoi(cfg.aoi.name)

    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dem_{aoi.name}.tif"
    cache_dir = INTERIM_DIR / "dem_tiles"
    log_file = INTERIM_DIR / "logs" / f"18_build_dem_{aoi.name}.log"
    setup_logging(cfg.log_level, log_file=log_file)

    log.info(f"=== Building DEM for AOI={aoi.name} ===")
    log.info(f"AOI bbox: {aoi.bbox}")

    if out_path.exists():
        log.warning(f"Output exists: {out_path}. Delete manually to rebuild.")
        return

    # Reference grid from the S2 composite (any year works; pick 2020)
    ref_path = PROCESSED_DIR / f"s2_composite_{aoi.name}_2020.tif"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference S2 composite missing: {ref_path}")
    log.info(f"Reference grid: {ref_path.name}")
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)
        ref_profile = ref.profile.copy()
    log.info(f"Target shape: {ref_shape}, CRS: {ref_crs}")

    # ---- 1. Download tiles ----
    tile_names = tiles_for_bbox(*aoi.bbox)
    log.info(f"\nNeed {len(tile_names)} tiles: {tile_names}")
    tile_paths = [download_tile(n, cache_dir) for n in tile_names]

    # ---- 2. Mosaic in native CRS (EPSG:4326) ----
    log.info("\nMosaicking tiles in native CRS (EPSG:4326)...")
    sources = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, mosaic_transform = merge(sources)
        mosaic_crs = sources[0].crs
        log.info(f"Mosaic shape: {mosaic.shape}, CRS: {mosaic_crs}")
    finally:
        for s in sources:
            s.close()

    # ---- 3. Reproject + resample to the S2 grid ----
    log.info(f"\nReprojecting + resampling to S2 grid (10 m, {ref_crs})...")
    elevation = np.full(ref_shape, np.nan, dtype=np.float32)
    reproject(
        source=mosaic[0].astype(np.float32),
        destination=elevation,
        src_transform=mosaic_transform,
        src_crs=mosaic_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
        src_nodata=0,            # ocean cells in Copernicus DEM are 0; treat as nodata-ish
        dst_nodata=np.nan,
    )
    valid = ~np.isnan(elevation)
    log.info(f"Elevation valid pixels: {valid.sum():,} / {elevation.size:,} "
             f"({valid.sum() / elevation.size * 100:.1f}%)")
    if valid.sum() > 0:
        e = elevation[valid]
        log.info(f"Elevation (m): min={e.min():.1f}, max={e.max():.1f}, "
                 f"mean={e.mean():.1f}, std={e.std():.1f}")

    # ---- 4. Compute slope on the resampled grid (10 m pixel size) ----
    log.info("\nComputing slope (Horn-style central differences, 10 m grid)...")
    slope = compute_slope(elevation, pixel_size_m=10.0)
    slope[~valid] = np.nan
    s_valid = slope[~np.isnan(slope)]
    if len(s_valid) > 0:
        log.info(f"Slope (deg): min={s_valid.min():.2f}, max={s_valid.max():.2f}, "
                 f"mean={s_valid.mean():.2f}, std={s_valid.std():.2f}")

    # ---- 5. Write 2-band GeoTIFF ----
    log.info(f"\nWriting {out_path}...")
    out_profile = ref_profile.copy()
    out_profile.update(
        count=2,
        dtype="float32",
        nodata=float("nan"),
        compress="deflate",
        predictor=3,   # float predictor
        tiled=True,
    )
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(elevation.astype(np.float32), 1)
        dst.write(slope, 2)
        dst.set_band_description(1, "elevation_m")
        dst.set_band_description(2, "slope_deg")
    log.info(f"Wrote {out_path.stat().st_size / 1e6:.1f} MB")

    # Final sanity check via rioxarray
    da = rioxarray.open_rasterio(out_path)
    log.info(f"\nFinal shape: {da.shape}, CRS: {da.rio.crs}")
    log.info(f"Bounds: {da.rio.bounds()}")
    log.info(f"Pixel size: {da.rio.resolution()}")

    log.info("\nDone.")


if __name__ == "__main__":
    main()