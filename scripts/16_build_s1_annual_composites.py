"""Phase 2, step 8 (Hyp3 path): assemble annual Sentinel-1 composites from
the 36 downloaded Hyp3 RTC scenes.

For each year (2020, 2021, 2022):
  - Read 12 monthly scenes (VV, VH, incidence map) from data/raw/s1_hyp3/
  - Reproject + resample each to the S2 composite grid (10 m, EPSG:32629)
  - Median across time per band (linear power for VV/VH, degrees for LIA)
  - Convert VV and VH to dB
  - Write data/processed/s1_composite_dev_<year>.tif (3 bands: VV_dB, VH_dB, LIA_deg)

Usage:
    uv run python scripts/16_build_s1_annual_composites.py
    uv run python scripts/16_build_s1_annual_composites.py year=2020
"""
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig
from rasterio.enums import Resampling
from rasterio.warp import reproject

from biomass.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from biomass.data.aoi import get_aoi
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

HYP3_OUT_DIR = RAW_DIR / "s1_hyp3"
MANIFEST_PATH = INTERIM_DIR / "s1_scene_manifest.json"

YEARS = [2020, 2021, 2022]


def find_band_files(scene_dir: Path) -> dict[str, Path]:
    """Locate VV, VH, and incidence-angle GeoTIFFs in a Hyp3 scene directory."""
    tifs = list(scene_dir.rglob("*.tif"))
    found = {}
    for t in tifs:
        name = t.name.lower()
        if name.endswith("_vv.tif"):
            found["VV"] = t
        elif name.endswith("_vh.tif"):
            found["VH"] = t
        elif "inc_map" in name:
            found["LIA"] = t
    return found


def reproject_to_reference(
    src_path: Path,
    ref_transform,
    ref_crs,
    ref_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Open src_path, reproject and resample to match the reference grid.
    Returns a float32 array of shape ref_shape with NaN for nodata.
    """
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        src_nodata = src.nodata
        # Hyp3 GeoTIFFs are usually float32 with 0 = no-data
        src_data = src_data.astype(np.float32)

        dst = np.full(ref_shape, np.nan, dtype=np.float32)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=resampling,
            src_nodata=src_nodata if src_nodata is not None else 0,
            dst_nodata=np.nan,
        )
    return dst


def build_year_composite(
    year: int,
    aoi_name: str,
    scenes_for_year: dict[str, dict],
    ref_path: Path,
    out_path: Path,
) -> None:
    """Build one annual composite from the 12 monthly scenes."""
    log.info(f"\n=== Building {year} composite ===")
    log.info(f"Reference grid: {ref_path.name}")

    # Read reference grid from S2 composite
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_shape = (ref.height, ref.width)
        ref_profile = ref.profile.copy()
    log.info(f"Target shape: {ref_shape}, CRS: {ref_crs}")

    # Gather scene directories for this year
    months = sorted(scenes_for_year.keys(), key=int)
    log.info(f"Found {len(months)} months: {months}")

    n_months = len(months)
    h, w = ref_shape

    # Pre-allocate stacks. We work with one band's stack at a time to
    # keep peak memory ~5-6 GB rather than 16+ GB.
    band_names = ["VV", "VH", "LIA"]
    out_bands: list[np.ndarray] = []

    for band_name in band_names:
        log.info(f"\n--- Band: {band_name} ---")
        stack = np.empty((n_months, h, w), dtype=np.float32)

        for i, month in enumerate(months):
            scene_name = scenes_for_year[month]["scene_name"]
            scene_dir = HYP3_OUT_DIR / scene_name
            band_files = find_band_files(scene_dir)
            if band_name not in band_files:
                log.warning(f"  month {month}: {band_name} not found, "
                            f"using all-NaN slice")
                stack[i] = np.nan
                continue

            log.info(f"  [{i+1}/{n_months}] {month}: reprojecting "
                     f"{band_files[band_name].name}")
            reprojected = reproject_to_reference(
                band_files[band_name],
                ref_transform=ref_transform,
                ref_crs=ref_crs,
                ref_shape=ref_shape,
                resampling=Resampling.bilinear,
            )
            stack[i] = reprojected

        # Mask non-positive (no-data) for VV/VH; LIA can legitimately be 0+
        if band_name in {"VV", "VH"}:
            stack[stack <= 0] = np.nan

        log.info(f"  Computing median across {n_months} months (row-chunked)...")
        median = np.empty(ref_shape, dtype=np.float32)
        chunk_rows = 1500
        for r0 in range(0, h, chunk_rows):
            r1 = min(r0 + chunk_rows, h)
            median[r0:r1] = np.nanmedian(stack[:, r0:r1, :], axis=0)
            log.info(f"    rows {r0}-{r1} of {h}")

        # Convert VV/VH from linear power to dB
        # Convert VV/VH from linear power to dB
        if band_name in {"VV", "VH"}:
            log.info(f"  Converting linear power to dB...")
            valid = ~np.isnan(median) & (median > 0)
            db_median = np.full_like(median, np.nan, dtype=np.float32)
            db_median[valid] = 10.0 * np.log10(median[valid])
            median = db_median
        elif band_name == "LIA":
            log.info(f"  Converting LIA from radians to degrees...")
            median = median * (180.0 / np.pi) 
            

        # Stats for sanity
        valid = median[~np.isnan(median)]
        if len(valid) > 0:
            log.info(f"  {band_name} stats: "
                     f"min={valid.min():.2f}, "
                     f"max={valid.max():.2f}, "
                     f"mean={valid.mean():.2f}, "
                     f"valid={len(valid):,} ({len(valid)/median.size*100:.1f}%)")

        out_bands.append(median)
        del stack
        gc.collect()

    # Stack the 3 output bands and write
    log.info(f"\nWriting 3-band composite to {out_path}...")
    output = np.stack(out_bands, axis=0)
    out_profile = ref_profile.copy()
    out_profile.update(
        count=3,
        dtype="float32",
        nodata=float("nan"),
        compress="deflate",
        predictor=3,  # float predictor
        tiled=True,
    )
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(output)
        dst.set_band_description(1, "VV_dB")
        dst.set_band_description(2, "VH_dB")
        dst.set_band_description(3, "LIA_deg")
    log.info(f"Wrote {out_path.stat().st_size / 1e6:.1f} MB")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    aoi = get_aoi(cfg.aoi.name)
    log_file = INTERIM_DIR / "logs" / f"16_s1_annual_{aoi.name}.log"
    setup_logging(cfg.log_level, log_file=log_file)

    # Allow filtering to a single year via Hydra override
    year_override = cfg.get("year", None)
    years = [int(year_override)] if year_override else YEARS

    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)

    for year in years:
        scenes_for_year = manifest["years"].get(str(year))
        if not scenes_for_year:
            log.warning(f"No scenes in manifest for {year}, skipping")
            continue

        # Reference grid: the S2 composite for the same year
        ref_path = PROCESSED_DIR / f"s2_composite_{aoi.name}_{year}.tif"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"Reference S2 composite missing: {ref_path}"
            )

        out_path = PROCESSED_DIR / f"s1_composite_{aoi.name}_{year}.tif"
        if out_path.exists():
            log.warning(f"{out_path} already exists, skipping. "
                        f"Delete it manually to rebuild.")
            continue

        build_year_composite(
            year=year,
            aoi_name=aoi.name,
            scenes_for_year=scenes_for_year,
            ref_path=ref_path,
            out_path=out_path,
        )

    log.info("\nDone.")


if __name__ == "__main__":
    main()