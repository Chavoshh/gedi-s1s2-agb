"""Phase 2, step 4: build one annual Sentinel-2 median composite over the AOI.

Cloud-masked using SCL. Output is a single multi-band GeoTIFF in
data/processed/s2_composite_<aoi>_<year>.tif.

Usage:
    uv run python scripts/09_build_s2_composite.py year=2020
    uv run python scripts/09_build_s2_composite.py year=2021 aoi=dev
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import hydra
import openeo
import rioxarray
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR, PROCESSED_DIR
from biomass.data.aoi import get_aoi
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

OPENEO_BACKEND = "openeofed.dataspace.copernicus.eu"

# Spectral bands to keep (10 m + 20 m, drop 60 m atmospheric)
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B11", "B12"]
# Plus SCL for cloud masking (dropped from output)
S2_BANDS_WITH_SCL = S2_BANDS + ["SCL"]


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    # The year is a hydra-overridable CLI arg (default 2020)
    year = int(cfg.get("year", 2020))
    aoi = get_aoi(cfg.aoi.name)

    out_dir = PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"s2_composite_{aoi.name}_{year}.tif"

    log_file = INTERIM_DIR / "logs" / f"09_s2_composite_{aoi.name}_{year}.log"
    setup_logging(cfg.log_level, log_file=log_file)

    log.info(f"=== S2 annual composite: AOI={aoi.name}, year={year} ===")
    log.info(f"Bbox: {aoi.bbox}")
    log.info(f"Output: {out_path}")

    if out_path.exists():
        log.warning(f"Output already exists at {out_path}. "
                    f"Delete it manually if you want to rebuild.")
        return

    log.info("Connecting to openEO...")
    connection = openeo.connect(OPENEO_BACKEND).authenticate_oidc()

    # Build the data cube
    spatial_extent = {
        "west":  aoi.west,
        "south": aoi.south,
        "east":  aoi.east,
        "north": aoi.north,
    }
    temporal_extent = [f"{year}-01-01", f"{year}-12-31"]

    log.info(f"Loading SENTINEL2_L2A: {S2_BANDS_WITH_SCL}")
    cube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=S2_BANDS_WITH_SCL,
        max_cloud_cover=70,
    )

    # Cloud mask: keep SCL classes 4 (vegetation), 5 (bare soil),
    # 6 (water), 7 (unclassified). Mask everything else.
    log.info("Applying cloud mask via SCL...")
    scl = cube.band("SCL")
    valid = (scl == 4) | (scl == 5) | (scl == 6) | (scl == 7)
    cube_masked = cube.mask(~valid)

    # Drop SCL from the composite output
    cube_for_composite = cube_masked.filter_bands(S2_BANDS)

    # Median across time
    log.info("Reducing time dimension with median...")
    composite = cube_for_composite.reduce_dimension(
        reducer="median", dimension="t"
    )

    # Submit as batch job
    job_title = f"biomass_s2_{aoi.name}_{year}_annual"
    log.info(f"\nSubmitting batch job: {job_title}")
    job = composite.create_job(title=job_title, out_format="GTiff")
    log.info(f"Job ID: {job.job_id}")
    log.info("Starting job. This may take 15-60 minutes...")
    job.start_and_wait()

    # Get final job status and cost
    desc = job.describe_job()
    log.info(f"\nJob finished. Status: {desc.get('status')}")
    log.info(f"Credits used: {desc.get('costs', 'unknown')}")
    log.info(f"Duration: {desc.get('duration', 'unknown')}")

    if desc.get("status") != "finished":
        log.error("Job did not finish successfully. Inspect on the openEO web editor.")
        return

    # Download results to a temp directory, then move/rename to the canonical output
    tmp_dir = out_dir / f"_tmp_{aoi.name}_{year}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"\nDownloading result files to {tmp_dir}...")
    job.get_results().download_files(str(tmp_dir))

    tifs = sorted(tmp_dir.glob("*.tif"))
    log.info(f"Got {len(tifs)} TIFF file(s):")
    for t in tifs:
        log.info(f"  {t.name}: {t.stat().st_size / 1e6:.1f} MB")

    if len(tifs) == 1:
        # Single multi-band file (expected)
        shutil.move(str(tifs[0]), str(out_path))
        log.info(f"\nMoved to {out_path}")
    else:
        log.warning(
            f"Got {len(tifs)} files instead of 1; leaving them in {tmp_dir} for inspection."
        )
        return

    # Clean up the temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Quick sanity check on the output
    log.info("\nInspecting output file...")
    da = rioxarray.open_rasterio(out_path)
    log.info(f"Shape: {da.shape}")
    log.info(f"CRS: {da.rio.crs}")
    log.info(f"Bounds: {da.rio.bounds()}")
    log.info(f"Pixel size: {da.rio.resolution()}")
    log.info(f"Approximate size: "
             f"{(da.rio.bounds()[2] - da.rio.bounds()[0]) / 1000:.1f} km wide x "
             f"{(da.rio.bounds()[3] - da.rio.bounds()[1]) / 1000:.1f} km tall")
    log.info(f"Final file size: {out_path.stat().st_size / 1e6:.1f} MB")

    log.info(f"\nDone. Composite at: {out_path}")
    log.info(f"Job ID for reference: {job.job_id}")


if __name__ == "__main__":
    main()