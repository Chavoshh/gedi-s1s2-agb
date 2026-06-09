"""Phase 2, step 3: empirical credit cost probe.

Runs a single batch job that mirrors the operations we'll use for real
composites: cloud-masked median over a single month, full S2 band set,
on a 10x10 km tile. The job ID and credit cost are logged so we can
extrapolate to full-AOI composite cost.
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import openeo
import rioxarray
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

OPENEO_BACKEND = "openeofed.dataspace.copernicus.eu"

# 10 km x 10 km tile, same general area as the smoke test, well inside the dev AOI.
PROBE_BBOX = {
    "west":  -7.60,
    "south": 42.95,
    "east":  -7.50,   # ~10 km wide at this latitude
    "north": 43.05,   # ~11 km tall
}
PROBE_WINDOW = ("2020-07-01", "2020-07-31")

# The S2 bands we'll actually use for the real composites.
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B11", "B12", "SCL"]


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    out_dir = INTERIM_DIR / "openeo_cost_probe"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = INTERIM_DIR / "logs" / "08_cost_probe.log"
    setup_logging(cfg.log_level, log_file=log_file)

    log.info("Connecting to openEO...")
    connection = openeo.connect(OPENEO_BACKEND).authenticate_oidc()

    log.info(f"Building S2 median composite for 10 km tile {PROBE_BBOX}")
    log.info(f"Time window: {PROBE_WINDOW}")
    log.info(f"Bands: {S2_BANDS}")

    cube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=PROBE_BBOX,
        temporal_extent=list(PROBE_WINDOW),
        bands=S2_BANDS,
        max_cloud_cover=70,
    )

    # Cloud mask using SCL: keep classes 4 (veg), 5 (bare), 6 (water), 7 (unclassified).
    scl = cube.band("SCL")
    valid = (scl == 4) | (scl == 5) | (scl == 6) | (scl == 7)
    cube_masked = cube.mask(~valid)

    # Drop SCL from output (we only need the spectral bands going forward)
    cube_for_composite = cube_masked.filter_bands(
        [b for b in S2_BANDS if b != "SCL"]
    )

    # Median across time
    composite = cube_for_composite.reduce_dimension(
        reducer="median", dimension="t"
    )

    # Submit as a batch job so we can read its credit cost afterward.
    log.info("\nSubmitting batch job...")
    job = composite.create_job(
        title="biomass_cost_probe_10km_S2_jul2020",
        out_format="GTiff",
    )
    log.info(f"Job ID: {job.job_id}")
    log.info("Starting job and waiting for completion...")
    job.start_and_wait()

    # Inspect the finished job's metadata for credit cost
    desc = job.describe_job()
    log.info(f"\nJob status: {desc.get('status')}")
    log.info(f"Credits used: {desc.get('costs', 'unknown')}")
    log.info(f"Duration: {desc.get('duration', 'unknown')}")

    # Download results
    log.info(f"\nDownloading result to {out_dir}...")
    job.get_results().download_files(str(out_dir))

    # Quick sanity check on one of the downloaded files
    tifs = sorted(out_dir.glob("*.tif"))
    if tifs:
        log.info(f"\nDownloaded {len(tifs)} files:")
        for t in tifs:
            log.info(f"  {t.name}: {t.stat().st_size / 1024:.1f} KB")
        da = rioxarray.open_rasterio(tifs[0])
        log.info(f"First file shape: {da.shape}")
        log.info(f"First file CRS: {da.rio.crs}")
        log.info(f"First file bounds: {da.rio.bounds()}")

    log.info("\nCost probe complete.")
    log.info(f"Job ID for reference: {job.job_id}")


if __name__ == "__main__":
    main()