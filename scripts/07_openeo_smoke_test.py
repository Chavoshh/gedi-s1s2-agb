"""Phase 2, step 2: minimal end-to-end openEO smoke test.

Builds a 1-month NDVI median composite over a 1 km x 1 km tile in central
Galicia, downloads the result as GeoTIFF, prints summary statistics, and
plots it. Designed to cost minimal credits (~5).
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import openeo
import rioxarray
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

OPENEO_BACKEND = "openeofed.dataspace.copernicus.eu"

# A 1 km x 1 km tile inside the dev AOI (forest-dominated area near Lugo).
# Roughly 0.01 deg lon x 0.01 deg lat at this latitude is ~800 m x 1100 m.
TEST_BBOX = {
    "west":  -7.55,
    "south": 43.00,
    "east":  -7.54,
    "north": 43.01,
}
TEST_WINDOW = ("2020-07-01", "2020-07-31")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    out_dir = INTERIM_DIR / "openeo_smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = INTERIM_DIR / "logs" / "07_smoke_test.log"
    setup_logging(cfg.log_level, log_file=log_file)

    log.info("Connecting to openEO...")
    connection = openeo.connect(OPENEO_BACKEND).authenticate_oidc()

    log.info(f"Building NDVI composite for {TEST_BBOX} over {TEST_WINDOW}")

    # Load 3 bands (red, NIR, plus SCL for cloud masking)
    cube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=TEST_BBOX,
        temporal_extent=list(TEST_WINDOW),
        bands=["B04", "B08", "SCL"],
        max_cloud_cover=70,
    )

    # Cloud mask: SCL classes 4 (vegetation), 5 (bare soil), 6 (water),
    # 7 (unclassified) are "clean ground". Mask out everything else.
    # Reference: https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-2a/algorithm-overview
    scl = cube.band("SCL")
    valid = (scl == 4) | (scl == 5) | (scl == 6) | (scl == 7)
    cube_masked = cube.mask(~valid)

    # Compute NDVI: (NIR - RED) / (NIR + RED)
    red = cube_masked.band("B04")
    nir = cube_masked.band("B08")
    ndvi = (nir - red) / (nir + red)

    # Reduce time axis: median across all clean observations in the month.
    ndvi_median = ndvi.reduce_dimension(reducer="median", dimension="t")

    # Synchronous download. Small enough to not need a batch job.
    output_path = out_dir / "ndvi_median.tif"
    log.info(f"Submitting synchronous download to {output_path}")
    ndvi_median.download(str(output_path), format="GTiff")
    log.info(f"Downloaded {output_path.stat().st_size / 1024:.1f} KB")

    # Inspect what we got
    log.info("\nReading result back with rioxarray...")
    da = rioxarray.open_rasterio(output_path)
    log.info(f"Shape: {da.shape}")
    log.info(f"CRS: {da.rio.crs}")
    log.info(f"Bounds: {da.rio.bounds()}")
    log.info(f"Pixel size: {da.rio.resolution()}")
    arr = da.values.squeeze()
    log.info(f"NDVI stats: "
             f"min={arr.min():.3f}, "
             f"max={arr.max():.3f}, "
             f"mean={arr.mean():.3f}, "
             f"n_nan={int((arr != arr).sum())}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
    plt.colorbar(im, ax=ax, label="NDVI")
    ax.set_title("NDVI median composite, July 2020\n"
                 f"Tile: {TEST_BBOX['west']}–{TEST_BBOX['east']}°E, "
                 f"{TEST_BBOX['south']}–{TEST_BBOX['north']}°N")
    ax.axis("off")
    plot_path = out_dir / "ndvi_median.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.info(f"\nPlot saved to {plot_path}")

    log.info("\nSmoke test complete.")


if __name__ == "__main__":
    main()