"""Phase 2, step 1 (revised): connect to CDSE openEO, authenticate, and confirm
the Sentinel-1 GRD and Sentinel-2 L2A collections are accessible."""
from __future__ import annotations

import logging

import hydra
import openeo
from omegaconf import DictConfig

from biomass.data.aoi import get_aoi
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

# Federated backend includes partner resources alongside core CDSE.
OPENEO_BACKEND = "openeofed.dataspace.copernicus.eu"

# openEO collection IDs (note: different from STAC IDs)
S2_COLLECTION = "SENTINEL2_L2A"
S1_COLLECTION = "SENTINEL1_GRD"


def describe(connection: openeo.Connection, collection_id: str) -> None:
    """Print key metadata for a collection."""
    log.info(f"\n=== {collection_id} ===")
    meta = connection.describe_collection(collection_id)
    log.info(f"Title: {meta.get('title', '?')}")

    # Temporal coverage
    extent = meta.get("extent", {}).get("temporal", {}).get("interval", [[None, None]])
    log.info(f"Temporal coverage: {extent[0][0]} to {extent[0][1] or 'present'}")

    # Available bands (under cube:dimensions / summaries)
    summaries = meta.get("summaries", {})
    bands = summaries.get("eo:bands") or summaries.get("bands")
    if bands:
        band_names = [b.get("name", "?") for b in bands]
        log.info(f"Bands ({len(band_names)}): {band_names}")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)

    aoi = get_aoi(cfg.aoi.name)
    log.info(f"AOI: {aoi.name} — bbox {aoi.bbox}")
    log.info(f"Connecting to openEO backend: {OPENEO_BACKEND}")

    connection = openeo.connect(OPENEO_BACKEND)
    log.info(f"Connected. Backend: {connection.capabilities().get('title', '?')}")

    log.info("\nAuthenticating via OIDC...")
    log.info("(First run will print a URL — visit it in your browser to log in.)")
    connection.authenticate_oidc()
    log.info("Authenticated.")

    describe(connection, S2_COLLECTION)
    describe(connection, S1_COLLECTION)

    log.info("\nopenEO access confirmed. Ready for Phase 2.2.")


if __name__ == "__main__":
    main()