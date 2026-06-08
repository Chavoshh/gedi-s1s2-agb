"""Phase 1, step 1: confirm Earthdata auth and count L4A granules over the AOI."""
from __future__ import annotations

import logging

import earthaccess
import hydra
from omegaconf import DictConfig

from biomass.config import L4A_SHORT_NAME
from biomass.data.aoi import get_aoi
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)

    aoi = get_aoi(cfg.aoi.name)
    log.info(f"AOI: {aoi.name} — {aoi.description}")
    log.info(f"  bbox: {aoi.bbox}")
    log.info(f"  time range: {cfg.time_range.start} to {cfg.time_range.end}")

    auth = earthaccess.login(strategy="netrc")
    assert auth.authenticated, "Earthdata auth failed — check credentials"

    results = earthaccess.search_data(
        short_name=L4A_SHORT_NAME,
        bounding_box=aoi.bbox,
        temporal=(cfg.time_range.start, cfg.time_range.end),
        count=-1,
    )
    log.info(f"Found {len(results)} L4A granules over {aoi.name}")

    if results:
        log.info(f"First granule: {results[0]['umm']['GranuleUR']}")


if __name__ == "__main__":
    main()