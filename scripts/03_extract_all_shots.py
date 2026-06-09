"""Phase 1, step 3: extract all quality-filtered GEDI L4A shots over an AOI."""
from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR, RAW_DIR
from biomass.data.aoi import get_aoi
from biomass.data.gedi_pipeline import run_extraction
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    aoi = get_aoi(cfg.aoi.name)

    log_file = INTERIM_DIR / "logs" / f"03_extract_{aoi.name}.log"
    setup_logging(cfg.log_level, log_file=log_file)

    log.info(f"=== GEDI extraction: AOI={aoi.name} ===")
    log.info(f"Log file: {log_file}")

    output = run_extraction(
        aoi=aoi,
        time_range=(cfg.time_range.start, cfg.time_range.end),
        cache_dir=RAW_DIR / "gedi",
        shards_dir=INTERIM_DIR / f"gedi_shards_{aoi.name}",
        state_file=INTERIM_DIR / f"gedi_state_{aoi.name}.json",
        failed_log=INTERIM_DIR / f"gedi_failed_{aoi.name}.txt",
        output_path=INTERIM_DIR / f"gedi_shots_{aoi.name}.parquet",
        quality=dict(cfg.quality) if cfg.quality else None,
    )

    log.info(f"Done. Final output: {output}")


if __name__ == "__main__":
    main()