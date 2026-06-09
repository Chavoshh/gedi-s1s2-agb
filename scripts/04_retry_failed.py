"""Retry granules listed in the per-AOI failures file.

Reads data/interim/gedi_failed_<aoi>.txt, attempts each granule again,
appends successful results as new shards, and re-aggregates the final parquet.
"""
from __future__ import annotations

import logging
from pathlib import Path

import earthaccess
import hydra
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR, L4A_SHORT_NAME, RAW_DIR
from biomass.data.aoi import get_aoi
from biomass.data.gedi_pipeline import aggregate_shards, process_one_granule
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)


def read_failed_urs(failed_log: Path) -> list[str]:
    """Parse failure log; return unique granule URs."""
    if not failed_log.exists():
        return []
    urs = set()
    for line in failed_log.read_text().splitlines():
        if not line.strip():
            continue
        urs.add(line.split("\t", 1)[0])
    return sorted(urs)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    aoi = get_aoi(cfg.aoi.name)

    log_file = INTERIM_DIR / "logs" / f"04_retry_{aoi.name}.log"
    setup_logging(cfg.log_level, log_file=log_file)

    failed_log_path = INTERIM_DIR / f"gedi_failed_{aoi.name}.txt"
    failed_urs = read_failed_urs(failed_log_path)
    log.info(f"=== Retry: AOI={aoi.name} ===")
    log.info(f"Failures to retry: {len(failed_urs)}")

    if not failed_urs:
        log.info("Nothing to retry. Done.")
        return

    # Re-search the catalog and match by GranuleUR.
    auth = earthaccess.login(strategy="netrc")
    assert auth.authenticated

    all_results = earthaccess.search_data(
        short_name=L4A_SHORT_NAME,
        bounding_box=aoi.bbox,
        temporal=(cfg.time_range.start, cfg.time_range.end),
        count=-1,
    )
    by_ur = {g["umm"]["GranuleUR"]: g for g in all_results}

    cache_dir = RAW_DIR / "gedi"
    shards_dir = INTERIM_DIR / f"gedi_shards_{aoi.name}"
    # IMPORTANT: write retries to a separate "retry" file so we know
    # what still failed after the second attempt without losing history.
    retry_failed_log = INTERIM_DIR / f"gedi_failed_{aoi.name}_retry.txt"

    total_kept = 0
    still_failed = 0
    for i, ur in enumerate(failed_urs):
        log.info(f"[{i+1}/{len(failed_urs)}] {ur}")
        g = by_ur.get(ur)
        if g is None:
            log.warning(f"  granule not found in catalog (skipping)")
            still_failed += 1
            continue

        n_kept = process_one_granule(
            granule=g,
            cache_dir=cache_dir,
            shards_dir=shards_dir,
            failed_log=retry_failed_log,
            aoi=aoi,
        )
        if n_kept == 0 and retry_failed_log.exists():
            still_failed += 1
        else:
            total_kept += n_kept
            log.info(f"  kept {n_kept} shots")

    log.info(f"\nRetry summary: {len(failed_urs) - still_failed} succeeded, "
             f"{still_failed} still failed")
    log.info(f"New shots added: {total_kept}")

    if total_kept > 0:
        # Re-aggregate the (now larger) shards directory into the final parquet.
        log.info("\nRe-aggregating shards...")
        output_path = INTERIM_DIR / f"gedi_shots_{aoi.name}.parquet"
        aggregate_shards(shards_dir, output_path)


if __name__ == "__main__":
    main()