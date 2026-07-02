"""Phase 5 step 5: mask boundary extrapolation artifacts in the biomass maps.

Patches straddling the AOI boundary receive partially nodata-filled input (missing
pixels replaced by channel mean). The model, seeing out-of-distribution input,
extrapolates to implausibly high values that were clamped to the training cap
(500 Mg/ha). These clipped pixels (~0.2% of the map, clustered in corners) are not
reliable predictions and should be nodata.

This script rewrites each map, converting pixels at/near the clip ceiling to nodata,
and reports how many were affected. Idempotent.

Usage:
    uv run python scripts/32_clean_map_artifacts.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import rasterio
from omegaconf import DictConfig

from biomass.config import PROCESSED_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

MAPS_DIR = PROCESSED_DIR / "maps"
NODATA = -9999.0
CLIP_THRESHOLD = 450.0  # pixels >= this are treated as unreliable extrapolation

MAPS = [
    "biomass_late_fusion_seed42.tif", "biomass_late_fusion_seed7.tif",
    "biomass_late_fusion_seed123.tif", "biomass_late_fusion_mean.tif",
    "biomass_s2_only_seed7.tif", "biomass_s1_only_seed7.tif",
    "biomass_early_fusion_seed7.tif",
]


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("Phase 5 step 5: masking boundary extrapolation artifacts")
    log.info(f"  Threshold: pixels >= {CLIP_THRESHOLD} Mg/ha -> nodata")

    for name in MAPS:
        path = MAPS_DIR / name
        if not path.exists():
            log.warning(f"  Missing {name}, skipping")
            continue
        with rasterio.open(path) as src:
            data = src.read(1)
            profile = src.profile.copy()
            nod = src.nodata if src.nodata is not None else NODATA

        valid = np.isfinite(data) & (data != nod)
        artifact = valid & (data >= CLIP_THRESHOLD)
        n_artifact = int(artifact.sum())
        n_valid = int(valid.sum())

        data[artifact] = NODATA
        profile.update(nodata=NODATA)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)

        log.info(f"  {name}: masked {n_artifact:,} artifact pixels "
                 f"({n_artifact/n_valid*100:.3f}% of valid)")

    log.info("")
    log.info("Done. Re-run script 29 (ensemble) and script 30 (CCI) and 31 (figures)")
    log.info("to propagate the cleaned maps through downstream products.")


if __name__ == "__main__":
    main()