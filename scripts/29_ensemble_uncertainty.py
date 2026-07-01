"""Phase 5 step 2: ensemble mean and uncertainty from the 3 late-fusion seed maps.

Loads the three late-fusion wall-to-wall maps (seeds 42, 7, 123) and computes:
  - biomass_late_fusion_mean.tif : per-pixel mean across seeds (the published map)
  - biomass_late_fusion_std.tif  : per-pixel std across seeds (uncertainty)

A pixel is valid only where all three seed maps are valid (all finite, none nodata).
The std map quantifies epistemic uncertainty: high std = the three initializations
disagree = lower confidence in the prediction.

Usage:
    uv run python scripts/29_ensemble_uncertainty.py
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
SEED_MAPS = [
    "biomass_late_fusion_seed42.tif",
    "biomass_late_fusion_seed7.tif",
    "biomass_late_fusion_seed123.tif",
]
MEAN_OUT = "biomass_late_fusion_mean.tif"
STD_OUT = "biomass_late_fusion_std.tif"
NODATA = -9999.0


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("Phase 5 step 2: ensemble mean + uncertainty from 3 late-fusion seeds")

    # Load the three seed maps and their profiles
    stacks = []
    ref_profile = None
    ref_transform = None
    ref_crs = None
    for name in SEED_MAPS:
        path = MAPS_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing seed map: {path}")
        with rasterio.open(path) as src:
            data = src.read(1)
            if ref_profile is None:
                ref_profile = src.profile.copy()
                ref_transform = src.transform
                ref_crs = src.crs
            else:
                # Sanity: all three maps must share the same grid
                if src.width != ref_profile["width"] or src.height != ref_profile["height"]:
                    raise ValueError(f"{name} grid mismatch vs reference")
        stacks.append(data)
        log.info(f"  Loaded {name}: shape {data.shape}")

    arr = np.stack(stacks, axis=0)  # (3, H, W)

    # Validity: a pixel is valid where all three seeds are finite and not nodata
    valid = np.all(np.isfinite(arr) & (arr != NODATA), axis=0)  # (H, W)
    n_valid = int(valid.sum())
    log.info(f"  Valid pixels (all 3 seeds present): {n_valid:,} "
             f"({n_valid / valid.size * 100:.1f}%)")

    # Compute mean and std only where valid; nodata elsewhere
    mean_map = np.full(valid.shape, NODATA, dtype=np.float32)
    std_map = np.full(valid.shape, NODATA, dtype=np.float32)

    # Masked computation
    valid_stack = arr[:, valid]  # (3, n_valid)
    mean_map[valid] = valid_stack.mean(axis=0)
    std_map[valid] = valid_stack.std(axis=0, ddof=1)  # sample std across 3 seeds

    # Report statistics
    mv = mean_map[valid]
    sv = std_map[valid]
    log.info("")
    log.info(f"  Ensemble mean:  min={mv.min():.1f}  mean={mv.mean():.1f}  "
             f"median={np.median(mv):.1f}  max={mv.max():.1f} Mg/ha")
    log.info(f"  Uncertainty std: min={sv.min():.2f}  mean={sv.mean():.2f}  "
             f"median={np.median(sv):.2f}  p95={np.percentile(sv, 95):.2f}  "
             f"max={sv.max():.2f} Mg/ha")
    # Coefficient of variation where mean is nontrivial
    cv = np.where(mv > 10, sv / mv * 100, np.nan)
    log.info(f"  Mean CV (std/mean, where mean>10): {np.nanmean(cv):.1f}%")

    # Write outputs
    out_profile = ref_profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=NODATA, compress="deflate",
                       tiled=True, blockxsize=256, blockysize=256, predictor=3)

    mean_path = MAPS_DIR / MEAN_OUT
    std_path = MAPS_DIR / STD_OUT
    with rasterio.open(mean_path, "w", **out_profile) as dst:
        dst.write(mean_map, 1)
    log.info(f"  Wrote {mean_path.name} ({mean_path.stat().st_size / 1e6:.1f} MB)")
    with rasterio.open(std_path, "w", **out_profile) as dst:
        dst.write(std_map, 1)
    log.info(f"  Wrote {std_path.name} ({std_path.stat().st_size / 1e6:.1f} MB)")

    log.info("")
    log.info("Done. Ensemble mean is the published biomass map; std is the uncertainty layer.")


if __name__ == "__main__":
    main()