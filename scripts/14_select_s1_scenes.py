"""Phase 2, step 6 (Hyp3 path): select 12 Sentinel-1 scenes per year over the
dev AOI, stratified by month, for Hyp3 RTC processing.

For each year (2020, 2021, 2022), and each month within that year, picks the
S1 IW GRD scene whose acquisition date is closest to the 15th of that month.
If a month has no scenes, that month is skipped and
logged.

Output: a JSON manifest at data/interim/s1_scene_manifest.json listing the
selected scene names per year/month. No jobs are submitted; no Hyp3 credits
are spent.

Usage:
    uv run python scripts/14_select_s1_scenes.py
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import asf_search as asf
import hydra
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR
from biomass.data.aoi import get_aoi
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

# We need to pick scenes across multiple years.
YEARS = [2020, 2021, 2022]

MANIFEST_PATH = INTERIM_DIR / "s1_scene_manifest.json"


def aoi_to_wkt(aoi) -> str:
    w, s, e, n = aoi.bbox
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def parse_scene_datetime(scene_props: dict) -> datetime:
    """Parse the scene's startTime into a timezone-aware datetime."""
    ts = scene_props["startTime"]  # e.g. "2020-08-27T18:20:21Z"
    # Strip trailing Z and parse as UTC
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def select_one_per_month(scenes: list, year: int) -> dict[int, dict]:
    """For each month 1-12, pick the scene closest to the 15th.

    Returns a dict mapping month_number -> scene dict (with key fields).
    Months with no available scene are omitted.
    """
    selected: dict[int, dict] = {}

    # Bucket all scenes by month for fast lookup
    scenes_by_month: dict[int, list] = {m: [] for m in range(1, 13)}
    for s in scenes:
        dt = parse_scene_datetime(s.properties)
        if dt.year == year:
            scenes_by_month[dt.month].append((dt, s))

    for month in range(1, 13):
        candidates = scenes_by_month[month]
        if not candidates:
            log.warning(f"  {year}-{month:02d}: no scenes available, skipping")
            continue

        target = datetime(year, month, 15, tzinfo=timezone.utc)
        # Pick the scene with minimum |acquisition - target| time
        best_dt, best_scene = min(
            candidates, key=lambda pair: abs((pair[0] - target).total_seconds())
        )
        days_off = abs((best_dt - target).days)

        selected[month] = {
            "scene_name":      best_scene.properties["sceneName"],
            "acquisition":     best_scene.properties["startTime"],
            "orbit_direction": best_scene.properties["flightDirection"],
            "path_number":     best_scene.properties["pathNumber"],
            "frame_number":    best_scene.properties["frameNumber"],
            "polarization":    best_scene.properties["polarization"],
            "size_gb":         best_scene.properties.get("bytes", 0) / 1e9,
            "days_from_15th":  days_off,
        }
        log.info(f"  {year}-{month:02d}: "
                 f"{best_scene.properties['sceneName']}  "
                 f"({best_scene.properties['flightDirection']} "
                 f"path {best_scene.properties['pathNumber']}, "
                 f"±{days_off} days from 15th)")

    return selected


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    aoi = get_aoi(cfg.aoi.name)

    log.info(f"=== Selecting S1 scenes for AOI '{aoi.name}', years {YEARS} ===")
    log.info(f"Bbox: {aoi.bbox}")

    manifest = {
        "aoi_name": aoi.name,
        "aoi_bbox": list(aoi.bbox),
        "selection_strategy": "one scene per month, closest to the 15th",
        "years": {},
    }

    for year in YEARS:
        log.info(f"\nQuerying ASF for {year}...")
        results = asf.geo_search(
            platform=asf.PLATFORM.SENTINEL1,
            processingLevel=asf.PRODUCT_TYPE.GRD_HD,
            beamMode=asf.BEAMMODE.IW,
            intersectsWith=aoi_to_wkt(aoi),
            start=f"{year}-01-01",
            end=f"{year}-12-31",
        )
        log.info(f"  Found {len(results)} total scenes intersecting AOI")

        log.info(f"\nSelecting one scene per month for {year}:")
        selected = select_one_per_month(results, year)
        manifest["years"][str(year)] = selected
        log.info(f"  Total selected for {year}: {len(selected)} scenes")

    # Save manifest
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"\nWrote scene manifest to {MANIFEST_PATH}")

    # Summary
    total_scenes = sum(len(months) for months in manifest["years"].values())
    total_size_gb = sum(
        s["size_gb"]
        for months in manifest["years"].values()
        for s in months.values()
    )
    log.info(f"\n=== Summary ===")
    log.info(f"Total scenes selected: {total_scenes} (target: {len(YEARS) * 12})")
    log.info(f"Combined input scene size: ~{total_size_gb:.1f} GB")
    log.info(f"Hyp3 credit cost when we process: 1 per scene = {total_scenes}")

    # Orbit balance check
    asc_count = 0
    desc_count = 0
    for months in manifest["years"].values():
        for s in months.values():
            if s["orbit_direction"] == "ASCENDING":
                asc_count += 1
            else:
                desc_count += 1
    log.info(f"\nOrbit balance: ASC={asc_count}, DESC={desc_count}")


if __name__ == "__main__":
    main()