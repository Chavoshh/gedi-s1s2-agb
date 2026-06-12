"""Audit script: cross-reference the scene manifest, Hyp3 jobs, and on-disk
RTC outputs. Tells us exactly where each of the 36 scenes stands.
"""
from __future__ import annotations

import json
from pathlib import Path

from hyp3_sdk import HyP3

from biomass.config import INTERIM_DIR, RAW_DIR

MANIFEST_PATH = INTERIM_DIR / "s1_scene_manifest.json"
HYP3_OUT_DIR = RAW_DIR / "s1_hyp3"


def inspect_scene_dir(scene_name: str) -> dict:
    """Inspect on-disk state for one scene."""
    d = HYP3_OUT_DIR / scene_name
    if not d.exists():
        return {"on_disk": False}

    all_files = list(d.rglob("*"))
    tifs = [f for f in all_files if f.suffix == ".tif"]
    zips = [f for f in all_files if f.suffix == ".zip"]
    vv = [f for f in tifs if f.name.endswith("_VV.tif")]
    vh = [f for f in tifs if f.name.endswith("_VH.tif")]
    inc = [f for f in tifs if "inc_map" in f.name.lower()]
    total_mb = sum(f.stat().st_size for f in all_files if f.is_file()) / 1e6

    return {
        "on_disk":  True,
        "n_tifs":   len(tifs),
        "has_vv":   len(vv) > 0,
        "has_vh":   len(vh) > 0,
        "has_inc":  len(inc) > 0,
        "has_zip":  len(zips) > 0,  # zip should be deleted post-unzip
        "total_mb": round(total_mb, 1),
    }


def main():
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)

    # Flatten manifest
    scenes = []
    for year, months in manifest["years"].items():
        for month, scene in months.items():
            scenes.append((year, month, scene["scene_name"]))

    # Fetch all Hyp3 biomass_s1 jobs
    hyp3 = HyP3()
    all_jobs = hyp3.find_jobs()
    biomass_jobs = [j for j in all_jobs
                    if j.name and j.name.startswith("biomass_s1_")]

    # Index Hyp3 jobs by granule
    jobs_by_granule: dict[str, list] = {}
    for j in biomass_jobs:
        granule = j.job_parameters["granules"][0]
        jobs_by_granule.setdefault(granule, []).append(j)

    # Header
    print(f"{'YEAR':<5} {'MO':<3} {'SCENE (last 12)':<14} "
          f"{'HYP3 STATUS':<11} {'ON DISK':<9} {'VV':<4} {'VH':<4} "
          f"{'INC':<5} {'ZIP':<5} {'SIZE_MB':<8}")
    print("-" * 100)

    counts = {
        "on_disk_complete": 0,
        "on_disk_partial":  0,
        "running_hyp3":     0,
        "succeeded_no_dl":  0,
        "failed_hyp3":      0,
        "no_job":           0,
    }

    for year, month, scene_name in scenes:
        # On-disk inspection
        disk = inspect_scene_dir(scene_name)

        # Hyp3 status — use the most recent job per granule
        jobs = jobs_by_granule.get(scene_name, [])
        if jobs:
            jobs.sort(key=lambda j: j.request_time, reverse=True)
            status = jobs[0].status_code
        else:
            status = "NONE"

        # Categorize
        if disk.get("on_disk"):
            if disk["has_vv"] and disk["has_vh"]:
                counts["on_disk_complete"] += 1
            else:
                counts["on_disk_partial"] += 1
        elif status in {"PENDING", "RUNNING"}:
            counts["running_hyp3"] += 1
        elif status == "SUCCEEDED":
            counts["succeeded_no_dl"] += 1
        elif status == "FAILED":
            counts["failed_hyp3"] += 1
        else:
            counts["no_job"] += 1

        short_name = scene_name[-12:]
        print(f"{year:<5} {month:<3} {short_name:<14} "
              f"{status:<11} "
              f"{'YES' if disk.get('on_disk') else 'no':<9} "
              f"{'Y' if disk.get('has_vv') else '-':<4} "
              f"{'Y' if disk.get('has_vh') else '-':<4} "
              f"{'Y' if disk.get('has_inc') else '-':<5} "
              f"{'Y' if disk.get('has_zip') else '-':<5} "
              f"{disk.get('total_mb', 0):<8}")

    print("-" * 100)
    print(f"\nSummary of {len(scenes)} manifest scenes:")
    print(f"  On disk, complete (VV+VH):     {counts['on_disk_complete']}")
    print(f"  On disk, partial:              {counts['on_disk_partial']}")
    print(f"  Running on Hyp3:               {counts['running_hyp3']}")
    print(f"  Succeeded but not downloaded:  {counts['succeeded_no_dl']}")
    print(f"  Failed on Hyp3:                {counts['failed_hyp3']}")
    print(f"  No Hyp3 job submitted yet:     {counts['no_job']}")

    info = hyp3.my_info()
    print(f"\nHyp3 credits remaining: {info.get('remaining_credits')}")


if __name__ == "__main__":
    main()