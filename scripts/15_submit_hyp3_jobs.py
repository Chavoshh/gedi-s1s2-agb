"""Phase 2, step 7 (Hyp3 path): submit Hyp3 RTC jobs for all scenes listed in
the manifest, wait for completion, download and unzip each result.

Resumable in two ways:
  1. Skips any scene that already has a downloaded RTC output directory.
  2. Adopts any in-flight Hyp3 jobs from previous runs (matched by name).
"""
from __future__ import annotations

import json
import logging
import time
import zipfile
from pathlib import Path

import hydra
from hyp3_sdk import HyP3, Job
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR, RAW_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

MANIFEST_PATH = INTERIM_DIR / "s1_scene_manifest.json"
HYP3_OUT_DIR = RAW_DIR / "s1_hyp3"

MAX_CONCURRENT = 5
PROBE_S = 60

RTC_PARAMS = dict(
    resolution=30,
    radiometry="gamma0",
    scale="power",
    speckle_filter=False,
    include_dem=False,
    include_inc_map=True,
    include_scattering_area=False,
    include_rgb=False,
)


def load_scenes_from_manifest() -> list[tuple[str, dict]]:
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)
    flat = []
    for year, months in manifest["years"].items():
        for month, scene in months.items():
            flat.append((scene["scene_name"], {
                "year": year, "month": month, **scene,
            }))
    return flat


def already_processed(scene_name: str) -> bool:
    """True if we have an extracted Hyp3 output folder for this scene."""
    scene_dir = HYP3_OUT_DIR / scene_name
    if not scene_dir.exists():
        return False
    vv_tifs = list(scene_dir.rglob("*_VV.tif"))
    return len(vv_tifs) > 0


def adopt_inflight_jobs(hyp3: HyP3) -> tuple[list[Job], set[str]]:
    """Find any biomass_s1_* jobs in flight from previous runs.

    Returns (jobs_to_track, set_of_scene_names_already_submitted).
    """
    all_recent = hyp3.find_jobs()
    adopted: list[Job] = []
    submitted_scenes: set[str] = set()

    for j in all_recent:
        # Only adopt biomass S1 jobs from this project (skip smoke tests, etc.)
        if not j.name or not j.name.startswith("biomass_s1_"):
            continue
        if j.status_code in {"PENDING", "RUNNING"}:
            scene_name = j.job_parameters["granules"][0]
            adopted.append(j)
            submitted_scenes.add(scene_name)
            log.info(f"  adopted: {j.job_id[:8]} {j.status_code:>9} "
                     f"name={j.name} granule={scene_name}")
        elif j.status_code == "SUCCEEDED":
            scene_name = j.job_parameters["granules"][0]
            if not already_processed(scene_name):
                # The job succeeded on Hyp3 but we haven't downloaded it yet.
                adopted.append(j)
                submitted_scenes.add(scene_name)
                log.info(f"  adopted (will download): {j.job_id[:8]} "
                         f"SUCCEEDED name={j.name} granule={scene_name}")

    return adopted, submitted_scenes


def submit_one(hyp3: HyP3, scene_name: str, job_name: str) -> Job:
    batch = hyp3.submit_rtc_job(
        granule=scene_name, name=job_name, **RTC_PARAMS,
    )
    return batch[0]


def download_and_extract(job: Job, out_root: Path) -> None:
    scene_name = job.job_parameters["granules"][0]
    target_dir = out_root / scene_name
    target_dir.mkdir(parents=True, exist_ok=True)

    files = job.download_files(target_dir, create=True)
    log.info(f"  {scene_name}: downloaded {len(files)} file(s)")

    zip_files = [f for f in files if f.suffix == ".zip"]
    for zf in zip_files:
        log.info(f"  Unzipping {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(target_dir)
        zf.unlink()
        log.info(f"  Removed {zf.name}")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    HYP3_OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = INTERIM_DIR / "logs" / "15_hyp3_submit.log"
    setup_logging(cfg.log_level, log_file=log_file)

    log.info("=== Hyp3 batch submission and download ===")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. "
            f"Run scripts/14_select_s1_scenes.py first."
        )

    scenes = load_scenes_from_manifest()
    log.info(f"Total scenes in manifest: {len(scenes)}")

    # Authenticate
    log.info("\nAuthenticating to Hyp3...")
    hyp3 = HyP3()
    info = hyp3.my_info()
    log.info(f"User: {info.get('user_id')}, "
             f"remaining credits: {info.get('remaining_credits')}")

    # Adopt any in-flight jobs from previous interrupted runs
    log.info("\nChecking for in-flight Hyp3 jobs from previous runs...")
    pending, already_submitted = adopt_inflight_jobs(hyp3)
    log.info(f"Adopted {len(pending)} in-flight job(s)")

    # Determine which scenes still need submitting
    queue: list[tuple[str, dict]] = []
    skipped_done = 0
    for name, meta in scenes:
        if already_processed(name):
            skipped_done += 1
            continue
        if name in already_submitted:
            continue  # already in flight, in `pending`
        queue.append((name, meta))

    log.info(f"\nAlready downloaded (skip):     {skipped_done}")
    log.info(f"Already in flight (track):     {len(pending)}")
    log.info(f"To submit this run:            {len(queue)}")

    if not queue and not pending:
        log.info("Nothing to do. Exiting.")
        return

    completed = 0
    failed = 0
    t0 = time.time()

    while queue or pending:
        # Fill the pending pipeline up to MAX_CONCURRENT
        while queue and len(pending) < MAX_CONCURRENT:
            scene_name, meta = queue.pop(0)
            job_name = f"biomass_s1_{meta['year']}_{meta['month']}"
            log.info(f"\n[submit] {meta['year']}-{meta['month']}: {scene_name}")
            try:
                job = submit_one(hyp3, scene_name, job_name)
                pending.append(job)
                log.info(f"  Job ID: {job.job_id}")
            except Exception as e:
                log.error(f"  submit failed: {e}")
                failed += 1

        # Poll all pending jobs
        time.sleep(PROBE_S)
        log.info(f"\n[poll] pending={len(pending)}, "
                 f"queued={len(queue)}, "
                 f"completed={completed}, "
                 f"failed={failed}, "
                 f"elapsed={(time.time()-t0)/60:.1f} min")

        # Refresh all pending job statuses in one call
        pending = [hyp3.refresh(j) for j in pending]
        still_pending: list[Job] = []
        for j in pending:
            status = j.status_code
            scene_name = j.job_parameters["granules"][0]
            if status == "SUCCEEDED":
                log.info(f"  [done]   {scene_name}")
                try:
                    download_and_extract(j, HYP3_OUT_DIR)
                    completed += 1
                except Exception as e:
                    log.error(f"  download/extract failed for {scene_name}: {e}")
                    failed += 1
            elif status == "FAILED":
                log.warning(f"  [FAILED] {scene_name}")
                failed += 1
            else:
                still_pending.append(j)
        pending = still_pending

    log.info(f"\n=== Done ===")
    log.info(f"Successfully processed:    {completed}")
    log.info(f"Failed:                    {failed}")
    log.info(f"Total wall time:           {(time.time()-t0)/60:.1f} min")

    info = hyp3.my_info()
    log.info(f"Hyp3 credits remaining:    {info.get('remaining_credits')}")


if __name__ == "__main__":
    main()