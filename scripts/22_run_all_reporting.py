"""Phase 3, step 2: orchestrate the 12 reporting runs (4 variants x 3 seeds).

Order is seed-major: all 4 variants for seed 42, then all 4 for seed 7,
then all 4 for seed 123. Degrades gracefully if interrupted - after 4 runs
you have one complete preliminary 4-way comparison.

Each individual run is launched as a subprocess that invokes
scripts/21_train.py via uv. Failures are logged but don't halt the batch.
Already-completed runs (detected by best.pt presence) are skipped to
support resumability.

Usage:
    uv run python scripts/22_run_all_reporting.py

Optional: dry run to see the schedule without actually training:
    uv run python scripts/22_run_all_reporting.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from biomass.config import INTERIM_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)


VARIANTS = ["s2_only", "s1_only", "early", "late"]
SEEDS = [42, 7, 123]
CHECKPOINT_DIR = Path("data/checkpoints")
STATE_FILE = INTERIM_DIR / "reporting_runs_state.json"


def run_name(variant: str, seed: int) -> str:
    """Match the naming used in scripts/21_train.py."""
    name_map = {
        "s2_only": "s2_only",
        "s1_only": "s1_only",
        "early": "early_fusion",
        "late": "late_fusion",
    }
    return f"{name_map[variant]}_seed{seed}"


def is_complete(variant: str, seed: int) -> bool:
    """Detect a finished run by the presence of best.pt."""
    return (CHECKPOINT_DIR / run_name(variant, seed) / "best.pt").exists()


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"runs": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def launch_run(variant: str, seed: int) -> int:
    """Launch a single training run as a subprocess. Returns its exit code.

    Uses `uv run python scripts/21_train.py model=<variant> train.seed=<seed>`
    so the training script sees the same Python environment as the
    orchestrator. Output is streamed live to the orchestrator's stdout/stderr
    so the user can see progress in real time.
    """
    cmd = [
        "uv", "run", "python", "scripts/21_train.py",
        f"model={variant}",
        f"train.seed={seed}",
    ]
    log.info(f"Launching: {' '.join(cmd)}")
    # shell=False is safer; cwd=Path('.') uses the current working directory
    return subprocess.call(cmd, shell=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the schedule without running anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging("INFO")

    # Build the seed-major schedule:
    # (seed=42, s2_only), (seed=42, s1_only), (seed=42, early), (seed=42, late),
    # (seed=7,  s2_only), ... (seed=123, late)
    schedule = [(seed, variant) for seed in SEEDS for variant in VARIANTS]
    assert len(schedule) == 12, "Expected 12 runs (4 variants x 3 seeds)"

    log.info("=" * 70)
    log.info("Phase 3 reporting batch: 12 runs (seed-major order)")
    log.info("=" * 70)
    log.info(f"Schedule:")
    for i, (seed, variant) in enumerate(schedule, start=1):
        log.info(f"  Run {i:2d}/12: variant={variant:10s} seed={seed:3d}  "
                 f"run_name={run_name(variant, seed)}")

    if args.dry_run:
        log.info("\nDry run: exiting without launching.")
        return 0

    state = load_state()
    n_skipped = n_completed = n_failed = 0
    t_start = time.time()

    for i, (seed, variant) in enumerate(schedule, start=1):
        name = run_name(variant, seed)
        log.info("")
        log.info(f"=== [{i:2d}/12] {name} (seed={seed}, variant={variant}) ===")

        # Skip if already complete
        if is_complete(variant, seed):
            log.info(f"  SKIPPED: best.pt already exists at "
                     f"{CHECKPOINT_DIR / name / 'best.pt'}")
            state["runs"][name] = {
                "status": "skipped_already_complete",
                "timestamp": datetime.now().isoformat(),
            }
            save_state(state)
            n_skipped += 1
            continue

        # Launch the training
        state["runs"][name] = {
            "status": "running",
            "started": datetime.now().isoformat(),
        }
        save_state(state)

        t_run = time.time()
        exit_code = launch_run(variant, seed)
        elapsed = time.time() - t_run

        if exit_code == 0:
            log.info(f"  COMPLETED in {elapsed/60:.1f} min")
            state["runs"][name] = {
                "status": "completed",
                "elapsed_min": round(elapsed / 60, 1),
                "timestamp": datetime.now().isoformat(),
            }
            n_completed += 1
        else:
            log.error(f"  FAILED with exit code {exit_code} after "
                      f"{elapsed/60:.1f} min")
            state["runs"][name] = {
                "status": "failed",
                "exit_code": exit_code,
                "elapsed_min": round(elapsed / 60, 1),
                "timestamp": datetime.now().isoformat(),
            }
            n_failed += 1

        save_state(state)

    total_elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 70)
    log.info(f"Reporting batch finished in {total_elapsed/3600:.1f} hours")
    log.info(f"  Completed: {n_completed}")
    log.info(f"  Skipped (already complete): {n_skipped}")
    log.info(f"  Failed:    {n_failed}")
    log.info("=" * 70)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())