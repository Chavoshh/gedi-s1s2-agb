"""Phase 3 reporting batch monitoring.

Reads:
  - W&B runs in the gedi-s1s2-agb project to get val/rmse values
  - Local state file (data/interim/reporting_runs_state.json) for orchestration status
  - Local checkpoint directory presence

Usage:
    uv run python _reporting_status.py
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb

PROJECT = "chavosh-personal/gedi-s1s2-agb"
STATE_FILE = Path("data/interim/reporting_runs_state.json")
CHECKPOINT_DIR = Path("data/checkpoints")

VARIANTS = ["s2_only", "s1_only", "early", "late"]
SEEDS = [42, 7, 123]

NAME_MAP = {
    "s2_only": "s2_only",
    "s1_only": "s1_only",
    "early": "early_fusion",
    "late": "late_fusion",
}


def expected_run_name(variant: str, seed: int) -> str:
    return f"{NAME_MAP[variant]}_seed{seed}"


def load_orchestration_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f).get("runs", {})
    return {}


def fetch_wandb_results() -> dict[str, dict]:
    """Get the latest sweep_run.summary for each unique run name.

    If multiple runs share a name (multiple training restarts), keeps the
    most recent one by created_at.
    """
    api = wandb.Api()
    runs = api.runs(PROJECT)
    by_name: dict[str, dict] = {}
    for run in runs:
        if run.sweep is not None:
            continue  # skip sweep runs
        name = run.name
        created = run.created_at
        if name not in by_name or created > by_name[name]["created"]:
            by_name[name] = {
                "created": created,
                "state": run.state,
                "best_val_rmse": run.summary.get("best_val_rmse"),
                "best_epoch": run.summary.get("best_epoch"),
                "val_r2": run.summary.get("val/r2"),
                "url": run.url,
            }
    return by_name


def main() -> None:
    orch_state = load_orchestration_state()
    wandb_results = fetch_wandb_results()

    print(f"Phase 3 reporting batch status")
    print("=" * 80)
    print()
    print(f"{'Run #':>5}  {'Run name':22s}  {'Orch':18s}  "
          f"{'W&B':12s}  {'val/rmse':>10s}  {'best_epoch':>10s}")
    print("-" * 80)

    n_done = n_running = n_failed = n_pending = 0
    rmse_by_variant: dict[str, list[float]] = {v: [] for v in VARIANTS}

    for i, (seed, variant) in enumerate(
        [(s, v) for s in SEEDS for v in VARIANTS], start=1
    ):
        name = expected_run_name(variant, seed)
        ckpt_path = CHECKPOINT_DIR / name / "best.pt"
        ckpt_exists = ckpt_path.exists()

        orch_status = orch_state.get(name, {}).get("status", "pending")
        wandb_info = wandb_results.get(name, {})
        wandb_state = wandb_info.get("state", "-")
        val_rmse = wandb_info.get("best_val_rmse")
        best_epoch = wandb_info.get("best_epoch")

        rmse_str = f"{val_rmse:.3f}" if isinstance(val_rmse, (int, float)) else "-"
        epoch_str = f"{best_epoch}" if isinstance(best_epoch, int) else "-"

        if isinstance(val_rmse, (int, float)):
            rmse_by_variant[variant].append(val_rmse)
            n_done += 1
        elif orch_status == "running" or wandb_state == "running":
            n_running += 1
        elif orch_status == "failed":
            n_failed += 1
        else:
            n_pending += 1

        print(f"{i:5d}  {name:22s}  {orch_status:18s}  "
              f"{wandb_state:12s}  {rmse_str:>10s}  {epoch_str:>10s}")

    print()
    print(f"Done: {n_done}  Running: {n_running}  Failed: {n_failed}  "
          f"Pending: {n_pending}  Total: 12")

    print()
    print("Per-variant val/rmse summary (mean, std across completed seeds):")
    print("-" * 50)
    for variant in VARIANTS:
        rmses = rmse_by_variant[variant]
        if not rmses:
            print(f"  {variant:10s}  no completed runs yet")
        elif len(rmses) == 1:
            print(f"  {variant:10s}  {rmses[0]:.3f}  (1 seed)")
        else:
            import statistics
            mean = statistics.mean(rmses)
            stdev = statistics.stdev(rmses) if len(rmses) > 1 else 0
            print(f"  {variant:10s}  {mean:.3f} +/- {stdev:.3f}  "
                  f"({len(rmses)} seeds)")


if __name__ == "__main__":
    main()