"""Phase 4 step 2: compute test-set metrics from saved predictions.

Loads the predictions parquet from script 25 and the land cover parquet from
script 24, computes overall + stratified metrics, and writes:
  - data/processed/test_metrics_overall.csv      (mean +/- std across seeds per variant)
  - data/processed/test_metrics_by_agbd_bin.csv  (per variant x bin)
  - data/processed/test_metrics_by_landcover.csv (per variant x land cover)

Also prints a paper-ready summary to stdout.

Usage:
    uv run python scripts/26_compute_metrics.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from biomass.config import PROCESSED_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

VARIANT_ORDER = ["s1_only", "s2_only", "early", "late"]
VARIANT_LABELS = {
    "s1_only": "S1-only",
    "s2_only": "S2-only",
    "early": "Early fusion",
    "late": "Late fusion",
}

# AGBD stratification bins (Lang 2023). Right-open intervals.
AGBD_BINS = [0, 50, 100, 150, 200, 500]
AGBD_BIN_LABELS = ["0-50", "50-100", "100-150", "150-200", "200-500"]

# Land cover classes to report; everything else aggregated into "other".
LC_REPORT_CLASSES = {
    10: "tree_cover",
    30: "grassland",
    40: "cropland",
}


def compute_metrics(true_y: np.ndarray, pred_y: np.ndarray) -> dict[str, float]:
    """Standard regression metrics. Predictions are clipped at 0 (no negative biomass)."""
    if len(true_y) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan,
                "bias": np.nan, "rel_rmse_pct": np.nan}
    pred_y = np.clip(pred_y, 0.0, None)
    err = pred_y - true_y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true_y - true_y.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mean_truth = float(true_y.mean())
    rel_rmse_pct = float(rmse / mean_truth * 100) if mean_truth > 0 else float("nan")
    return {"n": len(true_y), "rmse": rmse, "mae": mae, "r2": r2,
            "bias": bias, "rel_rmse_pct": rel_rmse_pct}


def aggregate_seed_metrics(per_seed: pd.DataFrame) -> dict[str, float]:
    """Given a DataFrame with one row per seed, return mean +/- std for each metric."""
    out = {}
    for metric in ["rmse", "mae", "r2", "bias", "rel_rmse_pct"]:
        out[f"{metric}_mean"] = float(per_seed[metric].mean())
        out[f"{metric}_std"] = float(per_seed[metric].std(ddof=1)) if len(per_seed) > 1 else 0.0
    out["n_seeds"] = len(per_seed)
    out["n_patches"] = int(per_seed["n"].iloc[0]) if len(per_seed) > 0 else 0
    return out


def compute_overall(preds: pd.DataFrame) -> pd.DataFrame:
    """Overall test metrics: per variant, mean +/- std across 3 seeds."""
    rows = []
    for variant in VARIANT_ORDER:
        per_seed = []
        for seed in sorted(preds[preds.variant == variant]["seed"].unique()):
            df = preds[(preds.variant == variant) & (preds.seed == seed)]
            m = compute_metrics(df["true_agbd"].values, df["pred_agbd"].values)
            m["variant"] = variant
            m["seed"] = seed
            per_seed.append(m)
        per_seed_df = pd.DataFrame(per_seed)
        agg = aggregate_seed_metrics(per_seed_df)
        agg["variant"] = variant
        rows.append(agg)
    return pd.DataFrame(rows)


def compute_stratified_by_agbd(preds: pd.DataFrame) -> pd.DataFrame:
    """Stratified metrics by AGBD bin. Returns one row per (variant, bin)."""
    rows = []
    for variant in VARIANT_ORDER:
        for i in range(len(AGBD_BINS) - 1):
            lo, hi = AGBD_BINS[i], AGBD_BINS[i + 1]
            bin_label = AGBD_BIN_LABELS[i]
            per_seed = []
            for seed in sorted(preds[preds.variant == variant]["seed"].unique()):
                df = preds[(preds.variant == variant) & (preds.seed == seed)]
                in_bin = (df["true_agbd"] >= lo) & (df["true_agbd"] < hi)
                df_bin = df[in_bin]
                m = compute_metrics(df_bin["true_agbd"].values, df_bin["pred_agbd"].values)
                per_seed.append(m)
            per_seed_df = pd.DataFrame(per_seed)
            agg = aggregate_seed_metrics(per_seed_df)
            agg["variant"] = variant
            agg["agbd_bin"] = bin_label
            rows.append(agg)
    return pd.DataFrame(rows)


def compute_stratified_by_landcover(
    preds: pd.DataFrame, lc: pd.DataFrame,
) -> pd.DataFrame:
    """Stratified metrics by land cover. Tree/grass/crop reported; others -> 'other'."""
    # Map each patch_id to a reported land cover category (or 'other')
    lc = lc.copy()
    lc["lc_report"] = lc["lc_class"].map(LC_REPORT_CLASSES).fillna("other")

    # Join predictions to land cover by patch_id
    merged = preds.merge(lc[["patch_id", "lc_report"]], on="patch_id", how="left")

    rows = []
    lc_categories = ["tree_cover", "grassland", "cropland", "other"]
    for variant in VARIANT_ORDER:
        for category in lc_categories:
            per_seed = []
            for seed in sorted(merged[merged.variant == variant]["seed"].unique()):
                df = merged[
                    (merged.variant == variant)
                    & (merged.seed == seed)
                    & (merged.lc_report == category)
                ]
                m = compute_metrics(df["true_agbd"].values, df["pred_agbd"].values)
                per_seed.append(m)
            per_seed_df = pd.DataFrame(per_seed)
            agg = aggregate_seed_metrics(per_seed_df)
            agg["variant"] = variant
            agg["landcover"] = category
            rows.append(agg)
    return pd.DataFrame(rows)


def print_overall_table(df: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 75)
    log.info("OVERALL TEST METRICS (mean +/- std across 3 seeds)")
    log.info("=" * 75)
    log.info(f"{'Variant':<14}  {'RMSE (Mg/ha)':>14}  {'MAE (Mg/ha)':>14}  "
             f"{'R^2':>10}  {'Bias (Mg/ha)':>14}")
    log.info("-" * 75)
    for variant in VARIANT_ORDER:
        row = df[df.variant == variant].iloc[0]
        rmse = f"{row['rmse_mean']:.3f} +/- {row['rmse_std']:.3f}"
        mae = f"{row['mae_mean']:.3f} +/- {row['mae_std']:.3f}"
        r2 = f"{row['r2_mean']:.3f}"
        bias = f"{row['bias_mean']:+.3f} +/- {row['bias_std']:.3f}"
        log.info(f"{VARIANT_LABELS[variant]:<14}  {rmse:>14}  {mae:>14}  "
                 f"{r2:>10}  {bias:>14}")


def print_agbd_bin_table(df: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 90)
    log.info("STRATIFIED METRICS BY AGBD BIN (RMSE Mg/ha, mean +/- std across 3 seeds)")
    log.info("=" * 90)
    header = f"{'Variant':<14}  " + "  ".join(f"{b:>14}" for b in AGBD_BIN_LABELS)
    log.info(header)
    log.info("-" * 90)
    for variant in VARIANT_ORDER:
        cells = []
        for bin_label in AGBD_BIN_LABELS:
            row = df[(df.variant == variant) & (df.agbd_bin == bin_label)].iloc[0]
            cells.append(f"{row['rmse_mean']:.2f} +/- {row['rmse_std']:.2f}")
        log.info(f"{VARIANT_LABELS[variant]:<14}  " + "  ".join(f"{c:>14}" for c in cells))

    # Patch counts per bin (same for all variants since AGBD is independent of variant)
    log.info("")
    counts = []
    first_variant = VARIANT_ORDER[0]
    for bin_label in AGBD_BIN_LABELS:
        row = df[(df.variant == first_variant) & (df.agbd_bin == bin_label)].iloc[0]
        counts.append(f"n={int(row['n_patches']):,}")
    log.info(f"{'Patches':<14}  " + "  ".join(f"{c:>14}" for c in counts))


def print_landcover_table(df: pd.DataFrame) -> None:
    log.info("")
    log.info("=" * 75)
    log.info("STRATIFIED METRICS BY LAND COVER (RMSE Mg/ha, mean +/- std)")
    log.info("=" * 75)
    lc_categories = ["tree_cover", "grassland", "cropland", "other"]
    header = f"{'Variant':<14}  " + "  ".join(f"{c:>14}" for c in lc_categories)
    log.info(header)
    log.info("-" * 75)
    for variant in VARIANT_ORDER:
        cells = []
        for category in lc_categories:
            row = df[(df.variant == variant) & (df.landcover == category)].iloc[0]
            cells.append(f"{row['rmse_mean']:.2f} +/- {row['rmse_std']:.2f}")
        log.info(f"{VARIANT_LABELS[variant]:<14}  " + "  ".join(f"{c:>14}" for c in cells))

    # Patch counts per category
    log.info("")
    counts = []
    for category in lc_categories:
        row = df[(df.variant == VARIANT_ORDER[0]) & (df.landcover == category)].iloc[0]
        counts.append(f"n={int(row['n_patches']):,}")
    log.info(f"{'Patches':<14}  " + "  ".join(f"{c:>14}" for c in counts))


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("Phase 4 step 2: compute test metrics from saved predictions")

    preds_path = PROCESSED_DIR / "test_predictions.parquet"
    lc_path = PROCESSED_DIR / f"patch_landcover_{cfg.aoi.name}.parquet"

    log.info(f"Loading {preds_path}")
    preds = pd.read_parquet(preds_path)
    log.info(f"  {len(preds):,} predictions, {preds['variant'].nunique()} variants, "
             f"{preds['seed'].nunique()} seeds")

    log.info(f"Loading {lc_path}")
    lc = pd.read_parquet(lc_path)
    log.info(f"  {len(lc):,} patch land cover assignments")

    # Compute the three tables
    overall = compute_overall(preds)
    by_bin = compute_stratified_by_agbd(preds)
    by_lc = compute_stratified_by_landcover(preds, lc)

    # Save to CSV
    overall_path = PROCESSED_DIR / "test_metrics_overall.csv"
    by_bin_path = PROCESSED_DIR / "test_metrics_by_agbd_bin.csv"
    by_lc_path = PROCESSED_DIR / "test_metrics_by_landcover.csv"
    overall.to_csv(overall_path, index=False)
    by_bin.to_csv(by_bin_path, index=False)
    by_lc.to_csv(by_lc_path, index=False)
    log.info(f"Wrote {overall_path}")
    log.info(f"Wrote {by_bin_path}")
    log.info(f"Wrote {by_lc_path}")

    # Print paper-ready tables to stdout
    print_overall_table(overall)
    print_agbd_bin_table(by_bin)
    print_landcover_table(by_lc)

    log.info("")
    log.info("=" * 75)
    log.info("Done.")
    log.info("=" * 75)


if __name__ == "__main__":
    main()