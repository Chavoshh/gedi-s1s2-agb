"""Phase 4 step 3: generate paper figures from test predictions and metrics.

Three figures:
  1. predicted_vs_observed.png  - 2x2 hexbin scatter, one panel per variant (seed 42)
  2. rmse_by_agbd_bin.png       - grouped bar chart of per-bin RMSE
  3. rmse_by_landcover.png      - grouped bar chart of per-land-cover RMSE

Outputs to data/processed/figures/.

Usage:
    uv run python scripts/27_make_figures.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from omegaconf import DictConfig

from biomass.config import PROCESSED_DIR
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

# Functional color grouping: cool = single-modality baselines, warm = fusion.
VARIANT_COLORS = {
    "s1_only": "#1f77b4",   # blue   - SAR baseline
    "s2_only": "#2ca02c",   # green  - optical baseline
    "early":   "#ff7f0e",   # orange - simple fusion
    "late":    "#d62728",   # red    - advanced fusion (highlight)
}
VARIANT_ORDER = ["s1_only", "s2_only", "early", "late"]
VARIANT_LABELS = {
    "s1_only": "S1-only",
    "s2_only": "S2-only",
    "early":   "Early fusion",
    "late":    "Late fusion",
}

AGBD_BIN_LABELS = ["0-50", "50-100", "100-150", "150-200", "200-500"]
LC_CATEGORIES = ["tree_cover", "grassland", "cropland", "other"]
LC_LABELS = {
    "tree_cover": "Tree cover",
    "grassland":  "Grassland",
    "cropland":   "Cropland",
    "other":      "Other",
}

FIGURE_DIR = PROCESSED_DIR / "figures"
SCATTER_SEED = 7


def fig_predicted_vs_observed(preds: pd.DataFrame) -> None:
    """2x2 hexbin scatter, one panel per variant. Uses seed 42 only."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # Common axis limits across all four panels for visual comparability.
    lim_max = 500.0
    bins_extent = (0, lim_max, 0, lim_max)

    for ax, variant in zip(axes_flat, VARIANT_ORDER):
        df = preds[(preds.variant == variant) & (preds.seed == SCATTER_SEED)]
        x = df["true_agbd"].values
        y = np.clip(df["pred_agbd"].values, 0, None)

        # Hexbin with log-normalized density so the high-density 0-50 region
        # doesn't dominate the colormap.
        hb = ax.hexbin(
            x, y,
            gridsize=60,
            extent=bins_extent,
            cmap="viridis",
            mincnt=1,
            norm=LogNorm(),
        )

        # 1:1 reference line
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=1, alpha=0.5, label="1:1")

        # Per-panel metrics annotation
        err = y - x
        rmse = float(np.sqrt(np.mean(err ** 2)))
        r2 = 1 - np.sum(err ** 2) / np.sum((x - x.mean()) ** 2)
        bias = float(np.mean(err))
        ax.text(
            0.04, 0.96,
            f"RMSE = {rmse:.1f} Mg/ha\n$R^2$ = {r2:.3f}\nBias = {bias:+.1f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )

        # Title in variant color
        ax.set_title(VARIANT_LABELS[variant], color=VARIANT_COLORS[variant],
                     fontsize=13, fontweight="bold")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect("equal")

    # Shared axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("True AGBD (Mg/ha)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted AGBD (Mg/ha)")

    # Shared colorbar
    cbar = fig.colorbar(hb, ax=axes, orientation="vertical", shrink=0.7,
                        label="Patch count (log scale)")

    fig.suptitle(
        f"Predicted vs. observed AGBD on the test set (seed {SCATTER_SEED}, n = {len(df):,} patches per panel)",
        fontsize=13, y=0.98,
    )

    out_path = FIGURE_DIR / "predicted_vs_observed.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {out_path}")


def fig_rmse_by_agbd_bin(by_bin: pd.DataFrame) -> None:
    """Grouped bar chart: per-bin RMSE for the 4 variants."""
    fig, ax = plt.subplots(figsize=(11, 6))

    n_bins = len(AGBD_BIN_LABELS)
    n_variants = len(VARIANT_ORDER)
    bar_width = 0.20
    group_centers = np.arange(n_bins)
    offsets = (np.arange(n_variants) - (n_variants - 1) / 2) * bar_width

    for i, variant in enumerate(VARIANT_ORDER):
        means = []
        stds = []
        for bin_label in AGBD_BIN_LABELS:
            row = by_bin[(by_bin.variant == variant) & (by_bin.agbd_bin == bin_label)].iloc[0]
            means.append(row["rmse_mean"])
            stds.append(row["rmse_std"])
        positions = group_centers + offsets[i]
        ax.bar(
            positions, means, bar_width,
            yerr=stds, capsize=3,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
            edgecolor="black", linewidth=0.5,
        )

    # Patch counts per bin (use first variant; they're identical across variants)
    first = VARIANT_ORDER[0]
    counts = [
        int(by_bin[(by_bin.variant == first) & (by_bin.agbd_bin == b)].iloc[0]["n_patches"])
        for b in AGBD_BIN_LABELS
    ]
    xtick_labels = [f"{b}\n(n = {c:,})" for b, c in zip(AGBD_BIN_LABELS, counts)]

    ax.set_xticks(group_centers)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("True AGBD bin (Mg/ha)")
    ax.set_ylabel("Test RMSE (Mg/ha)")
    ax.set_title("Test RMSE stratified by true AGBD bin (mean $\\pm$ std across 3 seeds)",
                 fontsize=13)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    out_path = FIGURE_DIR / "rmse_by_agbd_bin.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {out_path}")


def fig_rmse_by_landcover(by_lc: pd.DataFrame) -> None:
    """Grouped bar chart: per-land-cover RMSE for the 4 variants."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_cats = len(LC_CATEGORIES)
    n_variants = len(VARIANT_ORDER)
    bar_width = 0.20
    group_centers = np.arange(n_cats)
    offsets = (np.arange(n_variants) - (n_variants - 1) / 2) * bar_width

    for i, variant in enumerate(VARIANT_ORDER):
        means = []
        stds = []
        for cat in LC_CATEGORIES:
            row = by_lc[(by_lc.variant == variant) & (by_lc.landcover == cat)].iloc[0]
            means.append(row["rmse_mean"])
            stds.append(row["rmse_std"])
        positions = group_centers + offsets[i]
        ax.bar(
            positions, means, bar_width,
            yerr=stds, capsize=3,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
            edgecolor="black", linewidth=0.5,
        )

    # Patch counts per category
    first = VARIANT_ORDER[0]
    counts = [
        int(by_lc[(by_lc.variant == first) & (by_lc.landcover == cat)].iloc[0]["n_patches"])
        for cat in LC_CATEGORIES
    ]
    xtick_labels = [f"{LC_LABELS[c]}\n(n = {n:,})" for c, n in zip(LC_CATEGORIES, counts)]

    ax.set_xticks(group_centers)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Majority land cover class (ESA WorldCover 2021)")
    ax.set_ylabel("Test RMSE (Mg/ha)")
    ax.set_title("Test RMSE stratified by majority land cover (mean $\\pm$ std across 3 seeds)",
                 fontsize=13)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    out_path = FIGURE_DIR / "rmse_by_landcover.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {out_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    log.info("Phase 4 step 3: generating paper figures")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading predictions and metrics tables")
    preds = pd.read_parquet(PROCESSED_DIR / "test_predictions.parquet")
    by_bin = pd.read_csv(PROCESSED_DIR / "test_metrics_by_agbd_bin.csv")
    by_lc = pd.read_csv(PROCESSED_DIR / "test_metrics_by_landcover.csv")

    log.info("Generating figure 1: predicted vs. observed scatter")
    fig_predicted_vs_observed(preds)

    log.info("Generating figure 2: RMSE by AGBD bin")
    fig_rmse_by_agbd_bin(by_bin)

    log.info("Generating figure 3: RMSE by land cover")
    fig_rmse_by_landcover(by_lc)

    log.info("All figures saved to data/processed/figures/")


if __name__ == "__main__":
    main()