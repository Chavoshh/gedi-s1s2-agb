"""Phase 1, step 5: exploratory data analysis of the filtered GEDI parquet.

Produces:
  - data/interim/eda/<aoi>/  -- figures and summary stats
  - console + log file output

Run with the appropriate AOI:
  uv run python scripts/05_eda_gedi.py            # dev
  uv run python scripts/05_eda_gedi.py aoi=full   # later
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from biomass.config import INTERIM_DIR
from biomass.data.aoi import get_aoi
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

# Consistent styling
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"


def fig_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.png"


def plot_agbd_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    """Histograms of AGBD: linear and log scales, plus per-beam-type overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Linear-scale histogram (clipped to readable range)
    clip = 500  # Mg/ha
    axes[0].hist(df["agbd"].clip(upper=clip), bins=60, color="forestgreen",
                 alpha=0.85, edgecolor="black", linewidth=0.3)
    n_over = (df["agbd"] > clip).sum()
    axes[0].set_xlabel("AGBD (Mg/ha)")
    axes[0].set_ylabel("Number of shots")
    axes[0].set_title(f"Linear scale (clipped at {clip}; "
                      f"{n_over} shots > {clip} not shown)")

    # 2. Log-scale histogram on log-x to see the long tail
    axes[1].hist(df["agbd"][df["agbd"] > 0], bins=60, color="forestgreen",
                 alpha=0.85, edgecolor="black", linewidth=0.3)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("AGBD (Mg/ha)")
    axes[1].set_ylabel("Number of shots (log)")
    axes[1].set_title("Log-y, full range")

    # 3. Power vs coverage beams overlaid
    for label, sub in df.groupby("is_power_beam"):
        beam_type = "Power" if label else "Coverage"
        axes[2].hist(sub["agbd"].clip(upper=clip), bins=60, alpha=0.55,
                     label=f"{beam_type} (n={len(sub):,})",
                     edgecolor="black", linewidth=0.3)
    axes[2].set_xlabel("AGBD (Mg/ha)")
    axes[2].set_ylabel("Number of shots")
    axes[2].set_title("By beam type")
    axes[2].legend()

    fig.suptitle("AGBD distribution", fontsize=13, y=1.02)
    plt.savefig(fig_path(out_dir, "01_agbd_distributions"))
    plt.close()


def plot_spatial_map(gdf: gpd.GeoDataFrame, out_dir: Path) -> None:
    """Scatter map of shot locations colored by AGBD.

    Subsamples to 30k points to keep the figure readable and small on disk.
    """
    n_show = min(30_000, len(gdf))
    sample = gdf.sample(n=n_show, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        sample.geometry.x, sample.geometry.y,
        c=sample["agbd"].clip(upper=400),
        s=2, alpha=0.6, cmap="viridis",
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label("AGBD (Mg/ha, clipped at 400)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"GEDI shot locations (n={n_show:,} of {len(gdf):,} shown)")
    ax.set_aspect("equal")
    plt.savefig(fig_path(out_dir, "02_spatial_map"))
    plt.close()


def plot_uncertainty(df: pd.DataFrame, out_dir: Path) -> None:
    """AGBD prediction Standard Error: distribution and SE vs AGBD."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1. SE distribution
    se_clip = df["agbd_se"].quantile(0.99)
    axes[0].hist(df["agbd_se"].clip(upper=se_clip), bins=60,
                 color="firebrick", alpha=0.85,
                 edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("AGBD prediction SE (Mg/ha)")
    axes[0].set_ylabel("Number of shots")
    axes[0].set_title(f"Per-shot Standard Error (clipped at 99th percentile)")

    # 2. SE vs AGBD heatmap (2D histogram)
    sub = df[(df["agbd"] < 400) & (df["agbd_se"] < se_clip)]
    h = axes[1].hist2d(sub["agbd"], sub["agbd_se"], bins=60, cmap="viridis",
                      norm="log")
    plt.colorbar(h[3], ax=axes[1], label="Shots (log scale)")
    axes[1].set_xlabel("AGBD (Mg/ha)")
    axes[1].set_ylabel("AGBD SE (Mg/ha)")
    axes[1].set_title("Per-shot SE vs AGBD")

    fig.suptitle("Label uncertainty", fontsize=13, y=1.02)
    plt.savefig(fig_path(out_dir, "03_uncertainty"))
    plt.close()


def plot_temporal_coverage(df: pd.DataFrame, out_dir: Path) -> None:
    """Shots per month, with growing-season highlighting."""
    monthly = df.set_index("acq_datetime").resample("ME").size()

    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black",
                 linewidth=0.3)

    # Highlight growing-season months (April-September)
    for i, ts in enumerate(monthly.index):
        if 4 <= ts.month <= 9:
            ax.get_children()[i].set_color("forestgreen")

    ax.set_xlabel("Month")
    ax.set_ylabel("Number of shots")
    ax.set_title("Temporal coverage (green = growing season Apr-Sep)")
    ax.set_xticklabels([ts.strftime("%Y-%m") for ts in monthly.index],
                       rotation=45, ha="right")
    plt.savefig(fig_path(out_dir, "04_temporal_coverage"))
    plt.close()


def plot_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    """Sensitivity distribution and its relationship to AGBD."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["sensitivity"], bins=50, color="darkorange",
                 alpha=0.85, edgecolor="black", linewidth=0.3)
    axes[0].axvline(0.95, color="red", linestyle="--",
                    label="Filter threshold (0.95)")
    axes[0].axvline(0.98, color="red", linestyle=":",
                    label="Stricter alternative (0.98)")
    axes[0].set_xlabel("Sensitivity")
    axes[0].set_ylabel("Number of shots")
    axes[0].set_title("Sensitivity distribution (post-filter)")
    axes[0].legend()

    # Sensitivity vs AGBD — does high biomass correlate with low sensitivity?
    sub = df[df["agbd"] < 400]
    h = axes[1].hist2d(sub["sensitivity"], sub["agbd"], bins=50, cmap="viridis",
                      norm="log")
    plt.colorbar(h[3], ax=axes[1], label="Shots (log)")
    axes[1].set_xlabel("Sensitivity")
    axes[1].set_ylabel("AGBD (Mg/ha)")
    axes[1].set_title("Sensitivity vs AGBD")

    fig.suptitle("GEDI sensitivity diagnostics", fontsize=13, y=1.02)
    plt.savefig(fig_path(out_dir, "05_sensitivity"))
    plt.close()


def write_summary_stats(df: pd.DataFrame, out_dir: Path) -> None:
    """Compute and save numeric summary statistics."""
    summary_path = out_dir / "summary_stats.txt"
    with summary_path.open("w") as f:
        f.write(f"=== GEDI L4A filtered shots — summary statistics ===\n\n")
        f.write(f"Total shots: {len(df):,}\n\n")

        f.write(f"--- AGBD (Mg/ha) ---\n")
        f.write(df["agbd"].describe().to_string() + "\n\n")
        for q in [0.90, 0.95, 0.99, 0.995, 0.999]:
            f.write(f"  q{q:.3f}: {df['agbd'].quantile(q):>10.2f}\n")
        f.write(f"  shots with AGBD > 500:  {(df['agbd'] > 500).sum():>7,d}\n")
        f.write(f"  shots with AGBD > 1000: {(df['agbd'] > 1000).sum():>7,d}\n\n")

        f.write(f"--- AGBD SE (Mg/ha) ---\n")
        f.write(df["agbd_se"].describe().to_string() + "\n\n")
        f.write(f"  median SE / median AGBD: "
                f"{df['agbd_se'].median() / df['agbd'].median():.3f}\n\n")

        f.write(f"--- Beam type ---\n")
        f.write(df["is_power_beam"].value_counts().to_string() + "\n\n")

        f.write(f"--- Per-beam shot counts ---\n")
        f.write(df["beam"].value_counts().sort_index().to_string() + "\n\n")

        f.write(f"--- Year ---\n")
        f.write(df["acq_datetime"].dt.year.value_counts()
                                          .sort_index().to_string() + "\n\n")

        f.write(f"--- Season (Apr-Sep = growing) ---\n")
        f.write((df["acq_datetime"].dt.month.between(4, 9)
                ).value_counts().to_string() + "\n\n")

        f.write(f"--- PFT class ---\n")
        f.write(df["pft_class"].value_counts().sort_index().to_string() + "\n")

    log.info(f"Wrote {summary_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    aoi = get_aoi(cfg.aoi.name)

    out_dir = INTERIM_DIR / "eda" / aoi.name
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = INTERIM_DIR / "logs" / f"05_eda_{aoi.name}.log"
    setup_logging(cfg.log_level, log_file=log_file)

    parquet_path = INTERIM_DIR / f"gedi_shots_{aoi.name}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Run extraction first: {parquet_path} missing")

    log.info(f"Loading {parquet_path}")
    gdf = gpd.read_parquet(parquet_path)
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    log.info(f"Loaded {len(df):,} shots")

    log.info("Plotting AGBD distributions...")
    plot_agbd_distributions(df, out_dir)

    log.info("Plotting spatial map...")
    plot_spatial_map(gdf, out_dir)

    log.info("Plotting uncertainty diagnostics...")
    plot_uncertainty(df, out_dir)

    log.info("Plotting temporal coverage...")
    plot_temporal_coverage(df, out_dir)

    log.info("Plotting sensitivity diagnostics...")
    plot_sensitivity(df, out_dir)

    log.info("Writing summary statistics...")
    write_summary_stats(df, out_dir)

    log.info(f"\nAll outputs in: {out_dir}")
    log.info("Done.")


if __name__ == "__main__":
    main()