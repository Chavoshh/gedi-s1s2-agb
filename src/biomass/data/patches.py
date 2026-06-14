"""Patch extraction helpers for GEDI-supervised biomass modeling.

Functions here are pure (no I/O side effects on global state) so they can be
unit-tested. The CLI driver in scripts/20_extract_patches.py composes them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window

log = logging.getLogger(__name__)


# Channel definitions — the canonical band order used throughout the project.
CHANNEL_NAMES: list[str] = [
    "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12",
    "VV_dB", "VH_dB", "LIA_deg",
    "elevation_m", "slope_deg",
]
N_CHANNELS = len(CHANNEL_NAMES)

# Per-source slicing of the channel array
S2_BAND_IDX = slice(0, 10)
S1_BAND_IDX = slice(10, 13)
DEM_BAND_IDX = slice(13, 15)


@dataclass
class FilterCounts:
    """Bookkeeping for how many shots remain after each filter step."""
    raw: int = 0
    after_year_filter: int = 0
    after_agbd_cap: int = 0
    after_spatial_subsample: int = 0
    after_edge_drop: int = 0


def filter_shots_for_extraction(
    shots: pd.DataFrame,
    valid_years: set[int],
    agbd_cap_mgha: float,
) -> tuple[pd.DataFrame, FilterCounts]:
    """Apply year and AGBD filters. Returns (filtered_df, counts).

    Required input columns: agbd, acquisition_year (or similar — see below).
    The acquisition year is extracted from delta_time if needed.
    """
    counts = FilterCounts(raw=len(shots))

    # Extract acquisition year. GEDI L4A parquet has 'delta_time' (seconds since
    # 2018-01-01) but during EDA we should already have an `acquisition_year`
    # column. If not, fall back to 'year' or derive it.
    if "acq_datetime" in shots.columns:
        year_col = shots["acq_datetime"].dt.year
    elif "acquisition_year" in shots.columns:
        year_col = shots["acquisition_year"]
    elif "year" in shots.columns:
        year_col = shots["year"]
    else:
        # Last resort: derive from delta_time (seconds since 2018-01-01 00:00:00)
        epoch = pd.Timestamp("2018-01-01", tz="UTC")
        dts = epoch + pd.to_timedelta(shots["delta_time"], unit="s")
        year_col = dts.dt.year

    shots = shots.assign(_year=year_col.astype(int))
    shots = shots[shots["_year"].isin(valid_years)].reset_index(drop=True)
    counts.after_year_filter = len(shots)

    shots = shots[shots["agbd"] <= agbd_cap_mgha].reset_index(drop=True)
    counts.after_agbd_cap = len(shots)

    return shots, counts


def spatial_subsample_grid(
    shots: pd.DataFrame,
    cell_size_m: float,
    easting_col: str,
    northing_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Keep at most one shot per cell_size_m x cell_size_m UTM grid cell.

    When multiple shots fall in the same cell, picks one uniformly at random.
    The grid is anchored at (0, 0) UTM origin, which is consistent across runs
    as long as `rng` is seeded.
    """
    cell_x = (shots[easting_col].values // cell_size_m).astype(np.int64)
    cell_y = (shots[northing_col].values // cell_size_m).astype(np.int64)
    df = shots.assign(_cell_x=cell_x, _cell_y=cell_y)

    # For each (cell_x, cell_y) group, sample one shot uniformly
    n_before = len(df)
    df = df.sample(frac=1.0, random_state=rng.bit_generator.random_raw() % (2**32))
    df = df.drop_duplicates(subset=["_cell_x", "_cell_y"], keep="first")
    df = df.drop(columns=["_cell_x", "_cell_y"]).reset_index(drop=True)
    log.info(f"  spatial subsample {cell_size_m:.0f} m: "
             f"{n_before:,} -> {len(df):,} shots")
    return df


def assign_to_blocks(
    shots: pd.DataFrame,
    block_size_m: float,
    easting_col: str,
    northing_col: str,
) -> np.ndarray:
    """Assign each shot to a spatial block ID (int).

    Block IDs are constructed from the block's (col, row) in a UTM grid
    anchored at (0, 0). A block ID is `col * BLOCK_ID_SCALE + row`, where
    BLOCK_ID_SCALE is chosen large enough that there's no collision.
    """
    BLOCK_ID_SCALE = 100_000
    col = (shots[easting_col].values // block_size_m).astype(np.int64)
    row = (shots[northing_col].values // block_size_m).astype(np.int64)
    return col * BLOCK_ID_SCALE + row


def stratified_block_split(
    block_ids: np.ndarray,
    labels: np.ndarray,
    train_frac: float,
    val_frac: float,
    n_strata: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign each shot to train/val/test (0/1/2) by spatial block, with
    blocks stratified by their mean AGBD into n_strata strata.

    Within each stratum, blocks are split proportionally into train/val/test.
    All shots in a block go to the same split.
    """
    test_frac = 1.0 - train_frac - val_frac
    assert test_frac > 0, "train + val must be < 1.0"

    df = pd.DataFrame({"block": block_ids, "label": labels})
    block_means = df.groupby("block")["label"].mean()

    # Stratify blocks by AGBD percentile
    quantile_edges = np.quantile(
        block_means.values,
        np.linspace(0, 1, n_strata + 1),
    )
    quantile_edges[0] -= 1e-6  # ensure leftmost block is included
    block_strata = np.digitize(block_means.values, quantile_edges, right=True) - 1
    block_strata = np.clip(block_strata, 0, n_strata - 1)

    block_to_split: dict[int, int] = {}
    for stratum in range(n_strata):
        blocks_in_stratum = block_means.index[block_strata == stratum].values
        perm = rng.permutation(len(blocks_in_stratum))
        blocks_in_stratum = blocks_in_stratum[perm]
        n = len(blocks_in_stratum)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        for b in blocks_in_stratum[:n_train]:
            block_to_split[int(b)] = 0
        for b in blocks_in_stratum[n_train:n_train + n_val]:
            block_to_split[int(b)] = 1
        for b in blocks_in_stratum[n_train + n_val:]:
            block_to_split[int(b)] = 2

    split = np.array([block_to_split[int(b)] for b in block_ids], dtype=np.int8)
    return split


def world_to_grid(
    eastings: np.ndarray,
    northings: np.ndarray,
    ref_transform,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert UTM (east, north) coordinates to (row, col) on the raster grid."""
    rows, cols = rowcol(ref_transform, eastings, northings)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def read_patch(
    src: rasterio.io.DatasetReader,
    row_center: int,
    col_center: int,
    half_size: int,
) -> np.ndarray:
    """Read a (2*half_size+1) x (2*half_size+1) patch centered on (row, col).

    Returns the patch as (n_bands, H, W) float32. Pixels outside the raster
    are returned as NaN.
    """
    row_off = row_center - half_size
    col_off = col_center - half_size
    win_size = 2 * half_size + 1
    window = Window(col_off=col_off, row_off=row_off,
                    width=win_size, height=win_size)
    data = src.read(window=window, boundless=True, fill_value=np.nan,
                    out_dtype=np.float32)
    return data