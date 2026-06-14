"""Phase 2, step 10: extract 25x25 patches around GEDI shots from the
raster composites (S2, S1, DEM) and write them to a Zarr store with
labels, metadata, and a train/val/test split.

Usage:
    uv run python scripts/20_extract_patches.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import rasterio
import zarr
from omegaconf import DictConfig
from pyproj import Transformer

from biomass.config import INTERIM_DIR, PROCESSED_DIR
from biomass.data.aoi import get_aoi
from biomass.data.patches import (
    CHANNEL_NAMES, DEM_BAND_IDX, N_CHANNELS, S1_BAND_IDX, S2_BAND_IDX,
    assign_to_blocks, filter_shots_for_extraction, read_patch,
    spatial_subsample_grid, stratified_block_split, world_to_grid,
)
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

# ---- Configuration constants ----
PATCH_HALF = 12          # 25x25 patch (2 * 12 + 1)
PATCH_SIZE = 2 * PATCH_HALF + 1
VALID_YEARS = {2020, 2021, 2022}
AGBD_CAP = 500.0         # Mg/ha, per decision log 2026-06-09
SUBSAMPLE_CELL_M = 100.0
BLOCK_SIZE_M = 10_000.0  # 10 km blocks for train/val/test
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
N_STRATA = 5             # for AGBD-stratified split
ZARR_CHUNK_SIZE = 100    # patches per Zarr chunk
RNG_SEED = 42


def load_gedi_shots(parquet_path: Path) -> pd.DataFrame:
    """Read the GEDI parquet and return a DataFrame ready for filtering.

    Required columns produced or passed through:
      shot_id, lon_lowestmode, lat_lowestmode, agbd, acquisition_year (or derivable)
    """
    log.info(f"Reading GEDI shots from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    log.info(f"  loaded {len(df):,} shots, columns: {list(df.columns)}")
    return df


def add_utm_coords(
    shots: pd.DataFrame,
    target_crs,
    lon_col: str = "lon_lowestmode",
    lat_col: str = "lat_lowestmode",
) -> pd.DataFrame:
    """Add 'easting' and 'northing' columns by reprojecting lon/lat to target CRS."""
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    east, north = transformer.transform(shots[lon_col].values,
                                        shots[lat_col].values)
    return shots.assign(easting=east, northing=north)


def open_raster_set(aoi_name: str) -> dict[tuple[str, int], rasterio.io.DatasetReader]:
    """Open all 7 rasters; keys are ('S2', year), ('S1', year), ('DEM', 0)."""
    srcs: dict[tuple[str, int], rasterio.io.DatasetReader] = {}
    for year in VALID_YEARS:
        srcs[("S2", year)] = rasterio.open(
            PROCESSED_DIR / f"s2_composite_{aoi_name}_{year}.tif")
        srcs[("S1", year)] = rasterio.open(
            PROCESSED_DIR / f"s1_composite_{aoi_name}_{year}.tif")
    srcs[("DEM", 0)] = rasterio.open(PROCESSED_DIR / f"dem_{aoi_name}.tif")
    return srcs


def close_raster_set(srcs: dict) -> None:
    for s in srcs.values():
        s.close()


def patch_has_nodata(patch: np.ndarray, s2_nodata_int: int = -32768) -> bool:
    """Return True if the patch contains any no-data pixel.

    S2 channels use -32768 as int16 no-data; after float conversion these
    appear as -32768.0. S1 and DEM use NaN.
    """
    s2 = patch[S2_BAND_IDX]
    if np.any(s2 == s2_nodata_int):
        return True
    if np.any(np.isnan(patch)):
        return True
    return False


def compute_normalization_stats(
    zarr_root: zarr.Group,
    train_mask: np.ndarray,
) -> dict:
    """Compute per-channel mean and std over training patches, streamed."""
    log.info(f"\nComputing normalization stats over "
             f"{train_mask.sum():,} training patches...")
    patches = zarr_root["patches"]
    n_train = int(train_mask.sum())

    # Streaming Welford's algorithm by chunks
    counts = np.zeros(N_CHANNELS, dtype=np.int64)
    means = np.zeros(N_CHANNELS, dtype=np.float64)
    m2s = np.zeros(N_CHANNELS, dtype=np.float64)

    train_indices = np.where(train_mask)[0]
    batch = 1000
    for i0 in range(0, len(train_indices), batch):
        idx = train_indices[i0:i0 + batch]
        chunk = patches.oindex[idx]  # (B, C, H, W)
        for c in range(N_CHANNELS):
            vals = chunk[:, c].ravel().astype(np.float64)
            vals = vals[~np.isnan(vals) & (vals != -32768)]
            n = vals.size
            if n == 0:
                continue
            old_count = counts[c]
            counts[c] = old_count + n
            delta = vals.mean() - means[c]
            means[c] += delta * n / counts[c]
            m2s[c] += vals.var(ddof=0) * n + delta**2 * old_count * n / counts[c]

    stds = np.sqrt(m2s / np.maximum(counts, 1))
    return {
        name: {"mean": float(means[i]), "std": float(stds[i]),
               "n_pixels": int(counts[i])}
        for i, name in enumerate(CHANNEL_NAMES)
    }


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    aoi = get_aoi(cfg.aoi.name)
    log_file = INTERIM_DIR / "logs" / f"20_extract_patches_{aoi.name}.log"
    setup_logging(cfg.log_level, log_file=log_file)

    zarr_path = PROCESSED_DIR / f"patches_{aoi.name}.zarr"
    meta_path = PROCESSED_DIR / f"patches_{aoi.name}_meta.json"

    if zarr_path.exists():
        log.warning(f"{zarr_path} already exists. Delete manually to rebuild.")
        return

    rng = np.random.default_rng(RNG_SEED)
    log.info(f"=== Patch extraction for AOI={aoi.name} ===")
    log.info(f"RNG seed: {RNG_SEED}")
    t_start = time.time()

    # ---- 1. Load and filter GEDI shots ----
    parquet_path = INTERIM_DIR / f"gedi_shots_{aoi.name}.parquet"
    shots = load_gedi_shots(parquet_path)

    log.info("\nApplying year and AGBD filters...")
    shots, counts = filter_shots_for_extraction(
        shots, valid_years=VALID_YEARS, agbd_cap_mgha=AGBD_CAP,
    )
    log.info(f"  raw:                  {counts.raw:,}")
    log.info(f"  after year filter:    {counts.after_year_filter:,}")
    log.info(f"  after AGBD cap:       {counts.after_agbd_cap:,}")

    # ---- 2. Reproject to UTM and spatial-subsample ----
    log.info("\nReprojecting lon/lat to S2 grid CRS...")
    ref_s2 = rasterio.open(
        PROCESSED_DIR / f"s2_composite_{aoi.name}_2020.tif")
    ref_crs = ref_s2.crs
    ref_transform = ref_s2.transform
    ref_height = ref_s2.height
    ref_width = ref_s2.width
    ref_s2.close()

    shots = add_utm_coords(shots, ref_crs)
    log.info(f"  reprojected to {ref_crs}")

    log.info(f"\nSpatial subsampling at {SUBSAMPLE_CELL_M:.0f} m grid...")
    shots = spatial_subsample_grid(
        shots, cell_size_m=SUBSAMPLE_CELL_M,
        easting_col="easting", northing_col="northing", rng=rng,
    )
    counts.after_spatial_subsample = len(shots)

    # ---- 3. Assign to spatial blocks for train/val/test ----
    log.info(f"\nAssigning shots to {BLOCK_SIZE_M:.0f} m blocks...")
    block_ids = assign_to_blocks(
        shots, block_size_m=BLOCK_SIZE_M,
        easting_col="easting", northing_col="northing",
    )
    n_blocks = len(np.unique(block_ids))
    log.info(f"  {n_blocks:,} unique blocks")

    log.info(f"\nStratified train/val/test split "
             f"({TRAIN_FRAC*100:.0f}/{VAL_FRAC*100:.0f}/"
             f"{(1-TRAIN_FRAC-VAL_FRAC)*100:.0f}, {N_STRATA} strata)...")
    splits = stratified_block_split(
        block_ids=block_ids,
        labels=shots["agbd"].values,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        n_strata=N_STRATA,
        rng=rng,
    )
    n_train = int((splits == 0).sum())
    n_val = int((splits == 1).sum())
    n_test = int((splits == 2).sum())
    log.info(f"  train: {n_train:,}, val: {n_val:,}, test: {n_test:,}")

    # ---- 4. Convert UTM to (row, col) on the S2 grid ----
    rows, cols = world_to_grid(
        shots["easting"].values, shots["northing"].values, ref_transform,
    )

    # Pre-screen: drop shots whose patch would be partially outside the grid
    in_bounds = (
        (rows >= PATCH_HALF) & (rows < ref_height - PATCH_HALF) &
        (cols >= PATCH_HALF) & (cols < ref_width - PATCH_HALF)
    )
    n_dropped_bounds = int((~in_bounds).sum())
    log.info(f"\nIn-bounds check: dropped {n_dropped_bounds:,} shots near "
             f"grid edges")

    shots = shots.loc[in_bounds].reset_index(drop=True)
    block_ids = block_ids[in_bounds]
    splits = splits[in_bounds]
    rows = rows[in_bounds]
    cols = cols[in_bounds]
    n_candidates = len(shots)
    log.info(f"  remaining candidates for extraction: {n_candidates:,}")

    # ---- 5. Open all rasters ----
    log.info("\nOpening rasters...")
    srcs = open_raster_set(aoi.name)
    for k, s in srcs.items():
        log.info(f"  {k}: shape {s.shape}, CRS {s.crs}")

    # ---- 6. Initialize the Zarr store ----
    log.info(f"\nCreating Zarr store at {zarr_path}...")
    root = zarr.open_group(zarr_path, mode="w")
    patches_arr = root.create_dataset(
        "patches",
        shape=(n_candidates, N_CHANNELS, PATCH_SIZE, PATCH_SIZE),
        chunks=(ZARR_CHUNK_SIZE, N_CHANNELS, PATCH_SIZE, PATCH_SIZE),
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
    )
    labels_arr = root.create_dataset(
        "labels", shape=(n_candidates,), dtype="float32")
    shot_ids_arr = root.create_dataset(
        "shot_ids", shape=(n_candidates,), dtype="int64")
    years_arr = root.create_dataset(
        "years", shape=(n_candidates,), dtype="int16")
    east_arr = root.create_dataset(
        "eastings", shape=(n_candidates,), dtype="float64")
    north_arr = root.create_dataset(
        "northings", shape=(n_candidates,), dtype="float64")
    block_arr = root.create_dataset(
        "block_ids", shape=(n_candidates,), dtype="int64")
    split_arr = root.create_dataset(
        "splits", shape=(n_candidates,), dtype="int8")

    # ---- 7. Patch extraction loop ----
    log.info(f"\nExtracting patches (target {n_candidates:,})...")
    t_extract_start = time.time()

    n_written = 0
    n_dropped_nodata = 0
    log_every = max(n_candidates // 20, 1)

    # Cache the shot_id column accessor
    shot_id_col = ("shot_number" if "shot_number" in shots.columns
                   else "shot_id" if "shot_id" in shots.columns
                   else None)
    if shot_id_col is None:
        log.warning("  no shot_id column found, will store 0")

    for i in range(n_candidates):
        year = int(shots["_year"].iloc[i])
        r, c = int(rows[i]), int(cols[i])

        patch = np.empty(
            (N_CHANNELS, PATCH_SIZE, PATCH_SIZE), dtype=np.float32,
        )
        patch[S2_BAND_IDX] = read_patch(srcs[("S2", year)], r, c, PATCH_HALF)
        patch[S1_BAND_IDX] = read_patch(srcs[("S1", year)], r, c, PATCH_HALF)
        patch[DEM_BAND_IDX] = read_patch(srcs[("DEM", 0)], r, c, PATCH_HALF)

        if patch_has_nodata(patch):
            n_dropped_nodata += 1
            continue

        patches_arr[n_written] = patch
        labels_arr[n_written] = shots["agbd"].iloc[i]
        shot_ids_arr[n_written] = (int(shots[shot_id_col].iloc[i])
                                   if shot_id_col else 0)
        years_arr[n_written] = year
        east_arr[n_written] = shots["easting"].iloc[i]
        north_arr[n_written] = shots["northing"].iloc[i]
        block_arr[n_written] = int(block_ids[i])
        split_arr[n_written] = int(splits[i])
        n_written += 1

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t_extract_start
            rate = (i + 1) / elapsed
            eta = (n_candidates - i - 1) / rate
            log.info(f"  [{i+1:,}/{n_candidates:,}] "
                     f"written={n_written:,}, "
                     f"dropped(no-data)={n_dropped_nodata:,}, "
                     f"rate={rate:.0f}/s, "
                     f"eta={eta/60:.1f} min")

    extract_elapsed = time.time() - t_extract_start
    log.info(f"\nExtraction done in {extract_elapsed/60:.1f} min")
    log.info(f"  Patches written: {n_written:,}")
    log.info(f"  Dropped (no-data): {n_dropped_nodata:,}")

    # ---- 8. Trim arrays to actual size ----
    if n_written < n_candidates:
        log.info(f"\nTrimming arrays from {n_candidates:,} to {n_written:,}...")
        # zarr resize: each array independently
        patches_arr.resize(n_written, N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
        labels_arr.resize(n_written)
        shot_ids_arr.resize(n_written)
        years_arr.resize(n_written)
        east_arr.resize(n_written)
        north_arr.resize(n_written)
        block_arr.resize(n_written)
        split_arr.resize(n_written)

    counts.after_edge_drop = n_written

    close_raster_set(srcs)

    # ---- 9. Compute normalization stats over training partition ----
    train_mask = (split_arr[:] == 0)
    norm_stats = compute_normalization_stats(root, train_mask)

    # ---- 10. Write metadata JSON ----
    final_splits = split_arr[:]
    metadata = {
        "aoi": aoi.name,
        "rng_seed": RNG_SEED,
        "patch_size": PATCH_SIZE,
        "n_channels": N_CHANNELS,
        "channels": CHANNEL_NAMES,
        "valid_years": sorted(VALID_YEARS),
        "agbd_cap_mgha": AGBD_CAP,
        "subsample_cell_m": SUBSAMPLE_CELL_M,
        "block_size_m": BLOCK_SIZE_M,
        "split_fractions": {
            "train": TRAIN_FRAC, "val": VAL_FRAC,
            "test": 1 - TRAIN_FRAC - VAL_FRAC,
        },
        "n_strata": N_STRATA,
        "counts": {
            "raw": counts.raw,
            "after_year_filter": counts.after_year_filter,
            "after_agbd_cap": counts.after_agbd_cap,
            "after_spatial_subsample": counts.after_spatial_subsample,
            "after_edge_drop": counts.after_edge_drop,
            "n_train": int((final_splits == 0).sum()),
            "n_val": int((final_splits == 1).sum()),
            "n_test": int((final_splits == 2).sum()),
        },
        "normalization": norm_stats,
        "zarr_path": str(zarr_path.relative_to(PROCESSED_DIR.parent)),
    }
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"\nWrote metadata: {meta_path}")

    total_elapsed = time.time() - t_start
    log.info(f"\n=== Done in {total_elapsed/60:.1f} min ===")
    log.info(f"Output: {zarr_path}")
    log.info(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()