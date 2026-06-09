"""End-to-end GEDI L4A extraction pipeline: download → filter → shard → aggregate.

The functions here are AOI-agnostic. Hydra-driven scripts compose them.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import earthaccess
import geopandas as gpd
import pandas as pd

from biomass.config import L4A_SHORT_NAME
from biomass.data.aoi import AOI
from biomass.data.gedi import apply_quality_filter, read_granule

log = logging.getLogger(__name__)


# ---------- State (resumable runs) ----------

def load_state(state_file: Path) -> set[str]:
    """Load the set of granule IDs already processed."""
    if state_file.exists():
        return set(json.loads(state_file.read_text())["done"])
    return set()


def save_state(state_file: Path, done: set[str]) -> None:
    """Persist the set of completed granule IDs."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"done": sorted(done)}))


# ---------- Single-granule processing ----------

def process_one_granule(
    granule,
    cache_dir: Path,
    shards_dir: Path,
    failed_log: Path,
    aoi: AOI,
    quality: dict | None = None,
) -> int:
    """Download, read, filter, and shard one granule. Returns shots kept.

    On any failure, logs to failed_log and returns 0. Always deletes the
    downloaded granule afterward to keep peak disk usage bounded.
    """
    ur = granule["umm"]["GranuleUR"]

    # --- Download ---
    try:
        paths = earthaccess.download([granule], local_path=str(cache_dir))
        p = Path(paths[0])
    except Exception as e:
        log.warning(f"download failed for {ur}: {e}")
        with failed_log.open("a") as f:
            f.write(f"{ur}\tdownload\t{e}\n")
        return 0

    # --- Read + filter ---
    try:
        raw = read_granule(p, aoi.bbox)
        if raw.empty:
            p.unlink(missing_ok=True)
            return 0
        filtered = apply_quality_filter(raw, quality=quality, verbose=False)
    except Exception as e:
        log.warning(f"read/filter failed for {ur}: {e}")
        with failed_log.open("a") as f:
            f.write(f"{ur}\tread\t{e}\n")
        p.unlink(missing_ok=True)
        return 0

    # --- Shard ---
    n_kept = len(filtered)
    if n_kept > 0:
        shards_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shards_dir / f"{p.stem}.parquet"
        filtered.to_parquet(shard_path, index=False)

    # --- Always clean up the downloaded granule ---
    p.unlink(missing_ok=True)
    return n_kept


# ---------- Full run ----------

def run_extraction(
    aoi: AOI,
    time_range: tuple[str, str],
    cache_dir: Path,
    shards_dir: Path,
    state_file: Path,
    failed_log: Path,
    output_path: Path,
    quality: dict | None = None,
    state_save_every: int = 10,
) -> Path:
    """Run the full pipeline: query → loop → aggregate → write final parquet.

    Resumable: skips granules already in state_file. Reports progress every
    granule. Final output is a GeoParquet at output_path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Auth + query ---
    auth = earthaccess.login(strategy="netrc")
    assert auth.authenticated, "Earthdata auth failed"

    log.info(f"Querying L4A granules for AOI '{aoi.name}'")
    log.info(f"  bbox: {aoi.bbox}")
    log.info(f"  time: {time_range[0]} to {time_range[1]}")
    results = earthaccess.search_data(
        short_name=L4A_SHORT_NAME,
        bounding_box=aoi.bbox,
        temporal=time_range,
        count=-1,
    )
    log.info(f"Found {len(results)} granules")

    done = load_state(state_file)
    log.info(f"Already processed: {len(done)} (will skip)")

    # --- Main loop ---
    t0 = time.time()
    total_kept = 0
    for i, g in enumerate(results):
        ur = g["umm"]["GranuleUR"]
        if ur in done:
            continue

        elapsed_min = (time.time() - t0) / 60
        log.info(f"[{i+1}/{len(results)}] {ur}  (elapsed {elapsed_min:.1f} min)")

        n_kept = process_one_granule(
            granule=g,
            cache_dir=cache_dir,
            shards_dir=shards_dir,
            failed_log=failed_log,
            aoi=aoi,
            quality=quality,
        )
        total_kept += n_kept
        log.info(f"  kept {n_kept} shots (cumulative: {total_kept})")

        done.add(ur)
        if (i + 1) % state_save_every == 0:
            save_state(state_file, done)

    save_state(state_file, done)

    # --- Aggregate shards into a single GeoParquet ---
    return aggregate_shards(shards_dir, output_path)


def aggregate_shards(shards_dir: Path, output_path: Path) -> Path:
    """Concatenate all shard parquets into a single GeoParquet."""
    shards = sorted(shards_dir.glob("*.parquet"))
    log.info(f"Concatenating {len(shards)} shards...")

    if not shards:
        log.warning("No shards to aggregate; nothing written.")
        return output_path

    all_df = pd.concat(
        [pd.read_parquet(s) for s in shards],
        ignore_index=True,
    )
    log.info(f"Total shots: {len(all_df)}")

    gdf = gpd.GeoDataFrame(
        all_df,
        geometry=gpd.points_from_xy(all_df["lon_lowestmode"],
                                    all_df["lat_lowestmode"]),
        crs="EPSG:4326",
    )
    gdf.to_parquet(output_path, index=False)
    log.info(f"Wrote {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Summary stats — useful at the end of an overnight run
    log.info("AGBD summary (Mg/ha):")
    log.info("\n" + all_df["agbd"].describe().to_string())
    log.info("Power vs coverage beam split:")
    log.info("\n" + all_df["is_power_beam"].value_counts().to_string())
    log.info("Shots per year:")
    log.info("\n" + all_df["acq_datetime"].dt.year.value_counts()
                                              .sort_index().to_string())
    return output_path