"""Phase 1, step 2: download one granule, read all beams, apply quality filter."""
from __future__ import annotations

import logging
from pathlib import Path

import earthaccess
import geopandas as gpd
import h5py
import hydra
from omegaconf import DictConfig

from biomass.config import BEAMS, L4A_SHORT_NAME, RAW_DIR
from biomass.data.aoi import get_aoi
from biomass.data.gedi import apply_quality_filter, read_granule
from biomass.log_setup import setup_logging

log = logging.getLogger(__name__)

GEDI_CACHE = RAW_DIR / "gedi"
GEDI_CACHE.mkdir(parents=True, exist_ok=True)


def quick_land_check(local_path: Path, bbox: tuple) -> tuple[int, int]:
    """Cheap land-vs-ocean proxy: count shots with valid (non-fill) AGBD."""
    n_land = n_total = 0
    with h5py.File(local_path, "r") as h5:
        for beam in BEAMS:
            if beam not in h5:
                continue
            g = h5[beam]
            lat = g["lat_lowestmode"][:]
            lon = g["lon_lowestmode"][:]
            w, s, e, n = bbox
            mask = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
            if not mask.any():
                continue
            agbd = g["agbd"][mask]
            n_total += int(mask.sum())
            n_land += int((agbd > 0).sum())
    return n_land, n_total


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)

    aoi = get_aoi(cfg.aoi.name)
    log.info(f"AOI: {aoi.name} — bbox {aoi.bbox}")

    auth = earthaccess.login(strategy="netrc")
    assert auth.authenticated

    results = earthaccess.search_data(
        short_name=L4A_SHORT_NAME,
        bounding_box=aoi.bbox,
        temporal=(cfg.time_range.start, cfg.time_range.end),
        count=-1,
    )
    log.info(f"Total granules in AOI: {len(results)}")

    granule = None
    local_path = None
    for i, g in enumerate(results[:15]):
        log.info(f"granule {i}: downloading...")
        paths = earthaccess.download([g], local_path=str(GEDI_CACHE))
        p = Path(paths[0])
        n_land, n_total = quick_land_check(p, aoi.bbox)
        frac = n_land / n_total if n_total else 0
        log.info(f"granule {i}: {n_total} shots in AOI, "
                 f"{n_land} on land ({frac*100:.1f}%)")

        if n_land >= 200:
            granule = g
            local_path = p
            log.info(f"  -> using granule {i}")
            break
        p.unlink()

    if granule is None:
        raise RuntimeError("No suitable granule found in first 15")

    raw = read_granule(local_path, aoi.bbox)
    log.info(f"Raw shots in AOI: {len(raw)}")

    filtered = apply_quality_filter(raw, verbose=True)
    log.info(f"After quality filter: {len(filtered)} shots")

    gdf = gpd.GeoDataFrame(
        filtered,
        geometry=gpd.points_from_xy(filtered["lon_lowestmode"],
                                    filtered["lat_lowestmode"]),
        crs="EPSG:4326",
    )

    log.info("AGBD summary (Mg/ha):")
    log.info("\n" + filtered["agbd"].describe().to_string())
    log.info("Power vs coverage beam split:")
    log.info("\n" + filtered["is_power_beam"].value_counts().to_string())


if __name__ == "__main__":
    main()