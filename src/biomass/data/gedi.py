"""GEDI L4A acquisition, read, and quality filtering."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from biomass.config import BEAMS, DEFAULT_QUALITY, L4A_VARS, POWER_BEAMS

log = logging.getLogger(__name__)


# ---------- Granule filename parsing ----------

_GRANULE_RE = re.compile(
    r"GEDI04_A_(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})_O(\d+)_"
)


def parse_granule_meta(filename: str) -> dict:
    """Extract acquisition datetime and orbit from an L4A granule filename.

    Filename format: GEDI04_A_YYYYDDDHHMMSS_Oxxxxx_*.h5
    """
    m = _GRANULE_RE.search(filename)
    if not m:
        return {"acq_datetime": pd.NaT, "orbit": -1}
    year, doy, hh, mm, ss, orbit = m.groups()
    dt = (datetime(int(year), 1, 1, int(hh), int(mm), int(ss),
                   tzinfo=timezone.utc)
          + pd.Timedelta(days=int(doy) - 1))
    return {"acq_datetime": dt, "orbit": int(orbit)}


# ---------- Single-beam read ----------

def read_beam(
    h5: h5py.File,
    beam: str,
    bbox: tuple[float, float, float, float],
) -> pd.DataFrame | None:
    """Read one beam group from an open L4A file, AOI-clip, return DataFrame.

    Returns None if the beam doesn't exist or contains no shots in the bbox.
    """
    if beam not in h5:
        return None
    g = h5[beam]

    lat = g["lat_lowestmode"][:]
    lon = g["lon_lowestmode"][:]
    w, s, e, n = bbox
    mask = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)

    if not mask.any():
        return None

    data = {v: g[v][mask] for v in L4A_VARS}
    if "land_cover_data" in g and "pft_class" in g["land_cover_data"]:
        data["pft_class"] = g["land_cover_data"]["pft_class"][mask]
    else:
        data["pft_class"] = np.full(mask.sum(), -1, dtype=np.int8)

    df = pd.DataFrame(data)
    df["beam"] = beam
    df["is_power_beam"] = beam in POWER_BEAMS
    return df


# ---------- Whole-granule read ----------

def read_granule(
    local_path: Path,
    bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Read all 8 beams from one local L4A granule, AOI-clip, tag with metadata.

    Returns an unfiltered DataFrame (quality filtering happens separately).
    """
    per_beam = []
    with h5py.File(local_path, "r") as h5:
        for beam in BEAMS:
            df = read_beam(h5, beam, bbox)
            if df is not None:
                per_beam.append(df)

    if not per_beam:
        return pd.DataFrame()

    raw = pd.concat(per_beam, ignore_index=True)
    meta = parse_granule_meta(local_path.name)
    raw["acq_datetime"] = meta["acq_datetime"]
    raw["orbit"] = meta["orbit"]
    raw["granule"] = local_path.name
    return raw


# ---------- Quality filter ----------

def apply_quality_filter(
    df: pd.DataFrame,
    *,
    quality: dict | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Apply the GEDI L4A quality cascade.

    Parameters
    ----------
    df : input shots (one row per shot, output of read_granule).
    quality : override default quality thresholds. Falls back to DEFAULT_QUALITY.
    verbose : if True, print per-condition pass rates.
    """
    q = {**DEFAULT_QUALITY, **(quality or {})}
    n = len(df)
    if n == 0:
        return df

    conditions = {
        "l4_quality_flag":  df["l4_quality_flag"]  == q["l4_quality_flag"],
        "l2_quality_flag":  df["l2_quality_flag"]  == q["l2_quality_flag"],
        "degrade_flag":     df["degrade_flag"]     == q["degrade_flag"],
        "sensitivity":      df["sensitivity"]      >= q["min_sensitivity"],
        "agbd":             df["agbd"]             >= q["min_agbd"],
    }

    if verbose:
        for name, cond in conditions.items():
            log.info(f"  {name:20s}: {cond.sum():>6d} / {n} passed")

    combined = conditions["l4_quality_flag"]
    for cond in conditions.values():
        combined &= cond

    return df[combined].copy()