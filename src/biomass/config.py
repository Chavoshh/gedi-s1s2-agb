"""Project-wide constants. Not Hydra configs - these are code-level invariants
(GEDI variable names, beam IDs) that don't change between experiments."""
from __future__ import annotations

from pathlib import Path

# ---------- Project paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# ---------- GEDI L4A invariants ----------

# All 8 beam group names, in order. The last 4 are full-power beams.
BEAMS: tuple[str, ...] = (
    "BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011",
    "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011",
)
POWER_BEAMS: frozenset[str] = frozenset({
    "BEAM0101", "BEAM0110", "BEAM1000", "BEAM1011",
})

# Per-beam 1D variables we extract from each L4A granule.
L4A_VARS: tuple[str, ...] = (
    "shot_number",
    "lat_lowestmode", "lon_lowestmode",
    "agbd", "agbd_se",
    "l4_quality_flag", "l2_quality_flag", "degrade_flag",
    "sensitivity", "delta_time",
)

# Earthaccess short name for the cloud-hosted L4A V2.1 collection.
L4A_SHORT_NAME = "GEDI_L4A_AGB_Density_V2_1_2056"

# Quality cascade defaults. These can be overridden via Hydra config.
DEFAULT_QUALITY = {
    "l4_quality_flag": 1,
    "l2_quality_flag": 1,
    "degrade_flag": 0,
    "min_sensitivity": 0.95,
    "min_agbd": 0.0,  # negative AGBD is a known artifact / fill value
}