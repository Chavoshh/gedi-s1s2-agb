"""Smoke tests: verify the package imports cleanly and config is sane."""
from __future__ import annotations


def test_package_imports():
    import biomass
    assert biomass.__version__ == "0.1.0"


def test_config_constants():
    from biomass.config import BEAMS, POWER_BEAMS, L4A_VARS
    assert len(BEAMS) == 8
    assert len(POWER_BEAMS) == 4
    assert POWER_BEAMS.issubset(set(BEAMS))
    assert "agbd" in L4A_VARS
    assert "shot_number" in L4A_VARS


def test_aoi_lookup():
    from biomass.data.aoi import get_aoi
    dev = get_aoi("dev")
    full = get_aoi("full")
    assert dev.width_deg > 0
    assert full.width_deg > dev.width_deg  # full is larger


def test_aoi_bbox_order():
    """bbox order must be (west, south, east, north) for earthaccess."""
    from biomass.data.aoi import DEV
    w, s, e, n = DEV.bbox
    assert w < e
    assert s < n