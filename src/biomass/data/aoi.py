"""Named Areas Of Interest (AOIs) for this project.

Each AOI is defined as a (west, south, east, north) bounding box in EPSG:4326.
All downstream code should refer to AOIs by name, never hardcode coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AOI:
    """A named bounding box in WGS84 lon/lat."""
    name: str
    description: str
    west: float
    south: float
    east: float
    north: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) — the order earthaccess and STAC use."""
        return (self.west, self.south, self.east, self.north)

    @property
    def width_deg(self) -> float:
        return self.east - self.west

    @property
    def height_deg(self) -> float:
        return self.north - self.south


# ---------- Project AOIs ----------

DEV = AOI(
    name="dev",
    description="MGRS tile 29TNG, central Galicia. Pipeline prototyping.",
    west=-8.5, south=42.6, east=-7.3, north=43.6,
)

FULL = AOI(
    name="full",
    description="Northwest Iberia: Galicia + western Asturias. Paper-scale.",
    west=-9.5, south=41.5, east=-5.5, north=43.5,
)

AOIS: dict[str, AOI] = {"dev": DEV, "full": FULL}


def get_aoi(name: str) -> AOI:
    """Look up an AOI by name. Raises KeyError with helpful message."""
    try:
        return AOIS[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown AOI '{name}'. Available: {sorted(AOIS)}"
        ) from e