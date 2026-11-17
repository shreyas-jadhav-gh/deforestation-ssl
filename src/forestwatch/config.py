"""
Settings shared by the whole project: the Sentinel-2 bands, the label values, the four study
regions and the description of one training run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Band order of every image array (the same order as SSL4EO-S12 and EuroSAT).
BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12")
SCALE = 10000.0                      # raw Sentinel-2 value / SCALE = reflectance
FOREST, CLEARED, IGNORE = 0, 1, 255  # label values; IGNORE pixels are never scored

INITS = ("scratch", "ssl4eo", "moco_both", "moco_amazon")   # encoder starting points
INPUTS = ("post", "both")            # post = 2024 image only; both = 2024 and 2019 (change detection)
ROLES = ("labelled", "unlabelled")


def _lon(x: float) -> str:
    return f"{abs(x):g}°{'W' if x < 0 else 'E'}"


def _lat(y: float) -> str:
    return f"{abs(y):g}°{'S' if y < 0 else 'N'}"


@dataclass(frozen=True)
class Region:
    """A study area: a longitude/latitude box in degrees (west and south are negative) and
    the UTM zone its images are projected to, so that pixels are square 10 m cells."""

    name: str
    west: float
    south: float
    east: float
    north: float
    crs: str
    role: str = "labelled"

    def __post_init__(self):
        if not -180 <= self.west < self.east <= 180:
            raise ValueError(f"region {self.name}: need -180 <= west < east <= 180")
        if not -90 <= self.south < self.north <= 90:
            raise ValueError(f"region {self.name}: need -90 <= south < north <= 90")
        if self.role not in ROLES:
            raise ValueError(f"region {self.name}: role must be one of {ROLES}")

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """(west, south, east, north), the order Earth Engine expects."""
        return (self.west, self.south, self.east, self.north)

    def contains(self, lon: float, lat: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north

    def overlaps(self, other: "Region") -> bool:
        return (self.west < other.east and other.west < self.east
                and self.south < other.north and other.south < self.north)

    def gap_to(self, other: "Region") -> float:
        """Smallest east-west or north-south distance between the two boxes, in degrees
        (0 if they touch or overlap)."""
        dx = max(other.west - self.east, self.west - other.east, 0.0)
        dy = max(other.south - self.north, self.south - other.north, 0.0)
        return max(dx, dy)

    def __str__(self) -> str:
        return (f"{self.name:12s} {_lon(self.west)}-{_lon(self.east)}, "
                f"{_lat(self.north)}-{_lat(self.south)}  ({self.role}, {self.crs})")


REGIONS = {
    "amazon":   Region("amazon", -64.0, -10.5, -62.0, -8.5, "EPSG:32720"),   # Rondonia, Brazil
    "congo":    Region("congo", 24.0, -3.0, 26.0, -1.0, "EPSG:32735"),       # eastern DR Congo
    "amazon_u": Region("amazon_u", -66.0, -10.5, -64.1, -8.5, "EPSG:32720", "unlabelled"),
    "congo_u":  Region("congo_u", 26.1, -3.0, 28.0, -1.0, "EPSG:32735", "unlabelled"),
}

# Run names as the training script writes them, e.g. full_scratch_lr0.0002_f0.1_s0 or
# cd_moco_both_cd_lr0.0005_f1_s0 (the _cd marks a change-detection run).
_RUN_NAME = re.compile(r"^(?P<tag>[a-z]+)_(?P<init>[a-z0-9_]+?)(?P<cd>_cd)?"
                       r"_lr(?P<lr>[0-9.e+-]+)_f(?P<frac>[0-9.]+)_s(?P<seed>\d+)$")


@dataclass(frozen=True)
class RunConfig:
    """The settings that identify one fine-tuning run."""

    init: str
    lr: float
    frac: float
    seed: int
    input: str = "post"
    tag: str = "full"

    def __post_init__(self):
        if self.init not in INITS:
            raise ValueError(f"unknown init {self.init!r} (expected one of {INITS})")
        if not 0 < self.frac <= 1:
            raise ValueError(f"label fraction must be in (0, 1], got {self.frac}")
        if self.lr <= 0:
            raise ValueError(f"learning rate must be positive, got {self.lr}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {self.seed!r}")
        if self.input not in INPUTS:
            raise ValueError(f"input must be one of {INPUTS}, got {self.input!r}")

    @property
    def name(self) -> str:
        """The run's name, used for its checkpoint, histogram and per-patch files."""
        cd = "_cd" if self.input == "both" else ""
        return f"{self.tag}_{self.init}{cd}_lr{self.lr:g}_f{self.frac:g}_s{self.seed}"

    @property
    def label_seed(self) -> int:
        """Seed of the random draw that picks this run's labelled patches."""
        return 1000 + self.seed

    @classmethod
    def from_name(cls, name: str) -> "RunConfig":
        m = _RUN_NAME.match(name)
        if not m:
            raise ValueError(f"not a run name: {name!r}")
        return cls(init=m["init"], lr=float(m["lr"]), frac=float(m["frac"]), seed=int(m["seed"]),
                   input="both" if m["cd"] else "post", tag=m["tag"])
