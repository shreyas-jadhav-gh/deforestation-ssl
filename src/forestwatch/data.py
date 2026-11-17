"""
Image patches: one patch, a catalogue of patches, two ways of loading many of them at once,
and the nested label budgets used to train with 1% to 100% of the labels.

Every patch is a NumPy .npz file with three arrays:
    post   (13, 256, 256) uint16   2024 dry-season Sentinel-2 composite
    pre    (13, 256, 256) uint16   the same for 2019
    label  (256, 256)     uint8    0 forest, 1 cleared 2020-2023, 255 not scored
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .config import BANDS, CLEARED, FOREST, IGNORE, SCALE


# -----------------------------------------------------------------------------
# One patch
# -----------------------------------------------------------------------------
@dataclass
class Patch:
    """One 2.56 km x 2.56 km patch: its images and its label map."""

    patch_id: str
    post: np.ndarray
    label: np.ndarray
    pre: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.post.ndim != 3 or self.post.shape[0] != len(BANDS):
            raise ValueError(f"{self.patch_id}: post must have shape (13, H, W), got {self.post.shape}")
        if self.label.shape != self.post.shape[1:]:
            raise ValueError(f"{self.patch_id}: label shape {self.label.shape} does not match the image")
        if self.pre is not None and self.pre.shape != self.post.shape:
            raise ValueError(f"{self.patch_id}: pre and post images differ in shape")

    @classmethod
    def from_npz(cls, path) -> "Patch":
        path = Path(path)
        with np.load(path) as d:
            pre = d["pre"] if "pre" in d.files else None
            return cls(path.stem, d["post"], d["label"], pre)

    @property
    def region(self) -> str:
        return self.patch_id.split("_")[0]

    @property
    def shape(self) -> tuple[int, int]:
        return self.label.shape

    def pixel_counts(self) -> dict[str, int]:
        """Pixels per label value, named as in the dataset manifest."""
        return {"forest_px": int(np.sum(self.label == FOREST)),
                "cleared_px": int(np.sum(self.label == CLEARED)),
                "ignore_px": int(np.sum(self.label == IGNORE))}

    @property
    def cleared_share(self) -> float:
        """Share of the scored pixels that were cleared (NaN if nothing is scored)."""
        c = self.pixel_counts()
        scored = c["forest_px"] + c["cleared_px"]
        return c["cleared_px"] / scored if scored else float("nan")

    def image(self, date: str = "post") -> np.ndarray:
        img = {"post": self.post, "pre": self.pre}.get(date)
        if img is None:
            raise ValueError(f"{self.patch_id} has no {date!r} image")
        return img

    def reflectance(self, date: str = "post") -> np.ndarray:
        """The image as float32 reflectance (raw value / 10000), the model's input scale."""
        return self.image(date).astype(np.float32) / SCALE

    def rgb(self, date: str = "post", gain: float = 3.5) -> np.ndarray:
        """A true-colour (B4, B3, B2) picture of shape (H, W, 3) with values in [0, 1]."""
        idx = [BANDS.index(b) for b in ("B4", "B3", "B2")]
        return np.clip(self.reflectance(date)[idx].transpose(1, 2, 0) * gain, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Loading many patches
# -----------------------------------------------------------------------------
class PatchLoader(ABC):
    """Loads chosen arrays (e.g. 'post' and 'label') of many patch files into stacked arrays,
    one row per patch, reading the files in parallel threads."""

    def __init__(self, workers: int = 8, verbose: bool = False):
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.workers = workers
        self.verbose = verbose

    def load(self, paths, keys) -> dict[str, np.ndarray]:
        paths = [Path(p) for p in paths]
        keys = list(keys)
        if not paths:
            raise ValueError("no patch files given")
        if not keys:
            raise ValueError("no arrays requested")
        return self._load(paths, keys)

    @abstractmethod
    def _load(self, paths: list[Path], keys: list[str]) -> dict[str, np.ndarray]:
        """Subclasses decide where the stacked arrays live."""

    @staticmethod
    def _layout(paths, keys):
        """Shape and dtype of each requested array, read from the first file."""
        with np.load(paths[0]) as d:
            missing = [k for k in keys if k not in d.files]
            if missing:
                raise KeyError(f"{paths[0].name} has no array(s) {missing}")
            return {k: (d[k].shape, d[k].dtype) for k in keys}

    def _fill(self, paths, keys, out):
        """Copy every file's arrays into row i of the preallocated arrays in `out`."""
        def one(i):
            with np.load(paths[i]) as d:
                for k in keys:
                    out[k][i] = d[k]

        with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
            for n, _ in enumerate(pool.map(one, range(len(paths))), 1):
                if self.verbose and n % 500 == 0:
                    print(f"    {n}/{len(paths)} patches read")


class InMemoryLoader(PatchLoader):
    """Keeps the stacked arrays in this process's memory."""

    def _load(self, paths, keys):
        layout = self._layout(paths, keys)
        out = {k: np.empty((len(paths),) + shape, dtype=dtype) for k, (shape, dtype) in layout.items()}
        self._fill(paths, keys, out)
        return out


class SharedCacheLoader(PatchLoader):
    """Keeps the stacked arrays in memory-mapped .npy files in a cache folder.

    Processes running side by side (one per GPU on Kaggle) then share one copy of the images
    through the operating system's file cache instead of each holding its own. The files are
    built on first use and only mapped afterwards. Labels are small and returned as an ordinary
    in-memory copy, because the training script changes them."""

    def __init__(self, cache_dir, workers: int = 8, verbose: bool = False, prefix: str = "labeled"):
        super().__init__(workers, verbose)
        self.cache_dir = Path(cache_dir)
        self.prefix = prefix

    def cache_files(self, paths, keys) -> dict[str, Path]:
        """Cache file per array; the name depends on exactly which patches are loaded."""
        ids = "\n".join(Path(p).stem for p in paths)
        tag = hashlib.md5(ids.encode()).hexdigest()[:10]
        return {k: self.cache_dir / f"{self.prefix}_{k}_{len(paths)}_{tag}.npy" for k in keys}

    @staticmethod
    def _done_marker(file: Path) -> Path:
        return Path(f"{file}.done")

    def is_cached(self, paths, keys) -> bool:
        files = self.cache_files([Path(p) for p in paths], keys)
        return all(self._done_marker(f).exists() for f in files.values())

    def _build(self, paths, keys, files):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        layout = self._layout(paths, keys)
        tmp = {k: files[k].with_name(f"{files[k].stem}.{os.getpid()}.tmp.npy") for k in keys}
        arrays = {k: np.lib.format.open_memmap(tmp[k], mode="w+", dtype=layout[k][1],
                                               shape=(len(paths),) + layout[k][0]) for k in keys}
        self._fill(paths, keys, arrays)
        for k in keys:
            arrays.pop(k).flush()         # dropping the last reference closes the file,
            os.replace(tmp[k], files[k])  # which Windows needs before it can be renamed
            self._done_marker(files[k]).touch()

    def _load(self, paths, keys):
        files = self.cache_files(paths, keys)
        todo = [k for k in keys if not self._done_marker(files[k]).exists()]
        if todo:
            self._build(paths, todo, files)
        out = {k: np.load(files[k], mmap_mode="r") for k in keys}
        if "label" in out:
            out["label"] = np.array(out["label"])
        return out


# -----------------------------------------------------------------------------
# A catalogue of patches
# -----------------------------------------------------------------------------
class PatchCatalog:
    """The table of patches (one row each, with region, split and pixel counts) and the
    folder that holds their files. Selecting rows gives a smaller catalogue."""

    REQUIRED = ("patch_id", "region", "split")

    def __init__(self, table: pd.DataFrame, patch_dir):
        missing = [c for c in self.REQUIRED if c not in table.columns]
        if missing:
            raise ValueError(f"patch table is missing column(s) {missing}")
        if table.patch_id.duplicated().any():
            raise ValueError("patch table has duplicate patch ids")
        self.table = table.sort_values("patch_id").reset_index(drop=True)
        self.patch_dir = Path(patch_dir)

    @classmethod
    def from_manifest(cls, csv) -> "PatchCatalog":
        """From a dataset's manifests/patches_downloaded.csv (patch files in ../patches/)."""
        csv = Path(csv)
        return cls(pd.read_csv(csv), csv.parent.parent / "patches")

    @classmethod
    def from_folder(cls, folder) -> "PatchCatalog":
        """From a dataset folder such as data/ or sample_data/."""
        return cls.from_manifest(Path(folder) / "manifests" / "patches_downloaded.csv")

    def __len__(self) -> int:
        return len(self.table)

    def __contains__(self, patch_id) -> bool:
        return patch_id in set(self.table.patch_id)

    def __iter__(self) -> Iterator[Patch]:
        for pid in self.table.patch_id:
            yield self.patch(pid)

    def __getitem__(self, key) -> Patch:
        if isinstance(key, (int, np.integer)):
            return self.patch(self.table.patch_id.iloc[key])
        return self.patch(key)

    def __repr__(self) -> str:
        return f"PatchCatalog({len(self)} patches in {self.patch_dir})"

    @property
    def ids(self) -> list[str]:
        return list(self.table.patch_id)

    def path(self, patch_id: str) -> Path:
        return self.patch_dir / f"{patch_id}.npz"

    @property
    def paths(self) -> list[Path]:
        return [self.path(pid) for pid in self.table.patch_id]

    def patch(self, patch_id: str) -> Patch:
        if patch_id not in self:
            raise KeyError(f"no patch {patch_id!r} in the catalogue")
        return Patch.from_npz(self.path(patch_id))

    def missing_files(self) -> list[str]:
        return [pid for pid in self.table.patch_id if not self.path(pid).exists()]

    def select(self, region: Optional[str] = None, split: Optional[str] = None) -> "PatchCatalog":
        rows = self.table
        if region is not None:
            rows = rows[rows.region == region]
        if split is not None:
            rows = rows[rows.split == split]
        return PatchCatalog(rows, self.patch_dir)

    def summary(self) -> pd.DataFrame:
        """Patches and pixel counts per region and split."""
        counts = [c for c in ("forest_px", "cleared_px", "ignore_px") if c in self.table]
        out = self.table.groupby(["region", "split"]).agg(patches=("patch_id", "size"),
                                                          **{c: (c, "sum") for c in counts})
        if {"forest_px", "cleared_px"} <= set(counts):
            out["cleared_share"] = out.cleared_px / (out.forest_px + out.cleared_px)
        return out

    def load(self, keys=("post", "label"), loader: Optional[PatchLoader] = None) -> dict[str, np.ndarray]:
        return (loader or InMemoryLoader()).load(self.paths, keys)


# -----------------------------------------------------------------------------
# Label budgets
# -----------------------------------------------------------------------------
class LabelBudget:
    """Which training patches count as labelled when only a fraction of the labels is used.

    For one seed, the patches are put in a random order once, and a budget of k patches takes
    the first k. Smaller budgets are therefore always inside larger ones (nested), so two label
    fractions differ only in the amount of labelled data, not in which patches were drawn."""

    def __init__(self, n_total: int, seed: int):
        if n_total < 1:
            raise ValueError("need at least one training patch")
        self.n_total = n_total
        self.seed = seed
        self._order = np.random.default_rng(1000 + seed).permutation(n_total)

    def __repr__(self) -> str:
        return f"LabelBudget(n_total={self.n_total}, seed={self.seed})"

    def size(self, frac: float) -> int:
        if not 0 < frac <= 1:
            raise ValueError(f"label fraction must be in (0, 1], got {frac}")
        return max(1, math.ceil(frac * self.n_total - 1e-9))

    def indices(self, frac: float) -> np.ndarray:
        """Sorted indices (into the training patches) of the labelled subset."""
        return np.sort(self._order[: self.size(frac)])

    def sizes(self, fracs) -> dict[float, int]:
        return {f: self.size(f) for f in fracs}
