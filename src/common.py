"""
Shared helpers for training: band list, input scaling, finding the data (on the laptop
or on Kaggle), loading patches into memory, label-fraction subsets and metrics.
Nothing in this file needs PyTorch.
"""

import concurrent.futures as cf
import math
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
SCALE = 10000.0   # inputs are divided by this (reflectance), as in SSL4EO-S12 / TorchGeo weights
IGNORE = 255      # label value excluded from loss and metrics

REPO = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


# -----------------------------------------------------------------------------
# Finding the data
# -----------------------------------------------------------------------------
def search_roots(data_root=None):
    roots = [Path(data_root)] if data_root else []
    roots += [REPO / "data", KAGGLE_INPUT]
    return [r for r in roots if r.exists()]


def find_file(name, data_root=None):
    """First file called `name` under the data folders (laptop data/ first, then Kaggle input)."""
    roots = search_roots(data_root)
    for root in roots:
        hits = sorted(root.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"could not find {name} under {[str(r) for r in roots]}")


def labeled_index(data_root=None):
    """The labeled patch list (patches_downloaded.csv) with a 'path' column added."""
    csv = find_file("patches_downloaded.csv", data_root)
    df = pd.read_csv(csv)
    patch_dir = csv.parent.parent / "patches"
    df["path"] = [str(patch_dir / f"{pid}.npz") for pid in df.patch_id]
    missing = [p for p in df.path.iloc[:: max(1, len(df) // 50)] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"patch files not found next to {csv}, e.g. {missing[0]}")
    return df.sort_values("patch_id").reset_index(drop=True)


def results_dir():
    base = KAGGLE_WORKING if KAGGLE_WORKING.exists() else REPO
    return base / "results"


# -----------------------------------------------------------------------------
# Loading patches into memory
# -----------------------------------------------------------------------------
def load_patches(paths, keys, workers=8, name=""):
    """Load the arrays `keys` from every .npz in `paths` into preallocated stacked arrays."""
    paths = list(paths)
    with np.load(paths[0]) as d:
        first = {k: d[k] for k in keys}
    out = {k: np.empty((len(paths),) + first[k].shape, dtype=first[k].dtype) for k in keys}

    def fill(i):
        with np.load(paths[i]) as d:
            for k in keys:
                out[k][i] = d[k]

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(fill, range(len(paths))), 1):
            if n % 500 == 0:
                print(f"    {name}: {n}/{len(paths)} patches loaded")
    gb = sum(a.nbytes for a in out.values()) / 1e9
    print(f"  {name}: {len(paths)} patches in memory ({gb:.2f} GB)")
    return out


# -----------------------------------------------------------------------------
# Label fractions and batches
# -----------------------------------------------------------------------------
def nested_subset(n_total, frac, seed):
    """Indices of a frac-sized subset of range(n_total).

    For a given seed, smaller fractions are always contained in larger ones (nested),
    so differences between fractions come only from the amount of labeled data.
    Different seeds draw different subsets.
    """
    order = np.random.default_rng(1000 + seed).permutation(n_total)
    k = max(1, math.ceil(frac * n_total - 1e-9))
    return np.sort(order[:k])


def batch_stream(indices, batch_size, rng):
    """Endless stream of batches: shuffled passes over `indices` (repeats if fewer than a batch)."""
    indices = np.asarray(indices)
    buf = np.empty(0, dtype=indices.dtype)
    while True:
        while len(buf) < batch_size:
            buf = np.concatenate([buf, rng.permutation(indices)])
        yield buf[:batch_size]
        buf = buf[batch_size:]


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def metrics_from_counts(tp, fp, fn, tn):
    """Metrics for the 'cleared' class from pixel counts (ignore pixels already excluded)."""
    tp, fp, fn, tn = (float(v) for v in (tp, fp, fn, tn))

    def div(a, b):
        return a / b if b > 0 else float("nan")

    iou_cleared = div(tp, tp + fp + fn)
    iou_forest = div(tn, tn + fp + fn)
    return {"iou": iou_cleared,
            "f1": div(2 * tp, 2 * tp + fp + fn),
            "precision": div(tp, tp + fp),
            "recall": div(tp, tp + fn),
            "miou": (iou_cleared + iou_forest) / 2,
            "accuracy": div(tp + tn, tp + fp + fn + tn)}
