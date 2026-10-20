"""
Shared helpers for training: band list, input scaling, finding the data (on the laptop
or on Kaggle), loading patches into memory, label-fraction subsets and metrics.
Nothing in this file needs PyTorch.
"""

import concurrent.futures as cf
import hashlib
import math
import os
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


def cached_patches(paths, keys, workers=8, name=""):
    """Like load_patches, but the images live in memory-mapped .npy files in cache_dir(), so
    processes running side by side (one per GPU) share ONE copy in RAM (the operating system's
    file cache) instead of each holding its own. The files are built on first use; later calls
    only map them. Labels are small and returned as an ordinary in-memory copy (callers edit them)."""
    paths = list(paths)
    ids = "\n".join(Path(p).stem for p in paths)
    tag = hashlib.md5(ids.encode()).hexdigest()[:10]
    files = {k: cache_dir() / f"labeled_{k}_{len(paths)}_{tag}.npy" for k in keys}
    todo = [k for k in keys if not Path(f"{files[k]}.done").exists()]
    if todo:
        with np.load(paths[0]) as d:
            first = {k: d[k] for k in todo}
        tmp = {k: files[k].with_name(f"{files[k].stem}.{os.getpid()}.tmp.npy") for k in todo}
        arrays = {k: np.lib.format.open_memmap(tmp[k], mode="w+", dtype=first[k].dtype,
                                               shape=(len(paths),) + first[k].shape) for k in todo}

        def fill(i):
            with np.load(paths[i]) as d:
                for k in todo:
                    arrays[k][i] = d[k]

        print(f"  {name}: building the shared cache in {cache_dir()} (one-off)...")
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for n, _ in enumerate(pool.map(fill, range(len(paths))), 1):
                if n % 500 == 0:
                    print(f"    {name}: {n}/{len(paths)} patches cached")
        for k in todo:
            arrays.pop(k).flush()          # dropping the last reference closes the file
            os.replace(tmp[k], files[k])   # (Windows cannot rename a file that is still open)
            Path(f"{files[k]}.done").touch()
    out = {k: np.load(files[k], mmap_mode="r") for k in keys}
    if "label" in out:
        out["label"] = np.array(out["label"])
    gb = sum(a.nbytes for a in out.values()) / 1e9
    print(f"  {name}: {len(paths)} patches, memory-mapped from the shared cache ({gb:.2f} GB)")
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


# -----------------------------------------------------------------------------
# Threshold-free evaluation from probability histograms
# -----------------------------------------------------------------------------
# hist has shape (2, NBINS): row 0 counts forest pixels, row 1 counts cleared pixels,
# by the model's predicted probability of 'cleared' (bin k covers [k/NBINS, (k+1)/NBINS)).
NBINS = 1000


def counts_at_thresholds(hist):
    """tp, fp, fn, tn when predicting 'cleared' for probability >= k/NBINS, for every k."""
    neg, pos = hist[0].astype(np.float64), hist[1].astype(np.float64)
    tp = pos[::-1].cumsum()[::-1]
    fp = neg[::-1].cumsum()[::-1]
    return tp, fp, pos.sum() - tp, neg.sum() - fp


def average_precision(hist):
    """Area under the precision-recall curve for the 'cleared' class (step-wise, as in sklearn)."""
    tp, fp, fn, _ = counts_at_thresholds(hist)
    n_pos = tp[0] + fn[0]
    if n_pos == 0:
        return float("nan")
    tp, fp = tp[::-1], fp[::-1]                 # strictest threshold first: recall rises
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = tp / n_pos
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def best_threshold(hist):
    """Threshold bin k (predict cleared if prob >= k/NBINS) that maximises IoU."""
    tp, fp, fn, _ = counts_at_thresholds(hist)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    return int(np.argmax(iou))


def metrics_at_threshold(hist, k):
    tp, fp, fn, tn = (v[k] for v in counts_at_thresholds(hist))
    m = metrics_from_counts(tp, fp, fn, tn)
    m["pred_cleared_share"] = (tp + fp) / max(tp + fp + fn + tn, 1)
    return m


# -----------------------------------------------------------------------------
# Unlabeled patches (for self-supervised pretraining)
# -----------------------------------------------------------------------------
def unlabeled_index(data_root=None):
    """The unlabeled patch list (unlabeled_downloaded.csv) with a 'path' column added.
    The .npz files may be spread over several folders called 'unlabeled' (as on Kaggle)."""
    csv = find_file("unlabeled_downloaded.csv", data_root)
    df = pd.read_csv(csv)
    paths = {}
    for root in search_roots(data_root):
        paths = {p.stem: str(p) for p in root.rglob("unlabeled/*.npz")}
        if paths:
            break
    missing = sorted(set(df.patch_id) - set(paths))
    if missing:
        raise FileNotFoundError(f"{len(missing)} unlabeled patches not found, e.g. {missing[0]}")
    df["path"] = df.patch_id.map(paths)
    return df.sort_values("patch_id").reset_index(drop=True)


def cache_dir():
    """Scratch space for large temporary files (not saved with Kaggle output, not in git)."""
    d = Path("/tmp/defor_cache") if KAGGLE_WORKING.exists() else REPO / "data" / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_path(suffix, data_root=None):
    """A file whose path ends with `suffix` (e.g. 'pretrain/moco_both/encoder.pt'), searched in
    this session's results folder first, then in the data folders (e.g. an earlier Kaggle
    notebook output attached as input)."""
    suffix = suffix.replace("\\", "/").strip("/")
    name = suffix.split("/")[-1]
    for root in [results_dir()] + search_roots(data_root):
        if not root.exists():
            continue
        for p in sorted(root.rglob(name)):
            if p.as_posix().endswith("/" + suffix):
                return p
    raise FileNotFoundError(f"no file ending in {suffix}")
