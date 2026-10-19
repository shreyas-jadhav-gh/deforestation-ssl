"""
Patch grid, spatial splits and patch downloads for the deforestation SSL project.

Run from the repository root with the 'defor' environment active:

    python src/dataprep/patches.py grid       # 1. build grid, label stats, splits, selection
    python src/dataprep/patches.py pilot      # 2. download the 20-patch pilot set
    python src/dataprep/patches.py preview    # 3. save PNG previews of downloaded patches
    python src/dataprep/patches.py download   # 4. download every selected patch
    python src/dataprep/patches.py download --limit 400   # ...or only the next 400 of them
    python src/dataprep/patches.py summary    # 5. check files, write dataset index and band statistics
    python src/dataprep/patches.py repair     #    re-fetch only files that 'summary' found broken

Downloads can be stopped with Ctrl+C and resumed by running the same command again:
patches already on disk are skipped.

Each patch is saved as data/patches/<patch_id>.npz containing:
    post  : (13, 256, 256) uint16  Sentinel-2 L1C, cloud-masked dry-season (Jun-Sep) 2024 median
    pre   : (13, 256, 256) uint16  same for 2019
    label : (256, 256)     uint8   0   = forest in 2019, still forest
                                   1   = forest in 2019, cleared 2020-2023
                                   255 = ignore (not forest in 2019, cleared in 2024,
                                         water, or no clear 2024 view)
Band order: B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12 (same as SSL4EO-S12 and EuroSAT).
A value of 0 in every band of a pixel means "no clear observation".
"""

import argparse
import concurrent.futures as cf
import math
import os
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from pyproj import Transformer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT = "nice-dispatcher-512012-i4"
HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"  # for many parallel requests

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "data" / "manifests"
PATCH_DIR = ROOT / "data" / "patches"
PREVIEW_DIR = ROOT / "results" / "previews"
GRID_CSV = MANIFEST_DIR / "patches_all.csv"
LOG_CSV = MANIFEST_DIR / "download_log.csv"

PATCH_PX = 256                         # patch size in pixels
SCALE_M = 10                           # Sentinel-2 resolution in metres
PATCH_M = PATCH_PX * SCALE_M           # 2560 m per patch side
STATS_SCALE_M = 20                     # label statistics resolution (finer than Hansen's 30 m)
STATS_PX = PATCH_M // STATS_SCALE_M    # 128 statistics pixels per patch side
STATS_CHUNK = 32                       # patches per side in one label-statistics request

REGIONS = {
    # bbox = (lon_min, lat_min, lon_max, lat_max); crs = local UTM zone, so pixels are square 10 m
    "amazon": {"bbox": (-64.0, -10.5, -62.0, -8.5), "crs": "EPSG:32720"},  # Rondonia, UTM 20S
    "congo":  {"bbox": (24.0, -3.0, 26.0, -1.0),    "crs": "EPSG:32735"},  # Eastern DRC, UTM 35S
}

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
POST_YEAR, PRE_YEAR = 2024, 2019
DRY_START, DRY_END = "06-01", "10-01"  # June to September
CLEAR_THRESHOLD = 0.60                 # Cloud Score+ cs_cdf: keep pixels at least this clear
HANSEN = "UMD/hansen/global_forest_change_2024_v1_12"

SEED = 42
BLOCK_PATCHES = 10                     # spatial blocks of 10 x 10 patches (25.6 km)
SPLIT_FRACTIONS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
MIN_VALID_FRACTION = 0.20              # keep patches with at least 20% non-ignore pixels
CAPS = {("amazon", "train"): 2000, ("amazon", "val"): 400,
        ("amazon", "test"): 400, ("congo", "test"): 1000}
PILOT = {("amazon", "train"): 10, ("congo", "test"): 10}
WORKERS = 8                            # parallel download requests


# -----------------------------------------------------------------------------
# Earth Engine helpers
# -----------------------------------------------------------------------------
def init_ee():
    ee.Initialize(project=PROJECT, opt_url=HIGH_VOLUME_URL)


def make_grid(x0, y_top, crs, width, height, scale):
    """Pixel grid with upper-left corner (x0, y_top) in the given projected CRS."""
    return {
        "dimensions": {"width": int(width), "height": int(height)},
        "affineTransform": {"scaleX": scale, "shearX": 0, "translateX": float(x0),
                            "shearY": 0, "scaleY": -scale, "translateY": float(y_top)},
        "crsCode": crs,
    }


def compute_pixels(image, grid, retries=5):
    """Fetch an image on a grid as a numpy structured array, retrying transient errors."""
    for attempt in range(retries):
        try:
            return ee.data.computePixels({"expression": image,
                                          "fileFormat": "NUMPY_NDARRAY",
                                          "grid": grid})
        except Exception as e:  # rate limits, network hiccups, transient server errors
            if attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"    retrying in {wait}s: {str(e)[:120]}")
            time.sleep(wait)


def label_image():
    """0 = still forest, 1 = cleared 2020-2023, 255 = ignore."""
    gfc = ee.Image(HANSEN)
    treecover = gfc.select("treecover2000")
    lossyear = gfc.select("lossyear").unmask(0)  # never-cleared pixels have no value: fill with 0
    forest2019 = treecover.gte(30).And(lossyear.eq(0).Or(lossyear.gte(20)))
    loss_2020_23 = lossyear.gte(20).And(lossyear.lte(23))
    return (ee.Image(255)
            .where(forest2019, 0)
            .where(forest2019.And(loss_2020_23), 1)
            .where(lossyear.eq(24), 255)          # cleared in 2024: timing vs image unclear
            .rename("label")
            .toUint8())


def dry_season_composite(geom, year):
    """Cloud-masked (Cloud Score+) median of all June-September Sentinel-2 L1C scenes."""
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    def mask_clouds(img):
        return img.updateMask(img.select("cs_cdf").gte(CLEAR_THRESHOLD)).select(BANDS)

    return (s2.filterBounds(geom)
              .filterDate(f"{year}-{DRY_START}", f"{year}-{DRY_END}")
              .linkCollection(cs_plus, ["cs_cdf"])
              .map(mask_clouds)
              .median())


def patch_image(geom):
    post = (dry_season_composite(geom, POST_YEAR).unmask(0).toUint16()
            .rename([f"post_{b}" for b in BANDS]))
    pre = (dry_season_composite(geom, PRE_YEAR).unmask(0).toUint16()
           .rename([f"pre_{b}" for b in BANDS]))
    label = label_image().unmask(255).toUint16()
    return post.addBands(pre).addBands(label)


# -----------------------------------------------------------------------------
# Step 1: grid, label statistics, spatial splits, selection
# -----------------------------------------------------------------------------
def build_grid(name):
    """All 2560 m patches (in the local UTM zone) that lie fully inside the region's box."""
    cfg = REGIONS[name]
    lon0, lat0, lon1, lat1 = cfg["bbox"]
    to_utm = Transformer.from_crs("EPSG:4326", cfg["crs"], always_xy=True)
    to_ll = Transformer.from_crs(cfg["crs"], "EPSG:4326", always_xy=True)

    t = np.linspace(0.0, 1.0, 201)
    edge_lon = np.concatenate([lon0 + (lon1 - lon0) * t, np.full_like(t, lon1),
                               lon1 - (lon1 - lon0) * t, np.full_like(t, lon0)])
    edge_lat = np.concatenate([np.full_like(t, lat0), lat0 + (lat1 - lat0) * t,
                               np.full_like(t, lat1), lat1 - (lat1 - lat0) * t])
    ex, ey = to_utm.transform(edge_lon, edge_lat)
    ex, ey = np.asarray(ex), np.asarray(ey)

    x_origin = math.floor(ex.min() / PATCH_M) * PATCH_M
    y_origin = math.ceil(ey.max() / PATCH_M) * PATCH_M
    ncols = math.ceil((ex.max() - x_origin) / PATCH_M)
    nrows = math.ceil((y_origin - ey.min()) / PATCH_M)

    rows = []
    for r in range(nrows):
        for c in range(ncols):
            x0 = x_origin + c * PATCH_M
            y_top = y_origin - r * PATCH_M
            cx = np.array([x0, x0 + PATCH_M, x0 + PATCH_M, x0], dtype=float)
            cy = np.array([y_top, y_top, y_top - PATCH_M, y_top - PATCH_M], dtype=float)
            clon, clat = to_ll.transform(cx, cy)
            clon, clat = np.asarray(clon), np.asarray(clat)
            if clon.min() >= lon0 and clon.max() <= lon1 and clat.min() >= lat0 and clat.max() <= lat1:
                lon_c, lat_c = to_ll.transform(x0 + PATCH_M / 2, y_top - PATCH_M / 2)
                rows.append({"patch_id": f"{name}_r{r:03d}_c{c:03d}", "region": name,
                             "row": r, "col": c, "x0": x0, "y_top": y_top,
                             "lon": round(float(lon_c), 5), "lat": round(float(lat_c), 5)})
    return pd.DataFrame(rows)


def add_label_stats(name, df):
    """Fraction of label 0 and label 1 pixels in every patch, from 20 m label chunks."""
    crs = REGIONS[name]["crs"]
    label = label_image().unmask(255)
    x_origin = float((df.x0 - df.col * PATCH_M).iloc[0])
    y_origin = float((df.y_top + df.row * PATCH_M).iloc[0])
    rmax, cmax = int(df.row.max()), int(df.col.max())
    n_chunks = math.ceil((rmax + 1) / STATS_CHUNK) * math.ceil((cmax + 1) / STATS_CHUNK)

    fracs, k = {}, 0
    for r0 in range(0, rmax + 1, STATS_CHUNK):
        for c0 in range(0, cmax + 1, STATS_CHUNK):
            k += 1
            sub = df[(df.row >= r0) & (df.row < r0 + STATS_CHUNK) &
                     (df.col >= c0) & (df.col < c0 + STATS_CHUNK)]
            if sub.empty:
                continue
            nr = min(STATS_CHUNK, rmax + 1 - r0)
            nc = min(STATS_CHUNK, cmax + 1 - c0)
            grid = make_grid(x_origin + c0 * PATCH_M, y_origin - r0 * PATCH_M, crs,
                             nc * STATS_PX, nr * STATS_PX, STATS_SCALE_M)
            arr = compute_pixels(label, grid)["label"]
            blocks = arr.reshape(nr, STATS_PX, nc, STATS_PX).swapaxes(1, 2)
            for p in sub.itertuples(index=False):
                b = blocks[p.row - r0, p.col - c0]
                fracs[p.patch_id] = (float((b == 0).mean()), float((b == 1).mean()))
            print(f"  label chunk {k}/{n_chunks} done")

    df = df.copy()
    df["f0"] = df.patch_id.map(lambda i: fracs[i][0])
    df["f1"] = df.patch_id.map(lambda i: fracs[i][1])
    df["valid_frac"] = df.f0 + df.f1
    df["cleared_share"] = np.where(df.valid_frac > 0, df.f1 / df.valid_frac.clip(lower=1e-9), 0.0)
    return df


def assign_spatial_splits(df, rng):
    """Assign whole 10x10-patch blocks to train/val/test, then drop val/test patches touching train."""
    df = df.copy()
    df["block"] = (df.row // BLOCK_PATCHES).astype(str) + "_" + (df.col // BLOCK_PATCHES).astype(str)
    blocks = np.array(sorted(df.block.unique()))
    order = rng.permutation(len(blocks))
    n_train = round(len(blocks) * SPLIT_FRACTIONS[0][1])
    n_val = round(len(blocks) * SPLIT_FRACTIONS[1][1])
    block_split = {}
    for rank, i in enumerate(order):
        block_split[blocks[i]] = ("train" if rank < n_train
                                  else "val" if rank < n_train + n_val else "test")
    df["split"] = df.block.map(block_split)

    split_at = dict(zip(zip(df.row, df.col), df.split))

    def touches_train(r, c, s):
        if s == "train":
            return False
        return any(split_at.get((r + dr, c + dc)) == "train"
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))

    df["buffer_drop"] = [touches_train(r, c, s) for r, c, s in zip(df.row, df.col, df.split)]
    return df


def select_patches(df):
    df = df.copy()
    df["selected"] = False
    df["pilot"] = False
    eligible = (~df.buffer_drop.astype(bool)) & (df.valid_frac >= MIN_VALID_FRACTION)
    for (region, split), cap in CAPS.items():
        pool = df[eligible & (df.region == region) & (df.split == split)]
        take = pool.sample(n=min(cap, len(pool)), random_state=SEED)
        df.loc[take.index, "selected"] = True
    for (region, split), n in PILOT.items():
        pool = df[df.selected & (df.region == region) & (df.split == split)]
        # favour patches that contain clearing, so the pilot previews are informative
        take = pool.sample(n=min(n, len(pool)), weights=pool.cleared_share + 0.02,
                           random_state=SEED)
        df.loc[take.index, "pilot"] = True
    return df


def print_summary(df):
    print("\nregion  split   grid  no_buffer  valid>=20%  selected  cleared share (selected)")
    for (region, split), g in df.groupby(["region", "split"]):
        no_buf = int((~g.buffer_drop.astype(bool)).sum())
        valid = int(((~g.buffer_drop.astype(bool)) & (g.valid_frac >= MIN_VALID_FRACTION)).sum())
        s = g[g.selected]
        share = s.f1.sum() / max(s.valid_frac.sum(), 1e-9)
        print(f"{region:7s} {split:5s} {len(g):6d} {no_buf:10d} {valid:11d} {len(s):9d} {share:12.1%}")
    print(f"\npilot patches: {int(df.pilot.sum())}")
    print(f"saved {GRID_CSV}")


def cmd_grid():
    init_ee()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    frames = []
    for name in REGIONS:
        df = build_grid(name)
        print(f"{name}: {len(df)} patches in grid, computing label statistics...")
        df = add_label_stats(name, df)
        if name == "amazon":
            df = assign_spatial_splits(df, rng)
        else:
            df["split"] = "test"          # the whole Congo box is the cross-region test
            df["buffer_drop"] = False
        frames.append(df)
    df = select_patches(pd.concat(frames, ignore_index=True))
    df.to_csv(GRID_CSV, index=False)
    print_summary(df)


# -----------------------------------------------------------------------------
# Steps 2 and 4: download patches
# -----------------------------------------------------------------------------
def load_manifest(column):
    if not GRID_CSV.exists():
        raise SystemExit("No grid yet. Run:  python src/dataprep/patches.py grid")
    df = pd.read_csv(GRID_CSV)
    return df[df[column].astype(bool)]


def download_one(p):
    t0 = time.time()
    crs = REGIONS[p.region]["crs"]
    geom = ee.Geometry.Rectangle([float(p.x0), float(p.y_top - PATCH_M),
                                  float(p.x0 + PATCH_M), float(p.y_top)],
                                 proj=crs, geodesic=False)
    grid = make_grid(p.x0, p.y_top, crs, PATCH_PX, PATCH_PX, SCALE_M)
    arr = compute_pixels(patch_image(geom), grid)

    post = np.stack([arr[f"post_{b}"] for b in BANDS]).astype(np.uint16)
    pre = np.stack([arr[f"pre_{b}"] for b in BANDS]).astype(np.uint16)
    label = arr["label"].astype(np.uint8)
    post_nodata = (post == 0).all(axis=0)
    label[post_nodata] = 255              # no clear 2024 view: cannot be scored

    tmp = PATCH_DIR / f"{p.patch_id}.tmp.npz"
    final = PATCH_DIR / f"{p.patch_id}.npz"
    np.savez_compressed(tmp, post=post, pre=pre, label=label)
    os.replace(tmp, final)                # never leaves a half-written file behind

    return {"patch_id": p.patch_id, "seconds": round(time.time() - t0, 1),
            "MB": round(final.stat().st_size / 1e6, 2),
            "post_nodata": float(post_nodata.mean()),
            "pre_nodata": float((pre == 0).all(axis=0).mean()),
            "cleared_px": int((label == 1).sum()), "valid_px": int((label != 255).sum())}


def run_downloads(df, limit=0, worker=None, out_dir=None, log_csv=None):
    """Download the patches in df in parallel. worker/out_dir/log_csv default to the labeled set."""
    worker = worker or download_one
    out_dir = out_dir or PATCH_DIR
    log_csv = log_csv or LOG_CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in df.itertuples(index=False)
            if not (out_dir / f"{p.patch_id}.npz").exists()]
    remaining = len(todo)
    # Shuffle (reproducibly) so that a partial run is a balanced mix of regions and splits
    order = np.random.default_rng(SEED).permutation(len(todo))
    todo = [todo[i] for i in order]
    if limit:
        todo = todo[:limit]
    print(f"{len(df)} patches requested, {len(df) - remaining} already downloaded, "
          f"{remaining} remaining, downloading {len(todo)} now")
    if not todo:
        return
    init_ee()
    results, failed = [], []
    t_start = time.time()
    pool = cf.ThreadPoolExecutor(max_workers=WORKERS)
    futures = {pool.submit(worker, p): p.patch_id for p in todo}
    pending, i = set(futures), 0
    try:
        while pending:
            # short timeouts keep Ctrl+C responsive on Windows
            done, pending = cf.wait(pending, timeout=1.0, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                i += 1
                pid = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    failed.append(pid)
                    print(f"  FAILED {pid}: {str(e)[:200]}")
                if i % 50 == 0 or i == len(todo):
                    rate = i / max(time.time() - t_start, 1e-9)
                    eta = (len(todo) - i) / max(rate, 1e-9) / 60
                    print(f"  {i}/{len(todo)} done, {(time.time() - t_start) / 60:.1f} min elapsed, "
                          f"about {eta:.0f} min to go")
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C): finishing the requests already running, skipping the rest...")
        pool.shutdown(wait=True, cancel_futures=True)
        print("Stopped. Run the same command again later to continue where it left off.")
    finally:
        pool.shutdown(wait=True)

    if results:
        log = pd.DataFrame(results)
        log.to_csv(log_csv, mode="a", header=not log_csv.exists(), index=False)
        print(f"\nwall time: {(time.time() - t_start) / 60:.1f} min for {len(results)} patches")
        print(f"mean seconds per patch (single request): {log.seconds.mean():.1f}")
        print(f"mean file size: {log.MB.mean():.2f} MB")
        for col in [c for c in log.columns if c.endswith("nodata")]:
            print(f"mean {col} fraction: {log[col].mean():.2%}")
    if failed:
        print(f"\n{len(failed)} patches failed. Run the same command again to retry only those.")


# -----------------------------------------------------------------------------
# Step 3: previews
# -----------------------------------------------------------------------------
def to_rgb(img, bands):
    x = np.stack([img[BANDS.index(b)] for b in bands], axis=-1).astype(np.float32)
    valid = x.sum(axis=-1) > 0
    lo, hi = np.percentile(x[valid], [2, 98]) if valid.any() else (0.0, 1.0)
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def cmd_preview(limit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(PATCH_DIR.glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    if limit:
        files = files[:limit]
    cmap = ListedColormap(["#1a5e1a", "#ff0000", "#bbbbbb"])
    for f in files:
        d = np.load(f)
        post, pre, label = d["post"], d["pre"], d["label"]
        lab_vis = np.select([label == 0, label == 1], [0, 1], default=2)
        valid = (label != 255).sum()
        share = (label == 1).sum() / valid if valid else 0.0

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        panels = [(to_rgb(pre, ["B4", "B3", "B2"]), "2019 natural colour"),
                  (to_rgb(post, ["B4", "B3", "B2"]), "2024 natural colour"),
                  (to_rgb(post, ["B12", "B8", "B4"]), "2024 false colour")]
        for ax, (im, title) in zip(axes[:3], panels):
            ax.imshow(im)
            ax.set_title(title)
        axes[3].imshow(lab_vis, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        axes[3].set_title(f"label (red = cleared, {share:.1%} of valid)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f.stem)
        fig.savefig(PREVIEW_DIR / f"{f.stem}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {len(files)} previews to {PREVIEW_DIR}")


# -----------------------------------------------------------------------------
# Step 5: check every downloaded file, summarise the dataset, compute band statistics
# -----------------------------------------------------------------------------
DATASET_CSV = MANIFEST_DIR / "patches_downloaded.csv"
BAND_STATS_JSON = MANIFEST_DIR / "band_stats.json"
STATS_STEP = 4  # use every 4th row and column (1/16 of pixels) for band statistics


def check_one(path, want_stats):
    try:
        with np.load(path) as d:
            post, pre, label = d["post"], d["pre"], d["label"]
        shapes_ok = (post.shape == (len(BANDS), PATCH_PX, PATCH_PX) and post.dtype == np.uint16 and
                     pre.shape == (len(BANDS), PATCH_PX, PATCH_PX) and pre.dtype == np.uint16 and
                     label.shape == (PATCH_PX, PATCH_PX) and label.dtype == np.uint8)
        values_ok = bool(np.isin(label, (0, 1, 255)).all())
        row = {"patch_id": path.stem, "ok": shapes_ok and values_ok, "error": "",
               "forest_px": int((label == 0).sum()), "cleared_px": int((label == 1).sum()),
               "ignore_px": int((label == 255).sum()),
               "post_nodata_px": int((post == 0).all(axis=0).sum()),
               "pre_nodata_px": int((pre == 0).all(axis=0).sum()),
               "MB": path.stat().st_size / 1e6}
        sums = None
        if want_stats and row["ok"]:
            sums = {}
            for name, img in (("post", post), ("pre", pre)):
                x = img[:, ::STATS_STEP, ::STATS_STEP].reshape(len(BANDS), -1).astype(np.float64)
                x = x[:, (x > 0).any(axis=0)]          # drop no-data pixels
                sums[name] = (x.shape[1], x.sum(axis=1), (x ** 2).sum(axis=1))
        return row, sums
    except Exception as e:  # unreadable or truncated file
        return {"patch_id": path.stem, "ok": False, "error": str(e)[:200]}, None


def cmd_summary():
    import json

    manifest = pd.read_csv(GRID_CSV).set_index("patch_id")
    files = [f for f in sorted(PATCH_DIR.glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    print(f"checking {len(files)} files...")
    split_of = manifest.split.to_dict()
    region_of = manifest.region.to_dict()
    is_train = {pid for pid in manifest.index
                if region_of[pid] == "amazon" and split_of[pid] == "train"}

    rows, totals = [], {}
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for row, sums in pool.map(lambda f: check_one(f, f.stem in is_train), files):
            rows.append(row)
            if sums:
                for name, (n, s, s2) in sums.items():
                    t = totals.setdefault(name, [0, np.zeros(len(BANDS)), np.zeros(len(BANDS))])
                    t[0] += n
                    t[1] += s
                    t[2] += s2
    df = pd.DataFrame(rows)

    bad = df[~df.ok]
    unknown = df[~df.patch_id.isin(manifest.index)]
    df = df[df.ok & df.patch_id.isin(manifest.index)].copy()
    df["region"] = df.patch_id.map(region_of)
    df["split"] = df.patch_id.map(split_of)
    df["selected"] = df.patch_id.map(manifest.selected.to_dict()).astype(bool)
    df = df[df.selected]
    count_cols = ["forest_px", "cleared_px", "ignore_px", "post_nodata_px", "pre_nodata_px"]
    df[count_cols] = df[count_cols].astype(int)

    print("\nregion  split   patches  cleared share (of valid px)  valid px  2024 no-data  size GB")
    for (region, split), g in df.groupby(["region", "split"]):
        valid = g.forest_px.sum() + g.cleared_px.sum()
        total = len(g) * PATCH_PX * PATCH_PX
        print(f"{region:7s} {split:5s} {len(g):8d} {g.cleared_px.sum() / max(valid, 1):22.1%} "
              f"{valid / total:15.1%} {g.post_nodata_px.sum() / total:12.2%} {g.MB.sum() / 1e3:8.2f}")
    print(f"\ntotal: {len(df)} patches, {df.MB.sum() / 1e3:.2f} GB")
    no_valid = int(((df.forest_px + df.cleared_px) == 0).sum())
    print(f"patches with no valid label pixels: {no_valid}")
    print(f"unreadable or malformed files: {len(bad)}")
    for r in bad.itertuples():
        print(f"  {r.patch_id}: {r.error or 'wrong shape, dtype or label values'}")
        src = PATCH_DIR / f"{r.patch_id}.npz"
        os.replace(src, src.with_name(src.name + ".bad"))   # set aside so 'download' fetches it again
    if len(bad):
        print("  these were renamed to .npz.bad. Re-fetch only them with:  "
              "python src/dataprep/patches.py repair   (then run 'summary' again)")
    if len(unknown):
        print(f"files not in the manifest (ignored): {len(unknown)}")

    keep = ["patch_id", "region", "split", "forest_px", "cleared_px", "ignore_px",
            "post_nodata_px", "pre_nodata_px"]
    df[keep].sort_values("patch_id").to_csv(DATASET_CSV, index=False)
    print(f"\nsaved dataset index: {DATASET_CSV}")

    if totals:
        stats = {"bands": BANDS, "source": "amazon train split, every 4th pixel, no-data excluded"}
        for name, (n, s, s2) in totals.items():
            mean = s / n
            std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
            stats[name] = {"mean": [round(float(v), 2) for v in mean],
                           "std": [round(float(v), 2) for v in std], "pixels": int(n)}
        BAND_STATS_JSON.write_text(json.dumps(stats, indent=2))
        print(f"saved band statistics (for normalisation): {BAND_STATS_JSON}")
        print("2024 band means:", dict(zip(BANDS, [round(v) for v in stats["post"]["mean"]])))


def cmd_repair():
    """Re-download only the patches that 'summary' set aside as .npz.bad."""
    bad_ids = [f.name[:-len(".npz.bad")] for f in PATCH_DIR.glob("*.npz.bad")]
    if not bad_ids:
        print("nothing to repair")
        return
    manifest = pd.read_csv(GRID_CSV)
    run_downloads(manifest[manifest.patch_id.isin(bad_ids)])
    for pid in bad_ids:
        if (PATCH_DIR / f"{pid}.npz").exists():
            (PATCH_DIR / f"{pid}.npz.bad").unlink()


# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Patch grid and downloads")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("grid", help="build grid, label statistics, spatial splits and selection")
    sub.add_parser("pilot", help="download the small pilot set")
    dl = sub.add_parser("download", help="download all selected patches")
    dl.add_argument("--limit", type=int, default=0,
                    help="only download this many of the remaining patches (a balanced random mix)")
    pv = sub.add_parser("preview", help="save PNG previews of downloaded patches")
    pv.add_argument("--limit", type=int, default=0, help="only preview the first N patches")
    sub.add_parser("summary", help="check downloaded files, summarise the dataset, compute band statistics")
    sub.add_parser("repair", help="re-download only the files that 'summary' found broken")
    args = parser.parse_args()

    if args.command == "grid":
        cmd_grid()
    elif args.command == "pilot":
        run_downloads(load_manifest("pilot"))
    elif args.command == "download":
        run_downloads(load_manifest("selected"), limit=args.limit)
    elif args.command == "preview":
        cmd_preview(args.limit)
    elif args.command == "summary":
        cmd_summary()
    elif args.command == "repair":
        cmd_repair()


if __name__ == "__main__":
    main()
