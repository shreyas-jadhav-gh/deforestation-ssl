"""
Unlabeled Sentinel-2 patches for self-supervised (MoCo) pretraining.

The patches come from two NEW boxes next to the labeled ones, so no image used for
validation or testing is ever seen during pretraining:
    amazon_u : Rondonia/Acre/Amazonas frontier, west of the labeled Amazon box
    congo_u  : Maniema, east of the labeled Congo box
Each box stops 0.1 degrees (about 11 km) short of the labeled box, as a spatial buffer.

Every patch is one cloud-masked dry-season (June-September) median, built exactly like
the labeled images, for a year picked at random from 2020-2024 (more variety for SSL).

Run from the repository root with the 'defor' environment active:

    python src/dataprep/unlabeled.py grid                 # 1. grid + random selection (no Earth Engine needed)
    python src/dataprep/unlabeled.py download --limit 1000  # 2. download a first batch
    python src/dataprep/unlabeled.py download             # 3. download the rest
    python src/dataprep/unlabeled.py summary              # 4. check every file

Each patch is saved as data/unlabeled/<patch_id>.npz containing:
    image : (13, 256, 256) uint16, bands B1 ... B12 (same order as the labeled patches)
    year  : the composite year
"""

import argparse
import concurrent.futures as cf
import os
import time

import ee
import numpy as np
import pandas as pd

import patches as P   # shared settings and helpers (same folder)

UNLABELED_REGIONS = {
    "amazon_u": {"bbox": (-66.0, -10.5, -64.1, -8.5), "crs": "EPSG:32720"},  # UTM 20S
    "congo_u":  {"bbox": (26.1, -3.0, 28.0, -1.0),    "crs": "EPSG:32735"},  # UTM 35S
}
P.REGIONS.update(UNLABELED_REGIONS)   # lets the shared grid builder use these boxes

YEARS = [2020, 2021, 2022, 2023, 2024]
PER_REGION = 4000                     # patches selected per box
OUT_DIR = P.ROOT / "data" / "unlabeled"
GRID_CSV = P.MANIFEST_DIR / "unlabeled_all.csv"
LOG_CSV = P.MANIFEST_DIR / "unlabeled_download_log.csv"
INDEX_CSV = P.MANIFEST_DIR / "unlabeled_downloaded.csv"


def cmd_grid():
    P.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(P.SEED + 1)
    frames = []
    for name in UNLABELED_REGIONS:
        df = P.build_grid(name)
        df["selected"] = False
        take = df.sample(n=min(PER_REGION, len(df)), random_state=P.SEED + 1).index
        df.loc[take, "selected"] = True
        df["year"] = rng.choice(YEARS, size=len(df))
        frames.append(df)
        print(f"{name}: {len(df)} patches in grid, {int(df.selected.sum())} selected")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(GRID_CSV, index=False)
    sel = df[df.selected]
    print("selected per year:", sel.year.value_counts().sort_index().to_dict())
    print(f"saved {GRID_CSV}")


def download_one(p):
    t0 = time.time()
    crs = P.REGIONS[p.region]["crs"]
    geom = ee.Geometry.Rectangle([float(p.x0), float(p.y_top - P.PATCH_M),
                                  float(p.x0 + P.PATCH_M), float(p.y_top)],
                                 proj=crs, geodesic=False)
    image = P.dry_season_composite(geom, int(p.year)).unmask(0).toUint16()
    grid = P.make_grid(p.x0, p.y_top, crs, P.PATCH_PX, P.PATCH_PX, P.SCALE_M)
    arr = P.compute_pixels(image, grid)
    img = np.stack([arr[b] for b in P.BANDS]).astype(np.uint16)

    tmp = OUT_DIR / f"{p.patch_id}.tmp.npz"
    final = OUT_DIR / f"{p.patch_id}.npz"
    np.savez_compressed(tmp, image=img, year=np.int16(p.year))
    os.replace(tmp, final)
    return {"patch_id": p.patch_id, "seconds": round(time.time() - t0, 1),
            "MB": round(final.stat().st_size / 1e6, 2),
            "image_nodata": float((img == 0).all(axis=0).mean())}


def cmd_download(limit):
    if not GRID_CSV.exists():
        raise SystemExit("No grid yet. Run:  python src/dataprep/unlabeled.py grid")
    df = pd.read_csv(GRID_CSV)
    P.run_downloads(df[df.selected.astype(bool)], limit=limit, worker=download_one,
                    out_dir=OUT_DIR, log_csv=LOG_CSV)


def check_one(path):
    try:
        with np.load(path) as d:
            img, year = d["image"], int(d["year"])
        ok = img.shape == (len(P.BANDS), P.PATCH_PX, P.PATCH_PX) and img.dtype == np.uint16
        nodata = float((img == 0).all(axis=0).mean())
        return {"patch_id": path.stem, "ok": bool(ok), "year": year, "nodata": nodata,
                "MB": path.stat().st_size / 1e6, "error": ""}
    except Exception as e:
        return {"patch_id": path.stem, "ok": False, "error": str(e)[:200]}


def cmd_summary():
    grid = pd.read_csv(GRID_CSV).set_index("patch_id")
    files = [f for f in sorted(OUT_DIR.glob("*.npz")) if not f.name.endswith(".tmp.npz")]
    print(f"checking {len(files)} files...")
    with cf.ThreadPoolExecutor(max_workers=P.WORKERS) as pool:
        df = pd.DataFrame(list(pool.map(check_one, files)))
    bad = df[~df.ok]
    df = df[df.ok & df.patch_id.isin(grid.index)].copy()
    df["region"] = df.patch_id.map(grid.region.to_dict())
    mostly_empty = df.nodata > 0.5          # little clear data: not useful for pretraining
    print("\nregion     patches  mean no-data  >50% no-data  size GB")
    for region, g in df.groupby("region"):
        print(f"{region:9s} {len(g):8d} {g.nodata.mean():13.2%} {int((g.nodata > 0.5).sum()):13d} "
              f"{g.MB.sum() / 1e3:8.2f}")
    print("patches per year:", df.year.value_counts().sort_index().to_dict())
    print(f"unreadable or malformed files: {len(bad)}")
    for r in bad.itertuples():
        print(f"  {r.patch_id}: {r.error or 'wrong shape or dtype'}")
        src = OUT_DIR / f"{r.patch_id}.npz"
        os.replace(src, src.with_name(src.name + ".bad"))
    if len(bad):
        print("  these were renamed to .npz.bad and will be fetched again by the next 'download'")
    df[~mostly_empty][["patch_id", "region", "year", "nodata"]].sort_values("patch_id").to_csv(
        INDEX_CSV, index=False)
    print(f"\nsaved pretraining index ({int((~mostly_empty).sum())} patches): {INDEX_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Unlabeled patches for SSL pretraining")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("grid", help="build the grid and pick patches (no Earth Engine needed)")
    dl = sub.add_parser("download", help="download selected patches")
    dl.add_argument("--limit", type=int, default=0, help="only download this many now")
    sub.add_parser("summary", help="check downloaded files and write the pretraining index")
    args = parser.parse_args()
    if args.command == "grid":
        cmd_grid()
    elif args.command == "download":
        cmd_download(args.limit)
    elif args.command == "summary":
        cmd_summary()


if __name__ == "__main__":
    main()
