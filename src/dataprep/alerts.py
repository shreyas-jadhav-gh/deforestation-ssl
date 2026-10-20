"""
RADD forest-disturbance alerts (Sentinel-1 radar, Wageningen University; CC BY 4.0) on the exact
10 m grid of every downloaded TEST patch, as an independent reference: radar, a different sensor
from both the labels (Hansen, Landsat optical) and the model input (Sentinel-2).

Run from the repository root with the 'defor' environment active:

    python src/dataprep/alerts.py download     # one small request per test patch
    python src/dataprep/alerts.py summary      # how well Hansen labels and RADD alerts agree

Each patch is saved as data/alerts/<patch_id>.npz with
    date : (256, 256) uint16  alert date as YYDOY (21152 = day 152 of 2021); 0 = no alert;
                              65535 = outside RADD's forest baseline (no alerts are made there)
    conf : (256, 256) uint8   2 = low confidence, 3 = high confidence, 0 = no alert
"""

import argparse
import concurrent.futures as cf
import os
import time

import ee
import numpy as np
import pandas as pd

import patches as P   # shared settings and helpers (same folder)

RADD = "projects/radar-wur/raddalert/v1"
GEOGRAPHY = {"amazon": "sa", "congo": "africa"}
OUTSIDE = 65535
OUT_DIR = P.ROOT / "data" / "alerts"
LOG_CSV = P.MANIFEST_DIR / "alerts_download_log.csv"
WINDOW = (20001, 23366)          # alerts dated 2020-01-01 ... 2023-12-31, the label window


def radd_image(geography):
    col = ee.ImageCollection(RADD)
    alert = ee.Image(col.filterMetadata("layer", "contains", "alert")
                     .filterMetadata("geography", "equals", geography)
                     .sort("system:time_end", False).first())
    baseline = ee.Image(col.filterMetadata("layer", "contains", "forest_baseline")
                        .filterMetadata("geography", "equals", geography)
                        .sort("layer", False).first())
    inside = baseline.select(0).unmask(0).gt(0)
    date = alert.select("Date").unmask(0).toUint16().where(inside.Not(), OUTSIDE)
    conf = alert.select("Alert").unmask(0).toUint16().where(inside.Not(), 0)
    return date.rename("date").addBands(conf.rename("conf")), baseline


IMAGES = {}


def download_one(p):
    t0 = time.time()
    crs = P.REGIONS[p.region]["crs"]
    grid = P.make_grid(p.x0, p.y_top, crs, P.PATCH_PX, P.PATCH_PX, P.SCALE_M)
    arr = P.compute_pixels(IMAGES[p.region], grid)
    date = arr["date"].astype(np.uint16)
    conf = arr["conf"].astype(np.uint8)
    tmp = OUT_DIR / f"{p.patch_id}.tmp.npz"
    final = OUT_DIR / f"{p.patch_id}.npz"
    np.savez_compressed(tmp, date=date, conf=conf)
    os.replace(tmp, final)
    inside = date != OUTSIDE
    return {"patch_id": p.patch_id, "seconds": round(time.time() - t0, 1),
            "MB": round(final.stat().st_size / 1e6, 3),
            "outside_nodata": float((~inside).mean()),
            "alert_share": float(((date > 0) & inside).mean())}


def test_patches():
    downloaded = pd.read_csv(P.MANIFEST_DIR / "patches_downloaded.csv")
    test_ids = set(downloaded[downloaded.split == "test"].patch_id)
    grid = pd.read_csv(P.GRID_CSV)
    return grid[grid.patch_id.isin(test_ids)]


def cmd_download():
    P.init_ee()
    for region, geo in GEOGRAPHY.items():
        image, baseline = radd_image(geo)
        IMAGES[region] = image
        print(f"{region}: RADD geography '{geo}', forest baseline layer "
              f"'{baseline.get('layer').getInfo()}'")
    P.run_downloads(test_patches(), worker=download_one, out_dir=OUT_DIR, log_csv=LOG_CSV)


def cmd_summary():
    """Agreement between Hansen labels (cleared 2020-2023) and RADD alerts in the same window,
    over pixels that are scored (label 0/1) and inside RADD's forest baseline."""
    downloaded = pd.read_csv(P.MANIFEST_DIR / "patches_downloaded.csv")
    tests = downloaded[downloaded.split == "test"]

    def one(row):
        a = OUT_DIR / f"{row.patch_id}.npz"
        if not a.exists():
            return None
        with np.load(a) as d:
            date = d["date"]
        with np.load(P.PATCH_DIR / f"{row.patch_id}.npz") as d:
            label = d["label"]
        scored = (label != 255) & (date != OUTSIDE)
        hansen = (label == 1) & scored
        radd = (date >= WINDOW[0]) & (date <= WINDOW[1]) & scored
        return {"region": row.region, "scored": int(scored.sum()),
                "outside": int(((label != 255) & (date == OUTSIDE)).sum()),
                "both": int((hansen & radd).sum()), "hansen_only": int((hansen & ~radd).sum()),
                "radd_only": int((radd & ~hansen).sum())}

    with cf.ThreadPoolExecutor(max_workers=P.WORKERS) as pool:
        rows = [r for r in pool.map(one, tests.itertuples(index=False)) if r]
    if not rows:
        raise SystemExit("no alert files yet; run the download first")
    df = pd.DataFrame(rows).groupby("region").sum()
    print(f"{len(rows)} test patches with alerts\n")
    print("region  scored px inside RADD forest  (outside)  Hansen&RADD  Hansen only  RADD only"
          "  IoU   Hansen pixels with a RADD alert   RADD pixels Hansen labels cleared")
    for region, r in df.iterrows():
        union = r.both + r.hansen_only + r.radd_only
        print(f"{region:7s} {r.scored:28,d} {r.outside:10,d} {r.both:12,d} {r.hansen_only:12,d} "
              f"{r.radd_only:10,d} {r.both / max(union, 1):5.2f} "
              f"{r.both / max(r.both + r.hansen_only, 1):33.1%} "
              f"{r.both / max(r.both + r.radd_only, 1):35.1%}")


def main():
    p = argparse.ArgumentParser(description="RADD alerts for the test patches")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("summary")
    args = p.parse_args()
    cmd_download() if args.command == "download" else cmd_summary()


if __name__ == "__main__":
    main()
