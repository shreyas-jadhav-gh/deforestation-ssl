"""
Pack the downloaded patches into zip files for uploading to Kaggle.

Run from the repository root with the 'defor' environment active:

    python src/dataprep/pack.py

Creates, in data/upload/:
    labeled.zip            patches/*.npz (the labeled set) + manifests/ (index, splits, band stats)
    unlabeled_part1.zip    unlabeled/*.npz, first half  + manifests/ (unlabeled index)
    unlabeled_part2.zip    unlabeled/*.npz, second half
Each zip stays under 4 GB, so browser uploads and unzip tools have no trouble with it.
The .npz files are already compressed, so the zips just store them (fast).
Every zip is re-opened and fully checked (CRC of every file) after writing.
"""

import os
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MANIFESTS = DATA / "manifests"
OUT = DATA / "upload"


def pack(zip_path, members):
    """members: list of (file on disk, name inside the zip)."""
    missing = [str(p) for p, _ in members if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} files are missing, e.g. {missing[:3]}. "
                         "Run the 'summary' commands first.")
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    total_gb = sum(p.stat().st_size for p, _ in members) / 1e9
    print(f"\nwriting {zip_path.name}: {len(members)} files, {total_gb:.2f} GB")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, (path, arcname) in enumerate(members, 1):
            z.write(path, arcname)
            if i % 1000 == 0:
                print(f"  {i}/{len(members)}")
    print("  checking...")
    with zipfile.ZipFile(tmp) as z:
        names = z.namelist()
        bad = z.testzip()            # reads every file and checks its CRC
    if len(names) != len(members) or bad is not None:
        raise SystemExit(f"  check FAILED for {zip_path.name} (bad entry: {bad}); nothing was saved")
    os.replace(tmp, zip_path)
    print(f"  OK: {zip_path} ({zip_path.stat().st_size / 1e9:.2f} GB)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    labeled = pd.read_csv(MANIFESTS / "patches_downloaded.csv")
    members = [(DATA / "patches" / f"{pid}.npz", f"patches/{pid}.npz") for pid in labeled.patch_id]
    members += [(MANIFESTS / name, f"manifests/{name}")
                for name in ("patches_downloaded.csv", "patches_all.csv", "band_stats.json")]
    pack(OUT / "labeled.zip", members)

    unlabeled = pd.read_csv(MANIFESTS / "unlabeled_downloaded.csv").sort_values("patch_id")
    ids = list(unlabeled.patch_id)
    half = (len(ids) + 1) // 2
    for part, chunk in ((1, ids[:half]), (2, ids[half:])):
        members = [(DATA / "unlabeled" / f"{pid}.npz", f"unlabeled/{pid}.npz") for pid in chunk]
        if part == 1:
            members += [(MANIFESTS / name, f"manifests/{name}")
                        for name in ("unlabeled_downloaded.csv", "unlabeled_all.csv")]
        pack(OUT / f"unlabeled_part{part}.zip", members)

    print("\nAll zips written and checked. Upload the files in", OUT)


if __name__ == "__main__":
    main()
