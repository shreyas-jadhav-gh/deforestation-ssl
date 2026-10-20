"""
Do the models' 'false alarms' hide real clearing? Compare every seed-0 model's predictions on the
test patches with the Hansen labels AND with independent RADD radar alerts (data/alerts/).

    python src/validate_alerts.py --results results/kaggle_full/results
    python src/validate_alerts.py --results results/kaggle_full/results results/kaggle_cd/results --tags full cd

Only pixels that are scored (label 0 or 1) and inside RADD's forest baseline count. Per region:
  iou_hansen     IoU against the Hansen labels (the usual score, restricted to this area)
  iou_radd       IoU against RADD alerts dated 2020-2023 (the label window)
  pred_share     share of the area the model calls cleared
  fa_confirmed   share of the model's false alarms (predicted cleared, Hansen says forest) that
                 RADD flagged between Jan 2020 and Sep 2024 (i.e. before the end of the 2024 image)
  base_rate      the same share for ALL Hansen-forest pixels: what a random pick would give
  fa_lift        fa_confirmed / base_rate: how many times more often RADD backs a false alarm
                 than a random forest pixel (well above 1 = many 'false alarms' are real clearing)
A first row, 'Hansen labels', scores the labels themselves against RADD for reference.
Writes <first results folder>/figures/table_radd_validation.csv. Runs on the laptop (GPU if available).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from predict_examples import load_model  # noqa: E402
from train_seg import batch_input, to_input  # noqa: E402

OUTSIDE = 65535
LABEL_WINDOW = (20001, 23366)     # 2020-01-01 ... 2023-12-31 (YYDOY)
CONFIRM_WINDOW = (20001, 24274)   # 2020-01-01 ... 2024-09-30, up to the end of the 2024 image
ALERT_DIR = C.REPO / "data" / "alerts"


def reference_masks(label, date):
    area = (label != C.IGNORE) & (date != OUTSIDE)
    hansen = (label == 1) & area
    radd = (date >= LABEL_WINDOW[0]) & (date <= LABEL_WINDOW[1]) & area
    confirm = (date >= CONFIRM_WINDOW[0]) & (date <= CONFIRM_WINDOW[1]) & area
    return area, hansen, radd, confirm


def counts_for(pred, label, date):
    """[area, tp_h, fp_h, fn_h, tp_r, fp_r, fn_r, false_alarms, false_alarms_confirmed,
        forest, forest_confirmed] for one patch."""
    area, hansen, radd, confirm = reference_masks(label, date)
    p = pred & area
    fa = p & (label == 0)
    forest = (label == 0) & area
    return np.array([area.sum(), (p & hansen).sum(), (p & ~hansen).sum(), (~p & hansen & area).sum(),
                     (p & radd).sum(), (p & ~radd).sum(), (~p & radd & area).sum(),
                     fa.sum(), (fa & confirm).sum(), forest.sum(), (forest & confirm).sum()],
                    dtype=np.int64)


def to_row(name, region, c, extra):
    area, tp_h, fp_h, fn_h, tp_r, fp_r, fn_r, fa, fa_c, forest, forest_c = c.astype(float)
    base = forest_c / max(forest, 1)
    conf = fa_c / fa if fa > 0 else float("nan")
    return {"region": region, "model": name, **extra,
            "iou_hansen": tp_h / max(tp_h + fp_h + fn_h, 1),
            "iou_radd": tp_r / max(tp_r + fp_r + fn_r, 1),
            "pred_share": (tp_h + fp_h) / max(area, 1),
            "fa_confirmed": conf, "base_rate": base,
            "fa_lift": conf / base if base > 0 else float("nan")}


@torch.no_grad()
def model_counts(model, data, dates, regions, threshold, device, batch_size=8):
    acc = {r: np.zeros(11, dtype=np.int64) for r in set(regions)}
    for s in range(0, len(dates), batch_size):
        x = to_input(batch_input(data, slice(s, s + batch_size)), device)
        prob = model(x).float().softmax(1)[:, 1].cpu().numpy()
        for j, pr in enumerate(prob):
            i = s + j
            acc[regions[i]] += counts_for(pr >= threshold, data["label"][i], dates[i])
    return acc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, nargs="+",
                   help="downloaded results folder(s), each with <tag>_all.csv and checkpoints/")
    p.add_argument("--fracs", nargs="+", type=float, default=[0.1, 1.0])
    p.add_argument("--tags", nargs="+", default=["full"], help="which *_all.csv files to use")
    args = p.parse_args()
    folders = [Path(f) for f in args.results]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = []
    for folder in folders:
        for tag in args.tags:
            f = folder / f"{tag}_all.csv"
            if f.exists():
                r = pd.read_csv(f)
                runs.append(r[(r.tag == tag) & (r.seed == 0)].assign(folder=str(folder)))
    if not runs:
        raise SystemExit(f"none of {[t + '_all.csv' for t in args.tags]} found in {args.results}")
    runs = pd.concat(runs, ignore_index=True)
    runs["input"] = runs["input"].fillna("post") if "input" in runs else "post"
    runs = runs[runs.frac.round(6).isin([round(f, 6) for f in args.fracs])]

    models = []
    for row in runs.itertuples():
        cd = "_cd" if row.input == "both" else ""
        hits = sorted((Path(row.folder) / "checkpoints").glob(f"{row.tag}_{row.init}{cd}_lr*_f{row.frac:g}_s0.pt"))
        if hits:
            models.append((row, hits[0]))
    if not models:
        raise SystemExit(f"no seed-0 checkpoints for fractions {args.fracs} in {args.results}")
    need_pre = any(row.input == "both" for row, _ in models)

    index = C.labeled_index()
    index = index[index.split == "test"].reset_index(drop=True)
    have = [pid for pid in index.patch_id if (ALERT_DIR / f"{pid}.npz").exists()]
    index = index[index.patch_id.isin(have)].reset_index(drop=True)
    print(f"{len(index)} test patches with RADD alerts; {len(models)} models")
    data = C.load_patches(index.path, ["post", "pre", "label"] if need_pre else ["post", "label"],
                          name="test patches")
    dates = np.stack([np.load(ALERT_DIR / f"{pid}.npz")["date"] for pid in index.patch_id])
    regions = index.region.tolist()

    rows = []
    ref = {r: np.zeros(11, dtype=np.int64) for r in set(regions)}
    for i in range(len(dates)):     # the Hansen labels themselves, as a 'model'
        ref[regions[i]] += counts_for(data["label"][i] == 1, data["label"][i], dates[i])
    for region, c in ref.items():
        rows.append(to_row("Hansen labels", region, c, {"input": "", "frac": np.nan, "threshold": np.nan}))

    for row, ckpt in models:
        model, two_dates = load_model(ckpt, device)
        d = data if two_dates else {"post": data["post"], "label": data["label"]}
        print(f"  {ckpt.stem} (threshold {row.threshold:.3f})")
        for region, c in model_counts(model, d, dates, regions, row.threshold, device).items():
            rows.append(to_row(row.init, region, c,
                               {"input": row.input, "frac": row.frac, "threshold": row.threshold}))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows).sort_values(["region", "input", "frac", "model"], na_position="first")
    out = folders[0] / "figures"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "table_radd_validation.csv", index=False)
    show = table.copy()
    for col in ["iou_hansen", "iou_radd", "threshold"]:
        show[col] = show[col].round(3)
    for col in ["pred_share", "fa_confirmed", "base_rate"]:
        show[col] = show[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "-")
    show["fa_lift"] = show["fa_lift"].map(lambda v: f"{v:.1f}x" if pd.notna(v) else "-")
    show = show.fillna("")
    print(show.to_string(index=False))
    print(f"\nsaved {out / 'table_radd_validation.csv'}")


if __name__ == "__main__":
    main()
