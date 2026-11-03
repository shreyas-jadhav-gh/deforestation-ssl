"""
Quick demo: run the trained models on a few Sentinel-2 test patches and show what they predict.

Needs only the small sample in sample_data/ (16 test patches, 19 MB, included in the repository)
and the model files from the GitHub release, unzipped in the repository root (see README).
Runs on a laptop CPU in a minute or two. From the repository root:

    python src/demo.py                 # every seed-0 model trained with 10% of the labels
    python src/demo.py --frac 1        # ... trained with all labels (also 0.01, 0.05, 0.25)
    python src/demo.py --ckpt results/kaggle_cd/results/checkpoints/cd_scratch_cd_lr0.0002_f0.1_s0.pt
    python src/demo.py --data data     # the full test sets, if you downloaded the dataset

For each model it prints IoU, precision and recall of the 'cleared' class and the share of
the scored area it calls cleared, separately for the Amazon and the Congo patches. It uses the
probability threshold that was tuned on Amazon validation when the model was trained (read from
the results CSVs in results/). It also saves results/demo/demo_f<frac>.png with the 2019 and
2024 images, the label and each model's prediction, coloured by outcome.

Sixteen patches are only a sanity check; the numbers in the report are on the full test sets
(308 Amazon and 766 Congo patches).
"""

import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from make_figures import INITS, INK, INK2, SURFACE  # noqa: E402
from predict_examples import OUTCOME_CMAP, OUTCOMES, load_model, outcome_map, predict, rgb  # noqa: E402

RESULTS = {"full": C.REPO / "results" / "kaggle_full" / "results",
           "cd": C.REPO / "results" / "kaggle_cd" / "results"}
SHORT = {"scratch": "Random init", "ssl4eo": "SSL4EO-S12", "moco_both": "MoCo ours\n(Amazon+Congo)",
         "moco_amazon": "MoCo ours\n(Amazon)"}
# e.g. full_scratch_lr0.0002_f0.1_s0  or  cd_moco_both_cd_lr0.0005_f1_s0
NAME = re.compile(r"^(full|cd)_(.+?)(?:_cd)?_lr([0-9.e-]+)_f([0-9.]+)_s(\d+)$")


def describe(ckpt):
    """Run settings (tag, init, lr, frac, seed) read from a checkpoint's file name."""
    m = NAME.match(Path(ckpt).stem)
    if not m:
        raise SystemExit(f"cannot read the run settings from the file name {Path(ckpt).name}")
    tag, init, lr, frac, seed = m.groups()
    return {"tag": tag, "init": init, "lr": float(lr), "frac": float(frac), "seed": int(seed)}


def tuned_threshold(run):
    """The threshold chosen on Amazon validation for this run, from the results CSV (else 0.5)."""
    csv = RESULTS[run["tag"]] / f"{run['tag']}_all.csv"
    if not csv.exists():
        print(f"  ({csv.name} not found: using threshold 0.5)")
        return 0.5
    runs = pd.read_csv(csv)
    row = runs[(runs.tag == run["tag"]) & (runs.init == run["init"]) & (runs.seed == run["seed"])
               & ((runs.lr - run["lr"]).abs() < 1e-12) & ((runs.frac - run["frac"]).abs() < 1e-9)]
    return float(row.threshold.iloc[0]) if len(row) else 0.5


def find_models(frac):
    """Seed-0 checkpoints for one label fraction: single-date first, then change detection."""
    order = list(INITS)
    found = []
    for tag in ("full", "cd"):
        hits = (RESULTS[tag] / "checkpoints").glob(f"{tag}_*_f{frac:g}_s0.pt")
        found += sorted(hits, key=lambda p: order.index(describe(p)["init"])
                        if describe(p)["init"] in order else len(order))
    return found


def scores(prob, label, threshold):
    """Cleared-class metrics over all scored pixels of a set of patches."""
    scored = label != C.IGNORE
    pred = (prob >= threshold) & scored
    cleared = (label == 1) & scored
    tp = np.sum(pred & cleared)
    fp = np.sum(pred & ~cleared)
    fn = np.sum(~pred & cleared)
    tn = np.sum(scored) - tp - fp - fn
    m = C.metrics_from_counts(tp, fp, fn, tn)
    m["called_cleared"] = (tp + fp) / max(np.sum(scored), 1)
    m["truly_cleared"] = (tp + fn) / max(np.sum(scored), 1)
    return m


def draw(data, rows, models, preds):
    """One row per patch: 2019, 2024, label, then every model's prediction coloured by outcome."""
    n_cols = 3 + len(models)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.6 * n_cols, 1.7 * len(rows) + 1.0),
                             squeeze=False)
    for r, i in enumerate(rows):
        label = data["label"][i]
        panels = [(rgb(data["pre"][i], ["B4", "B3", "B2"]), "2019 image"),
                  (rgb(data["post"][i], ["B4", "B3", "B2"]), "2024 image"),
                  (outcome_map((label == 1).astype(float), label, 0.5), "label")]
        for m in models:
            panels.append((outcome_map(preds[m["key"]][i], label, m["threshold"]), m["column"]))
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c]
            if img.ndim == 2:
                ax.imshow(img, cmap=OUTCOME_CMAP, vmin=0, vmax=len(OUTCOMES) - 1, interpolation="nearest")
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=7, color=INK)
        axes[r, 0].set_ylabel(data["patch_id"][i], fontsize=7, color=INK2)
    handles = [Patch(facecolor=c, edgecolor=INK2, linewidth=0.4, label=n) for n, c in OUTCOMES]
    fig.legend(handles=handles, loc="lower center", ncol=len(OUTCOMES), fontsize=8, frameon=False)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(C.REPO / "sample_data"),
                   help="folder with manifests/patches_downloaded.csv and patches/ (default: sample_data)")
    p.add_argument("--frac", type=float, default=0.1, help="label fraction of the models to run")
    p.add_argument("--ckpt", nargs="+", help="run these checkpoint files instead")
    p.add_argument("--show", type=int, default=3, help="patches per region drawn in the figure")
    p.add_argument("--out", default=str(C.REPO / "results" / "demo"))
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = [Path(c) for c in args.ckpt] if args.ckpt else find_models(args.frac)
    if not ckpts:
        raise SystemExit("No model files found. Download them from the repository's Releases page and "
                         "unzip them in the repository root (see 'Model files' in the README).")

    index = C.labeled_index(args.data)
    index = index[index.split == "test"].reset_index(drop=True)
    print(f"{len(index)} test patches from {args.data}: "
          + ", ".join(f"{n} {r}" for r, n in index.region.value_counts().sort_index().items()))
    data = C.load_patches(index.path, ["post", "pre", "label"], name="patches")
    data["patch_id"] = list(index.patch_id)

    models, preds, table = [], {}, []
    for ckpt in ckpts:
        run = describe(ckpt)
        run["threshold"] = tuned_threshold(run)
        two = run["tag"] == "cd"
        run["key"] = ckpt.stem
        run["column"] = ("2019+2024: " if two else "2024 only: ") + SHORT.get(run["init"], run["init"])
        run["column"] = run["column"].replace(": ", ":\n", 1)
        t0 = time.time()
        model, two_dates = load_model(ckpt, device)
        d = data if two_dates else {"post": data["post"], "label": data["label"]}
        preds[run["key"]] = predict(model, d, device)
        print(f"  {ckpt.name}: {time.time() - t0:.0f} s")
        row = {"input": "2019 + 2024" if two_dates else "2024 only",
               "encoder start": INITS.get(run["init"], (run["init"],))[0],
               "labels": f"{run['frac']:.0%}", "threshold": round(run["threshold"], 3)}
        for region in ("amazon", "congo"):
            sel = (index.region == region).to_numpy()
            if sel.any():
                m = scores(preds[run["key"]][sel], data["label"][sel], run["threshold"])
                row[f"{region} IoU"] = m["iou"]
                row[f"{region} precision"] = m["precision"]
                row[f"{region} recall"] = m["recall"]
                row[f"{region} called cleared"] = f"{m['called_cleared']:.1%} (true {m['truly_cleared']:.1%})"
        models.append(run)
        table.append(row)

    print("\nCleared class on these patches, threshold tuned on Amazon validation:")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(table).to_string(index=False))

    rows = []
    for region in ("amazon", "congo"):
        rows += list(index.index[index.region == region][: args.show])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"demo_f{args.frac:g}" if not args.ckpt else "demo_selected"
    fig = draw(data, rows, models, preds)
    fig.savefig(out / f"{name}.png", dpi=150)
    print(f"\nsaved {out / (name + '.png')}")


if __name__ == "__main__":
    main()
