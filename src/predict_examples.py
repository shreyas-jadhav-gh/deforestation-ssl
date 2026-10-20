"""
Example prediction maps for the report: for a few test patches, show the 2019 and 2024 images,
the label, and every model's prediction coloured by outcome (found clearing / false alarm /
missed clearing). Uses the seed-0 checkpoints saved by train_seg.py and the patches on disk.

    python src/predict_examples.py --results results/kaggle_full/results --frac 0.1

Writes <results>/figures/fig_examples_f<frac>.png and a CSV listing the patches shown.
Runs on the laptop (CPU is fine; a GPU is used if available).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from make_figures import INITS, INK, INK2, SURFACE  # noqa: E402
from train_seg import SiameseDiffUnet, batch_input, to_input  # noqa: E402

# outcome colours: status-style, with a text legend (never colour alone)
OUTCOMES = [("not scored", "#ffffff"), ("forest, correct", "#e1e0d9"),
            ("clearing found", "#0ca30c"), ("false alarm", "#d03b3b"), ("clearing missed", "#fab219")]
OUTCOME_CMAP = ListedColormap([c for _, c in OUTCOMES])


def load_model(ckpt_path, device):
    state = {k: v.float() if v.is_floating_point() else v
             for k, v in torch.load(ckpt_path, map_location="cpu", weights_only=True).items()}
    unet = smp.Unet("resnet18", encoder_weights=None, in_channels=len(C.BANDS), classes=2)
    model = SiameseDiffUnet(unet) if any(k.startswith("unet.") for k in state) else unet
    model.load_state_dict(state)
    return model.to(device).eval(), isinstance(model, SiameseDiffUnet)


@torch.no_grad()
def predict(model, data, device):
    probs = []
    for i in range(len(data["label"])):
        x = to_input(batch_input(data, [i]), device)
        probs.append(model(x).float().softmax(1)[0, 1].cpu().numpy())
    return np.stack(probs)


def outcome_map(prob, label, threshold):
    pred = prob >= threshold
    out = np.ones(label.shape, dtype=np.uint8)              # forest, correct
    out[(label == 1) & pred] = 2                            # clearing found
    out[(label == 0) & pred] = 3                            # false alarm
    out[(label == 1) & ~pred] = 4                           # clearing missed
    out[label == C.IGNORE] = 0                              # not scored
    return out


def rgb(img, bands):
    x = np.stack([img[C.BANDS.index(b)] for b in bands], -1).astype(np.float32)
    valid = x.sum(-1) > 0
    lo, hi = np.percentile(x[valid], [2, 98]) if valid.any() else (0, 1)
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)


def pick_patches(index, per_patch, n_amazon, n_congo, seed):
    """Random test patches with clearing, plus the Congo patch with the most scratch false alarms."""
    rng = np.random.default_rng(seed)
    labels = {}
    for region, n, min_share in (("amazon", n_amazon, 0.05), ("congo", n_congo, 0.02)):
        df = index[(index.region == region) & (index.split == "test")]
        share = df.cleared_px / (df.forest_px + df.cleared_px).clip(lower=1)
        pool = df[share >= min_share].patch_id.tolist()
        for pid in rng.choice(pool, size=min(n, len(pool)), replace=False):
            labels[pid] = f"{region} test, random patch with clearing"
    if per_patch is not None:
        congo = per_patch[per_patch.set == "congo_test"].sort_values("fp", ascending=False)
        for pid in congo.patch_id:
            if pid not in labels:
                labels[pid] = "congo test, most false alarms by scratch"
                break
    return labels


def draw_examples(chosen, data, preds, frac):
    n_cols = 3 + len(preds)
    fig, axes = plt.subplots(len(chosen), n_cols, figsize=(1.9 * n_cols, 1.95 * len(chosen) + 0.8),
                             squeeze=False)
    for r, pid in enumerate(chosen):
        panels = [(rgb(data["pre"][r], ["B4", "B3", "B2"]), "2019"),
                  (rgb(data["post"][r], ["B4", "B3", "B2"]), "2024"),
                  (outcome_map((data["label"][r] == 1).astype(float), data["label"][r], 0.5),
                   "label\n(green = cleared)")]
        for init, (prob, thr) in preds.items():
            panels.append((outcome_map(prob[r], data["label"][r], thr), INITS[init][0]))
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
                ax.set_title(title.replace(": ", ":\n").replace(" (", "\n("), fontsize=7.5, color=INK)
        axes[r, 0].set_ylabel(f"{pid}\n{chosen[pid].split(',')[0]}", fontsize=7, color=INK2)
    handles = [Patch(facecolor=c, edgecolor=INK2, linewidth=0.4, label=n) for n, c in OUTCOMES]
    fig.legend(handles=handles, loc="lower center", ncol=len(OUTCOMES), fontsize=8, frameon=False)
    fig.suptitle(f"Example predictions, models trained with {frac:.0%} of the labels "
                 f"(seed 0, threshold tuned on Amazon validation)", fontsize=10, color=INK, y=0.995)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="downloaded results folder (with checkpoints/)")
    p.add_argument("--frac", type=float, default=0.1, help="label fraction of the models to show")
    p.add_argument("--tag", default="full")
    p.add_argument("--input", choices=["post", "both"], default="post")
    p.add_argument("--n-amazon", type=int, default=3)
    p.add_argument("--n-congo", type=int, default=3)
    p.add_argument("--seed", type=int, default=0, help="which random patches to show")
    args = p.parse_args()
    results = Path(args.results)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = pd.read_csv(results / f"{args.tag}_all.csv")
    runs["input"] = runs["input"].fillna("post") if "input" in runs else "post"
    runs = runs[(runs.tag == args.tag) & (runs.seed == 0) & (runs.input == args.input)
                & ((runs.frac - args.frac).abs() < 1e-9)]
    cd = "_cd" if args.input == "both" else ""
    models = []
    for init in INITS:
        row = runs[runs.init == init]
        hits = sorted((results / "checkpoints").glob(f"{args.tag}_{init}{cd}_lr*_f{args.frac:g}_s0.pt"))
        if row.empty or not hits:
            continue
        models.append((init, hits[0], float(row.threshold.iloc[0])))
    if not models:
        raise SystemExit(f"no seed-0 checkpoints for fraction {args.frac:g} in {results / 'checkpoints'}")
    print("models:", ", ".join(f"{i} (threshold {t:.3f})" for i, _, t in models))

    index = C.labeled_index()          # includes per-patch forest/cleared pixel counts
    scratch = [m for m in models if m[0] == "scratch"]
    per_patch = None
    if scratch:
        f = results / "per_patch" / f"{scratch[0][1].stem}.csv"
        per_patch = pd.read_csv(f) if f.exists() else None
    chosen = pick_patches(index, per_patch, args.n_amazon, args.n_congo, args.seed)
    rows = index.set_index("patch_id").loc[list(chosen)]
    data = C.load_patches(rows.path, ["post", "pre", "label"], name="examples")

    preds = {}
    for init, ckpt, thr in models:
        model, two_dates = load_model(ckpt, device)
        d = data if two_dates else {"post": data["post"], "label": data["label"]}
        preds[init] = (predict(model, d, device), thr)

    fig = draw_examples(chosen, data, preds, args.frac)
    out = results / "figures"
    out.mkdir(exist_ok=True)
    name = f"fig_examples{cd}_f{args.frac:g}"
    fig.savefig(out / f"{name}.png", dpi=170)
    pd.DataFrame({"patch_id": list(chosen), "why": list(chosen.values())}).to_csv(out / f"{name}.csv", index=False)
    print(f"saved {out / (name + '.png')}")


if __name__ == "__main__":
    main()
