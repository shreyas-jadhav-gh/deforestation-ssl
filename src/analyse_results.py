"""
Two checks on the finished experiments, using the saved seed-0 models and the test patches:

1. Same-pixel comparison. Change-detection (2019 + 2024) scores leave out pixels with no clear
   2019 view; 2024-only scores do not. Here every model is scored on BOTH pixel sets, so
   2024-only and change detection can be compared on exactly the same pixels.
2. Haze and false alarms. Haze index of a patch = median top-of-atmosphere reflectance of its
   intact-forest pixels (label 0) in band B1 (coastal aerosol, the band most sensitive to haze).
   Test patches are split into haze quartiles per region, and for every model we report the
   false-alarm rate (share of forest pixels predicted 'cleared') in each quartile.

    python src/analyse_results.py --results results/kaggle_full/results results/kaggle_cd/results

Writes to <first results folder>/figures/:
    table_same_pixels.csv     IoU on all scored pixels and on the pixels change detection scores
    table_haze.csv            false-alarm rate per haze quartile, region, model
    per_patch_analysis.csv    the per-patch numbers behind both tables
    fig_haze_false_alarms.png false-alarm rate vs haze, Congo test patches
Runs on the laptop (a GPU is used if available; about 30 s per model).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from make_figures import DPI, INITS, INK2, SURFACE, titles  # noqa: E402
from predict_examples import load_model  # noqa: E402
from train_seg import batch_input, to_input  # noqa: E402

HAZE_BAND = "B1"
COUNTS = ["tp_all", "fp_all", "fn_all", "forest_all", "tp_cd", "fp_cd", "fn_cd", "forest_cd"]


def find_models(folders, fracs):
    """Seed-0 runs of the main grids (tags full and cd) with their checkpoint files."""
    found = []
    for folder in folders:
        for tag in ["full", "cd"]:
            f = folder / f"{tag}_all.csv"
            if not f.exists():
                continue
            runs = pd.read_csv(f)
            runs["input"] = runs["input"].fillna("post") if "input" in runs else "post"
            runs = runs[(runs.tag == tag) & (runs.seed == 0) & runs.init.isin(INITS)
                        & runs.frac.round(6).isin([round(x, 6) for x in fracs])]
            for row in runs.itertuples():
                cd = "_cd" if row.input == "both" else ""
                hits = sorted((folder / "checkpoints").glob(f"{tag}_{row.init}{cd}_lr*_f{row.frac:g}_s0.pt"))
                if hits:
                    found.append({"init": row.init, "input": row.input, "frac": row.frac,
                                  "threshold": row.threshold, "ckpt": hits[0]})
    return found


@torch.no_grad()
def patch_counts(model, data, no_pre, threshold, device, batch_size=8):
    """Per patch: tp, fp, fn, forest pixels on all scored pixels and on the change-detection pixels."""
    out = np.zeros((len(data["label"]), len(COUNTS)), dtype=np.int64)
    amp = device.type == "cuda"
    for s in range(0, len(data["label"]), batch_size):
        sl = slice(s, s + batch_size)
        x = to_input(batch_input(data, sl), device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        pred = logits.float().softmax(1)[:, 1] >= threshold
        y = torch.from_numpy(data["label"][sl]).to(device)
        keep_cd = ~torch.from_numpy(no_pre[sl]).to(device)
        cols = []
        for scored in (y != C.IGNORE, (y != C.IGNORE) & keep_cd):
            cleared, forest = (y == 1) & scored, (y == 0) & scored
            cols += [(pred & cleared).sum((1, 2)), (pred & forest).sum((1, 2)),
                     (~pred & cleared).sum((1, 2)), forest.sum((1, 2))]
        out[sl] = torch.stack(cols, 1).cpu().numpy()
    return out


def haze_index(post, pre, label, no_pre):
    """Median B1 reflectance of the forest pixels in 2024 and in 2019 (NaN if no forest)."""
    b = C.BANDS.index(HAZE_BAND)
    forest = label == 0
    h24 = np.median(post[b][forest]) / C.SCALE if forest.any() else np.nan
    f19 = forest & ~no_pre
    h19 = np.median(pre[b][f19]) / C.SCALE if f19.any() else np.nan
    return h24, h19


def spearman(a, b):
    """Rank correlation without scipy: Pearson correlation of the ranks."""
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank()) if ok.sum() >= 3 else np.nan


def iou(tp, fp, fn):
    return tp / max(tp + fp + fn, 1)


def same_pixel_table(pp):
    rows = []
    for (region, init, inp, frac), g in pp.groupby(["region", "init", "input", "frac"]):
        s = g[COUNTS].sum()
        rows.append({"region": region, "init": init, "input": inp, "frac": frac,
                     "iou_all_scored": iou(s.tp_all, s.fp_all, s.fn_all),
                     "iou_cd_pixels": iou(s.tp_cd, s.fp_cd, s.fn_cd),
                     "false_alarm_rate_cd_pixels": s.fp_cd / max(s.forest_cd, 1)})
    table = pd.DataFrame(rows)
    table.loc[table.input == "both", "iou_all_scored"] = np.nan   # not defined for change detection
    return table.sort_values(["region", "frac", "init", "input"])


def haze_table(pp):
    rows = []
    for (region, init, inp, frac, q), g in pp.groupby(["region", "init", "input", "frac", "haze_q"]):
        rows.append({"region": region, "init": init, "input": inp, "frac": frac, "haze_quartile": q,
                     "patches": len(g), "haze_b1_median": g.haze_2024.median(),
                     "false_alarm_rate": g.fp_cd.sum() / max(g.forest_cd.sum(), 1)})
    return pd.DataFrame(rows)


def haze_figure(ht, pp, frac, out):
    inputs = [("post", "2024 image only"), ("both", "2019 + 2024 (change detection)")]
    inputs = [i for i in inputs if ((ht.input == i[0]) & (ht.frac.round(6) == round(frac, 6))).any()]
    if not inputs:
        return False
    congo = pp[(pp.region == "congo")].drop_duplicates("patch_id")
    edges = congo.haze_2024.quantile([0, .25, .5, .75, 1]).values
    ticks = [f"Q{i + 1}\n{edges[i]:.3f}–{edges[i + 1]:.3f}" for i in range(4)]
    fig, axes = plt.subplots(1, len(inputs), figsize=(4.6 * len(inputs) + 0.4, 3.6), sharey=True,
                             squeeze=False)
    positive = ht.false_alarm_rate[ht.false_alarm_rate > 0]
    lo = max(positive.min() * 100 / 1.5, 0.01) if len(positive) else 0.01   # zeros are drawn at the floor
    for ax, (inp, name) in zip(axes[0], inputs):
        g = ht[(ht.region == "congo") & (ht.input == inp) & (ht.frac.round(6) == round(frac, 6))]
        for init, (label, colour, marker) in INITS.items():
            r = g[g.init == init].sort_values("haze_quartile")
            if r.empty:
                continue
            ax.plot(r.haze_quartile, np.clip(r.false_alarm_rate * 100, lo, None), color=colour, marker=marker, lw=1.5,
                    ms=5.5, mec=SURFACE, mew=1.0, label=label)
        ax.set_yscale("log")
        ax.set_xticks(range(4))
        ax.set_xticklabels(ticks, fontsize=7.5)
        ax.set_xlabel(f"Congo test patches by haze (median {HAZE_BAND} reflectance of forest, 2024)",
                      fontsize=8)
        ax.set_title(name, loc="left")
        ax.grid(axis="x", visible=False)
        ax.minorticks_off()
    axes[0][0].set_ylim(lo * 0.8, 100)
    yt = [t for t in [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] if t >= lo]
    axes[0][0].set_yticks(yt)
    axes[0][0].set_yticklabels([f"{t:g}%" for t in yt])
    axes[0][0].set_ylabel("forest pixels predicted 'cleared'\n(false-alarm rate)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=4, bbox_to_anchor=(0.01, 1.0), fontsize=8)
    titles(fig, "Do hazier Congo patches get more false alarms?",
           f"models trained with {frac:.0%} of the labels, seed 0, threshold tuned on Amazon validation; "
           "Congo test patches in haze quartiles, Q1 = clearest, Q4 = haziest")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, nargs="+", help="downloaded results folder(s)")
    p.add_argument("--fracs", nargs="+", type=float, default=[0.05, 0.1, 1.0])
    p.add_argument("--haze-frac", type=float, default=0.1, help="label fraction shown in the haze figure")
    args = p.parse_args()
    folders = [Path(f) for f in args.results]
    out = folders[0] / "figures"
    out.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = find_models(folders, args.fracs)
    if not models:
        raise SystemExit(f"no seed-0 checkpoints for fractions {args.fracs} in {args.results}")
    print(f"{len(models)} models: " + ", ".join(sorted({f"{m['init']}/{m['input']}" for m in models})))

    index = C.labeled_index()
    index = index[index.split == "test"].reset_index(drop=True)
    data = C.load_patches(index.path, ["post", "pre", "label"], name="test patches")
    no_pre = np.stack([(img == 0).all(axis=0) for img in data["pre"]])   # (N, H, W): no clear 2019 view
    haze = np.array([haze_index(data["post"][i], data["pre"][i], data["label"][i], no_pre[i])
                     for i in range(len(index))])
    patches = index[["patch_id", "region"]].assign(haze_2024=haze[:, 0], haze_2019=haze[:, 1])
    patches["haze_q"] = patches.groupby("region").haze_2024.transform(
        lambda h: pd.qcut(h.rank(method="first"), 4, labels=False))

    # how much does leaving out 'no clear 2019 view' remove?
    lab = data["label"]
    for region in ["amazon", "congo"]:
        m = (index.region == region).values
        scored = lab[m] != C.IGNORE
        gone = scored & no_pre[m]
        cleared = (lab[m] == 1)
        print(f"{region}: {gone.sum() / max(scored.sum(), 1):.2%} of scored pixels and "
              f"{(gone & cleared).sum() / max(cleared.sum(), 1):.2%} of cleared pixels have no clear "
              f"2019 view; forest B1 median 2024 {np.nanmedian(haze[m, 0]):.4f}, "
              f"2019 {np.nanmedian(haze[m, 1]):.4f}")

    frames = []
    for m in sorted(models, key=lambda m: (m["input"], m["frac"], m["init"])):
        model, two_dates = load_model(m["ckpt"], device)
        d = data if two_dates else {"post": data["post"], "label": data["label"]}
        counts = patch_counts(model, d, no_pre, m["threshold"], device)
        frames.append(patches.assign(init=m["init"], input=m["input"], frac=m["frac"],
                                     **{c: counts[:, i] for i, c in enumerate(COUNTS)}))
        print(f"  {m['ckpt'].stem}: done")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pp = pd.concat(frames, ignore_index=True)
    pp.to_csv(out / "per_patch_analysis.csv", index=False)

    sp = same_pixel_table(pp)
    sp.to_csv(out / "table_same_pixels.csv", index=False)
    show = sp.copy()
    for c in ["iou_all_scored", "iou_cd_pixels"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    show["false_alarm_rate_cd_pixels"] = show.false_alarm_rate_cd_pixels.map(lambda v: f"{v:.2%}")
    print("\nSame-pixel comparison (seed 0; IoU on the pixels change detection scores):")
    print(show.to_string(index=False))

    ht = haze_table(pp)
    ht.to_csv(out / "table_haze.csv", index=False)
    print("\nFalse-alarm rate by haze quartile (0 = clearest ... 3 = haziest), seed 0:")
    wide = ht.pivot_table(index=["region", "input", "frac", "init"], columns="haze_quartile",
                          values="false_alarm_rate")
    print((wide * 100).round(2).astype(str).add("%").to_string())
    corr = {key: spearman(g.fp_cd / g.forest_cd.clip(lower=1), g.haze_2024)
            for key, g in pp.groupby(["region", "input", "frac", "init"])}
    print("\nRank correlation between a patch's haze and its false-alarm rate (+1 = hazier, more "
          "false alarms):")
    print(pd.Series(corr).rename_axis(["region", "input", "frac", "init"]).round(2).to_string())
    if haze_figure(ht, pp, args.haze_frac, out / "fig_haze_false_alarms.png"):
        print(f"\nhaze figure written")
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
