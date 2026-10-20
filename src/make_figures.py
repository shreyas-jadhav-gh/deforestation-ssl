"""
Report figures and tables from a downloaded Kaggle results folder.

    python src/make_figures.py --results results/kaggle_full/results
    python src/make_figures.py --results results/kaggle_full/results results/kaggle_cd/results

Reads full_all.csv and cd_all.csv from every --results folder given, and pretrain/*/log.csv.
Writes PNG figures and summary tables to <first results folder>/figures/:
    fig_label_efficiency_iou.png   IoU vs label fraction, Amazon test and Congo test
    fig_label_efficiency_ap.png    same with average precision (threshold-free)
    fig_congo_false_alarms.png     share of Congo pixels predicted 'cleared', every run
    fig_pretraining.png            MoCo loss and accuracy during pretraining
    fig_change_detection.png       2024 image only vs 2019 + 2024 (if change-detection runs exist)
    table_results.csv / .md        mean +/- std over seeds, every initialisation and fraction
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# colours follow the initialisation (never the rank); validated categorical slots 1-4
INITS = {  # name in results -> (label, colour, marker)
    "scratch": ("Random init (scratch)", "#2a78d6", "o"),
    "ssl4eo": ("SSL4EO-S12 MoCo (public)", "#eb6834", "s"),
    "moco_both": ("MoCo, ours: Amazon + Congo", "#1baf7a", "D"),
    "moco_amazon": ("MoCo, ours: Amazon only", "#eda100", "^"),
}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DPI = 200

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8.5,
})


def load_runs(folders):
    frames = []
    for results in folders:
        for name in ["full_all.csv", "cd_all.csv"]:
            f = results / name
            if f.exists():
                frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit(f"no full_all.csv or cd_all.csv in {[str(f) for f in folders]}")
    runs = pd.concat(frames, ignore_index=True)
    runs = runs[runs.tag != "search"].copy()
    runs["input"] = runs["input"].fillna("post") if "input" in runs else "post"
    return runs[runs.init.isin(INITS)]


def end_labels(ax, ends):
    """Direct labels at line ends; spread apart with leader lines when they would overlap."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * 0.065
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):                      # push labels up until they clear
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    shift = max(0.0, ys[-1] - (hi - min_gap * 0.5))  # keep inside the axes
    ys = [y - shift for y in ys]
    x_end = ends[0][0]
    for (x, y, text), y_lab in zip(ends, ys):
        ax.annotate(text, xy=(x, y), xytext=(x * 1.18, y_lab), textcoords="data",
                    va="center", ha="left", fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=3)
                    if abs(y_lab - y) > min_gap * 0.25 else None)
    ax.set_xlim(right=x_end * 3.2)


def label_efficiency(runs, metric, title, ylabel, out, tag_input="post"):
    sets = [("amazon_test", "Amazon test (same region)"), ("congo_test", "Congo test (new region)")]
    r = runs[runs.input == tag_input]
    fracs = sorted(r.frac.unique())
    n_train = r.groupby("frac").n_train.first()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
    ends_per_panel, top = [], 0.0
    for ax, (s, panel_title) in zip(axes, sets):
        ends = []
        for init, (label, colour, marker) in INITS.items():
            g = r[r.init == init].groupby("frac")[f"{s}_{metric}"]
            if g.ngroups == 0:
                continue
            mean, std = g.mean(), g.std().fillna(0)
            ax.fill_between(mean.index, mean - std, mean + std, color=colour, alpha=0.10, lw=0)
            ax.plot(mean.index, mean.values, color=colour, lw=1.5, marker=marker, ms=5.5,
                    mec=SURFACE, mew=1.2, solid_capstyle="round", label=label, zorder=3)
            ends.append((mean.index[-1], mean.values[-1], f"{mean.values[-1]:.2f}"))
            top = max(top, float((mean + std).max()))
        ends_per_panel.append(ends)
        ax.set_xscale("log")
        ax.set_xticks(fracs)
        ax.set_xticklabels([f"{f:.0%}\n{n_train[f]}" for f in fracs], fontsize=7.5)
        ax.minorticks_off()
        ax.set_title(panel_title, loc="left")
        ax.set_xlabel("labeled training data: share of Amazon training patches\n(number of patches)",
                      fontsize=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(0, min(1.0, top * 1.12))       # final y-range first, then place the labels
    for ax, ends in zip(axes, ends_per_panel):
        end_labels(ax, ends)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=4, bbox_to_anchor=(0.01, 1.0), handlelength=2.2)
    fig.suptitle(title, x=0.01, y=1.09, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 1.03, "mean over 3 seeds; band = ±1 standard deviation", ha="left",
             fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def congo_false_alarms(runs, out):
    r = runs[runs.input == "post"].copy()
    col = "congo_test_pred_cleared_share_t"
    ok = (r.congo_test_recall_t > 0)
    true_share = float((r[col] * r.congo_test_precision_t / r.congo_test_recall_t)[ok].median())
    present = [i for i in INITS if i in set(r.init)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rng = np.random.default_rng(0)
    for x, init in enumerate(present):
        label, colour, marker = INITS[init]
        v = r[r.init == init][col].values * 100
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=34, color=colour, marker=marker,
                   edgecolors=SURFACE, linewidths=1.0, zorder=3)
        worst = v.max()
        ax.annotate(f"max {worst:.1f}%", xy=(x, worst), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK2)
    ax.axhline(true_share * 100, color=INK2, lw=0.8, zorder=2)
    ax.annotate(f"true share {true_share:.1%}", xy=(0.5, true_share * 100),   # in the gap between columns
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.3%", "1%", "3%", "10%", "30%"])
    ax.minorticks_off()
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([INITS[i][0].replace(": ", ":\n").replace(" (", "\n(") for i in present],
                       fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_ylabel("Congo pixels predicted 'cleared'")
    ax.set_title("How much of the Congo test area each model calls cleared", loc="left", pad=22)
    ax.text(0, 1.04, "every fine-tuning run (all label fractions and seeds); threshold tuned on Amazon "
            "validation", transform=ax.transAxes, fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def pretraining(folders, out):
    logs = {}
    for name in ["moco_both", "moco_amazon"]:
        for results in folders:
            f = results / "pretrain" / name / "log.csv"
            if f.exists() and name not in logs:
                logs[name] = pd.read_csv(f)
    if not logs:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for name, df in logs.items():
        label, colour, marker = INITS[name]
        axes[0].plot(df.step, df.loss, color=colour, lw=1.5, label=label)
        axes[1].plot(df.step, df.acc * 100, color=colour, lw=1.5, label=label)
    queue = 4096
    axes[0].axhline(np.log(queue + 1), color=MUTED, lw=0.8)
    last_step = max(df.step.max() for df in logs.values())
    axes[0].annotate(f"chance level, ln({queue + 1})", xy=(last_step, np.log(queue + 1)),
                     xytext=(0, -11), textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    axes[0].set_title("Contrastive loss (lower is better)", loc="left")
    axes[1].set_title("Positive ranked first among 4,097 (%)", loc="left")
    for ax in axes:
        ax.set_xlabel("pretraining step (batch of 256 patches)")
        ax.grid(axis="x", visible=False)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="lower right")
    fig.suptitle("MoCo pretraining on the unlabeled patches", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def change_detection(runs, out):
    """Per initialisation: the usual model (2024 image) vs the Siamese change-detection model
    (2019 + 2024), IoU vs label fraction on both test sets."""
    cd = runs[runs.input == "both"]
    if cd.empty:
        return False
    inits = [i for i in INITS if i in set(cd.init)]
    sets = [("amazon_test", "Amazon test"), ("congo_test", "Congo test")]
    styles = [("post", "-", None, "2024 image only"), ("both", "--", SURFACE, "2019 + 2024 (change detection)")]
    fracs = sorted(cd.frac.unique())
    n_train = runs.groupby("frac").n_train.first()
    fig, axes = plt.subplots(2, len(inits), figsize=(2.9 * len(inits) + 0.9, 5.4),
                             sharex=True, sharey="row", squeeze=False)
    for c, init in enumerate(inits):
        label, colour, marker = INITS[init]
        for r_, (s, set_title) in enumerate(sets):
            ax = axes[r_, c]
            for inp, ls, face, _ in styles:
                g = runs[(runs.init == init) & (runs.input == inp) & runs.frac.isin(fracs)]
                g = g.groupby("frac")[f"{s}_iou_t"]
                if g.ngroups == 0:
                    continue
                mean, std = g.mean(), g.std().fillna(0)
                ax.fill_between(mean.index, mean - std, mean + std, color=colour, alpha=0.08, lw=0)
                ax.plot(mean.index, mean.values, color=colour, lw=1.5, ls=ls, marker=marker, ms=5,
                        mfc=face or colour, mec=colour if face else SURFACE, mew=1.1, zorder=3)
            ax.set_xscale("log")
            ax.set_xticks(fracs)
            ax.set_xticklabels([f"{f:.0%}\n{n_train[f]}" for f in fracs], fontsize=7)
            ax.minorticks_off()
            ax.grid(axis="x", visible=False)
            if r_ == 0:
                ax.set_title(label.replace(": ", ":\n"), loc="left", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{set_title}\nIoU, cleared class")
            if r_ == 1:
                ax.set_xlabel("labeled patches (share, number)", fontsize=7.5)
    for row in axes:
        row[0].set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=INK2, lw=1.5, ls=ls, marker="o", ms=5, mfc=face or INK2,
                          mec=INK2 if face else SURFACE, label=name) for _, ls, face, name in styles]
    fig.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(0.01, 1.0), handlelength=2.6)
    fig.suptitle("Does seeing the 2019 image help? Change detection vs a single 2024 image",
                 x=0.01, y=1.08, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 1.035, "mean over 3 seeds, band = ±1 standard deviation; threshold tuned on Amazon "
             "validation; change-detection scores leave out pixels with no clear 2019 view",
             ha="left", fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def tables(runs, out_dir):
    metrics = ["amazon_test_iou_t", "congo_test_iou_t", "amazon_test_ap", "congo_test_ap",
               "congo_test_pred_cleared_share_t"]
    g = runs.groupby(["input", "init", "frac"])[metrics]
    table = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
    table.insert(0, "seeds", g.size())
    table.to_csv(out_dir / "table_results.csv")
    lines = ["| input | initialisation | labels | seeds | " + " | ".join(metrics) + " |",
             "|" + "---|" * (len(metrics) + 4)]
    for (inp, init, frac), row in table.iterrows():
        lines.append(f"| {inp} | {INITS[init][0]} | {frac:.0%} | {row.seeds} | "
                     + " | ".join(row[m] for m in metrics) + " |")
    (out_dir / "table_results.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, nargs="+",
                   help="downloaded results folder(s) containing full_all.csv / cd_all.csv")
    args = p.parse_args()
    folders = [Path(f) for f in args.results]
    out = folders[0] / "figures"
    out.mkdir(exist_ok=True)
    runs = load_runs(folders)
    print(f"{len(runs)} runs: " + ", ".join(f"{k}={v}" for k, v in runs.init.value_counts().items()))
    label_efficiency(runs, "iou_t", "Forest-loss IoU vs amount of labeled data",
                     "IoU, cleared class", out / "fig_label_efficiency_iou.png")
    label_efficiency(runs, "ap", "Average precision vs amount of labeled data (threshold-free)",
                     "average precision, cleared class", out / "fig_label_efficiency_ap.png")
    congo_false_alarms(runs, out / "fig_congo_false_alarms.png")
    if pretraining(folders, out / "fig_pretraining.png"):
        print("pretraining figure written")
    if change_detection(runs, out / "fig_change_detection.png"):
        print("change-detection figure written")
    tables(runs, out)
    print(f"figures and tables written to {out}")


if __name__ == "__main__":
    main()
