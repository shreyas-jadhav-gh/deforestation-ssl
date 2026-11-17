"""
Command line for the toolkit. From the repository root:

    python src/forestwatch regions                  # the four study regions
    python src/forestwatch patches --check          # what is in sample_data/ (or --folder data)
    python src/forestwatch budget --n 1515          # labelled patches at each label fraction
    python src/forestwatch results                  # mean test IoU per starting point and fraction
    python src/forestwatch results --cd             # ... for the change-detection runs
    python src/forestwatch rescore full_ssl4eo_lr0.0001_f0.05_s0   # recompute one run's scores

After  pip install -e .  the same commands are available as  forestwatch regions  and so on.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from .config import REGIONS, RunConfig
from .data import LabelBudget, PatchCatalog
from .metrics import ProbabilityHistogram
from .results import RunResults

REPO = Path(__file__).resolve().parents[2]
RESULTS = {"full": REPO / "results" / "kaggle_full" / "results",
           "cd": REPO / "results" / "kaggle_cd" / "results",
           "lrsearch": REPO / "results" / "kaggle_lrsearch" / "results"}
FRACS = (0.01, 0.05, 0.1, 0.25, 1.0)
EVAL_SETS = ("amazon_val", "amazon_test", "congo_test")


def cmd_regions(args):
    labelled = [r for r in REGIONS.values() if r.role == "labelled"]
    for r in REGIONS.values():
        print(r)
    for r in REGIONS.values():
        if r.role == "unlabelled":
            gap = min(r.gap_to(l) for l in labelled)
            print(f"{r.name}: {gap:.1f} degrees from the nearest labelled region")


def cmd_patches(args):
    catalog = PatchCatalog.from_folder(args.folder)
    print(catalog)
    missing = catalog.missing_files()
    if missing:
        print(f"{len(missing)} patch files missing, e.g. {missing[0]}")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(catalog.summary().to_string())
    if args.check and not missing:
        bad = []
        for patch in catalog:
            row = catalog.table.set_index("patch_id").loc[patch.patch_id]
            if any(int(row[k]) != v for k, v in patch.pixel_counts().items() if k in row):
                bad.append(patch.patch_id)
        print(f"pixel counts checked against the manifest: {len(catalog) - len(bad)} match"
              + (f", {len(bad)} differ: {bad[:5]}" if bad else ""))


def cmd_budget(args):
    budget = LabelBudget(args.n, args.seed)
    for frac, size in budget.sizes(args.fracs).items():
        print(f"{frac:>6.0%}  {size:5d} labelled patches")


def cmd_results(args):
    folder = RESULTS["cd" if args.cd else "full"]
    runs = RunResults.from_folders(folder).filter(tag="cd" if args.cd else "full")
    table = runs.label_efficiency(args.metric)
    table.columns = [f"{f:.0%}" for f in table.columns]
    print(f"mean {args.metric} over seeds ({len(runs)} runs)\n")
    print(table.round(3).to_string())
    if not args.cd:
        ref = runs.filter(init="scratch", frac=0.1).summary(args.metric)["mean"].iloc[0]
        print(f"\nrandom start with 10% of labels: {ref:.3f}. Labels each start needs to match it:")
        for init in runs.inits:
            need = runs.labels_needed(init, ref, args.metric)
            print(f"  {init:12s} {'never' if need is None else f'{need:.0%}'}")


def cmd_rescore(args):
    config = RunConfig.from_name(args.run)
    folder = next((f for f in RESULTS.values() if (f / "hists" / f"{config.name}.npz").exists()), None)
    if folder is None:
        raise SystemExit(f"no saved histogram for {config.name} in any results folder")
    hist_file = folder / "hists" / f"{config.name}.npz"
    row = RunResults.from_folders(folder).run(config)
    hists = {s: ProbabilityHistogram.from_npz(hist_file, s) for s in EVAL_SETS}
    threshold = hists["amazon_val"].best_threshold()
    print(f"{config.name}: threshold tuned on Amazon validation = {threshold:.3f} "
          f"(saved: {row.threshold:.3f})\n")
    print(f"{'':12s} {'IoU now':>8s} {'IoU saved':>10s} {'AP now':>8s} {'AP saved':>9s}")
    worst = 0.0
    for s in EVAL_SETS:
        iou, ap = hists[s].metrics_at(threshold)["iou"], hists[s].average_precision()
        worst = max(worst, abs(iou - row[f"{s}_iou_t"]), abs(ap - row[f"{s}_ap"]))
        print(f"{s:12s} {iou:8.4f} {row[f'{s}_iou_t']:10.4f} {ap:8.4f} {row[f'{s}_ap']:9.4f}")
    print("\nmatch" if worst < 1e-4 and math.isclose(threshold, row.threshold, abs_tol=1e-9)
          else f"\nMISMATCH (largest difference {worst:.2e})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forestwatch", description="Deforestation-mapping toolkit")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("regions", help="list the study regions").set_defaults(func=cmd_regions)

    s = sub.add_parser("patches", help="summarise a dataset folder")
    s.add_argument("--folder", default=str(REPO / "sample_data"))
    s.add_argument("--check", action="store_true", help="also recount every patch's pixels")
    s.set_defaults(func=cmd_patches)

    s = sub.add_parser("budget", help="labelled patches per label fraction")
    s.add_argument("--n", type=int, default=1515, help="training patches (default: Amazon train)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--fracs", type=float, nargs="+", default=list(FRACS))
    s.set_defaults(func=cmd_budget)

    s = sub.add_parser("results", help="label-efficiency table from the saved results")
    s.add_argument("--metric", default="amazon_test_iou_t")
    s.add_argument("--cd", action="store_true", help="change-detection runs instead")
    s.set_defaults(func=cmd_results)

    s = sub.add_parser("rescore", help="recompute a run's scores from its saved histograms")
    s.add_argument("run", help="run name, e.g. full_ssl4eo_lr0.0001_f0.05_s0")
    s.set_defaults(func=cmd_rescore)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
