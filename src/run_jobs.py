"""
Run a named job on Kaggle: optional preparation steps first, then one 'part' per GPU
(Kaggle's "GPU T4 x2" gives two) running side by side. Progress is printed every 5 minutes
and a summary at the end.

    python -u src/run_jobs.py full
    python src/run_jobs.py cd --dry-run      # only print the steps (with the choices filled in)

Each part is a list of steps (scripts with arguments) run one after another on its GPU. If a
step fails, the part carries on with its next step (steps that depend on it fail quickly).
train_seg.py steps write results/<job>_<part>.csv (combined into results/<job>_all.csv);
pretrain_moco.py steps write results/pretrain/<name>/. Logs: log_<job>_<part>.txt.
'{results}' in step arguments is replaced by the results folder.

The jobs are defined below; the Job, Step and JobRunner classes that plan and run them are in
forestwatch/jobs.py.
"""

import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import common as C  # noqa: E402
from forestwatch.jobs import Job, JobError, JobRunner  # noqa: E402

FRACS = "0.01 0.05 0.1 0.25 1.0"
GRID = f"--fracs {FRACS} --seeds 0 1 2"
LR_CANDIDATES = "--lrs 5e-5 1e-4 2e-4 5e-4"
SEARCH = f"--fracs 0.05 1.0 --seeds 0 {LR_CANDIDATES}"
MOCO_BOTH = "moco_both={results}/pretrain/moco_both/encoder.pt"
MOCO_AMAZON = "moco_amazon={results}/pretrain/moco_amazon/encoder.pt"

# Jobs that build on the 'full' job (its output attached on Kaggle as an input) can use:
#   {best_moco}   the MoCo variant with the higher mean Amazon-validation IoU (tuned threshold)
#                 over the full job's label-fraction grid (same fractions and seeds for both)
#   {lr:NAME}     the learning rate NAME used in the full job's grid
EARLIER = ["full_moco_both.csv", "full_moco_amazon.csv"]
FIND_BEST = "{best_moco}=find:pretrain/{best_moco}/encoder.pt"
FIND_BOTH = "moco_both=find:pretrain/moco_both/encoder.pt"
FIND_AMAZON = "moco_amazon=find:pretrain/moco_amazon/encoder.pt"
LR_BEST = "{lr:{best_moco}}"
CD = "--input both --cache"

JOBS = {
    # Learning-rate search: every initialisation tries the same 4 learning rates at 5% and
    # 100% labels (seed 0). Rule, fixed in advance: per initialisation, pick the learning rate
    # with the highest mean Amazon-validation IoU (tuned threshold) over the two fractions.
    "lrsearch": {
        "parts": {"scratch": [("train_seg.py", f"--inits scratch {SEARCH}")],
                  "ssl4eo": [("train_seg.py", f"--inits ssl4eo {SEARCH}")]},
    },
    # Short MoCo pretraining test (both variants side by side): speed, memory, loss going down.
    "moco_test": {
        "before": [("pretrain_moco.py", "--build-cache")],
        "parts": {"both": [("pretrain_moco.py", "--name test_both --regions amazon_u congo_u --steps 300")],
                  "amazon": [("pretrain_moco.py", "--name test_amazon --regions amazon_u --steps 300")]},
    },
    # Everything else, one GPU per MoCo variant:
    #   MoCo pretraining -> learning-rate search for that encoder (same rule as 'lrsearch')
    #   -> label-fraction grid with the chosen rate -> final baseline grid with the rate
    #   chosen by 'lrsearch' (scratch 2e-4, ssl4eo 1e-4).
    "full": {
        "before": [("pretrain_moco.py", "--build-cache")],
        "parts": {
            "moco_both": [
                ("pretrain_moco.py", "--name moco_both --regions amazon_u congo_u --steps 6000"),
                ("train_seg.py", f"--inits {MOCO_BOTH} {SEARCH} --tag search"),
                ("train_seg.py", f"--inits {MOCO_BOTH} {GRID} --lr-from-tag search"),
                ("train_seg.py", f"--inits scratch --lrs 2e-4 {GRID}"),
            ],
            "moco_amazon": [
                ("pretrain_moco.py", "--name moco_amazon --regions amazon_u --steps 6000"),
                ("train_seg.py", f"--inits {MOCO_AMAZON} {SEARCH} --tag search"),
                ("train_seg.py", f"--inits {MOCO_AMAZON} {GRID} --lr-from-tag search"),
                ("train_seg.py", f"--inits ssl4eo --lrs 1e-4 {GRID}"),
            ],
        },
    },
    # Change detection (2019 + 2024 images, Siamese U-Net) for scratch, ssl4eo and the better
    # MoCo variant, each with the learning rate it used in the full job, over the same grid.
    # Plus a check at 1e-3 for both MoCo variants (the full job's search picked 5e-4, the top of
    # its range). The patches go through a shared cache: two change-detection processes holding
    # their own copies (about 10 GB each) would not fit in Kaggle's RAM.
    "cd": {
        "before": [("train_seg.py", f"{CD} --build-cache")],
        "parts": {
            "a": [
                ("train_seg.py", f"{CD} --inits scratch --lrs {{lr:scratch}} {GRID}"),
                ("train_seg.py", f"{CD} --inits {FIND_BEST} --lrs {LR_BEST} --fracs {FRACS} --seeds 0"),
                ("train_seg.py", f"--cache --inits {FIND_BOTH} {FIND_AMAZON} "
                                 "--fracs 0.05 1.0 --seeds 0 --lrs 1e-3 --tag search"),
            ],
            "b": [
                ("train_seg.py", f"{CD} --inits ssl4eo --lrs {{lr:ssl4eo}} {GRID}"),
                ("train_seg.py", f"{CD} --inits {FIND_BEST} --lrs {LR_BEST} --fracs {FRACS} --seeds 1 2"),
            ],
        },
    },
}

PRINT_EVERY_S = 300
SUMMARY_METRICS = ["amazon_val_iou_t", "amazon_test_iou_t", "congo_test_iou_t",
                   "amazon_test_ap", "congo_test_ap", "congo_test_pred_cleared_share_t"]


def say(message):
    print(message, flush=True)


def earlier_runs():
    """The full job's fine-tuning results (found in this session's results or the inputs)."""
    frames = []
    for name in EARLIER:
        try:
            path = C.find_path(name)
        except FileNotFoundError:
            raise SystemExit(f"{name} not found: this job needs the 'full' job's output "
                             "(on Kaggle: Add Input -> that notebook version's output)")
        print(f"earlier results: {path}")
        frames.append(pd.read_csv(path))
    runs = pd.concat(frames, ignore_index=True)
    runs["input"] = runs["input"].fillna("post") if "input" in runs else "post"
    return runs[(runs.tag == "full") & (runs.input == "post")]


def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def summarise_training(job, out):
    files = [out / f"{job.name}_{part}.csv" for part in job.parts]
    found = [f for f in files if f.exists()]
    if not found:
        return
    res = pd.concat([pd.read_csv(f) for f in found], ignore_index=True)
    res.to_csv(out / f"{job.name}_all.csv", index=False)
    print(f"\n{len(res)} fine-tuning runs in {out / f'{job.name}_all.csv'}")
    metrics = [m for m in SUMMARY_METRICS if m in res.columns]
    for tag, r in res.groupby("tag"):
        agg = ["mean", "std"] if r.seed.nunique() > 1 else ["mean"]
        print(f"\n--- runs tagged '{tag}' ---")
        print(r.groupby(["init", "lr", "frac"])[metrics].agg(agg).round(3).to_string())
        if r.groupby("init").lr.nunique().max() > 1:   # a learning-rate search
            best = r.groupby(["init", "lr"]).amazon_val_iou_t.mean().round(4)
            print("\nmean Amazon validation IoU per learning rate:\n" + best.to_string())
            print("chosen learning rate per initialisation:")
            for init, s in best.groupby(level="init"):
                print(f"  {init}: {s.droplevel('init').idxmax():g}")


def summarise_pretraining(job):
    for step in [s for steps in job.parts.values() for s in steps]:
        if step.script != "pretrain_moco.py":
            continue
        name = step.option("--name", "moco_both")
        log = C.results_dir() / "pretrain" / name / "log.csv"
        if log.exists():
            df = pd.read_csv(log)
            print(f"\npretraining '{name}': first and last logged steps")
            shown = df if len(df) <= 6 else pd.concat([df.head(2), df.tail(4)])
            print(shown.to_string(index=False))


def main():
    argv = [a for a in sys.argv[1:] if a != "--dry-run"]
    if len(argv) != 1 or argv[0] not in JOBS:
        raise SystemExit(f"usage: python src/run_jobs.py JOB [--dry-run]   (JOB: {', '.join(JOBS)})")
    job = Job.from_spec(argv[0], JOBS[argv[0]])
    try:
        if job.needs_earlier_runs:
            job = job.resolve(earlier_runs())
    except JobError as e:
        raise SystemExit(str(e))
    out = C.results_dir()
    if "--dry-run" in sys.argv:   # show the steps that would run, then stop
        for part, cmd in job.plan(out):
            print(f"{part:7s} {shlex.join(cmd[2:])}")
        return
    out.mkdir(parents=True, exist_ok=True)
    for step in job.before:
        cmd = step.command(job.name, "before", "cuda:0" if gpu_count() else "cpu", out)
        print(f"preparing: {step.script} {step.args}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit("preparation step failed")
    runner = JobRunner(job, out, log_dir=out.parent, n_gpus=gpu_count(),
                       print_every_s=PRINT_EVERY_S, log=say)
    results = runner.run()
    print("\nstep results (exit code 0 = success):")
    for part, result in results.items():
        for step, code in result.steps:
            print(f"  {part:11s} {'ok    ' if code == 0 else f'FAILED ({code})'} {step}")
        if not result.ok:
            print(f"\n--- last lines of {result.log.name} ---")
            print("\n".join(Path(result.log).read_text().strip().splitlines()[-25:]))
    summarise_training(job, out)
    summarise_pretraining(job)


if __name__ == "__main__":
    main()
