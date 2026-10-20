"""
Run a named group of training runs on Kaggle, one process per GPU (Kaggle's "GPU T4 x2"
gives two), print progress every 5 minutes, and print a summary table at the end.

    python -u src/run_jobs.py lrsearch

Results go to results/<job>_<part>.csv (one file per process) and results/<job>_all.csv
(combined). Logs go to log_<job>_<part>.txt next to the results folder.
If a job is interrupted, running it again skips the runs that already finished.
"""

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import common as C  # noqa: E402

FRACS = "0.01 0.05 0.1 0.25 1.0"

JOBS = {
    # Learning-rate search: every initialisation tries the same 4 learning rates at 5% and
    # 100% labels (seed 0). Rule, fixed in advance: per initialisation, pick the learning rate
    # with the highest mean Amazon-validation IoU (tuned threshold) over the two fractions.
    "lrsearch": {
        "common": "--fracs 0.05 1.0 --seeds 0 --lrs 5e-5 1e-4 2e-4 5e-4",
        "parts": {"scratch": "--inits scratch", "ssl4eo": "--inits ssl4eo"},
    },
}

PRINT_EVERY_S = 300
SUMMARY_METRICS = ["amazon_val_iou_t", "amazon_test_iou_t", "congo_test_iou_t",
                   "amazon_test_ap", "congo_test_ap"]


def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def last_line(path):
    try:
        lines = Path(path).read_text().strip().splitlines()
        return lines[-1] if lines else "(starting)"
    except FileNotFoundError:
        return "(starting)"


def launch(name, part, extra, job, device, out):
    log = out.parent / f"log_{name}_{part}.txt"
    cmd = [sys.executable, "-u", str(SRC / "train_seg.py"), *job["common"].split(), *extra.split(),
           "--tag", name, "--device", device, "--results-file", f"{name}_{part}.csv"]
    print(f"starting {part} on {device}: {' '.join(cmd[2:])}")
    return subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT), log


def run_parallel(name, job, out):
    """One process per part, each on its own GPU; parts beyond the GPU count wait their turn."""
    n_gpus = max(1, gpu_count())
    pending = list(job["parts"].items())
    running, finished = {}, {}
    t0, last_print = time.time(), 0.0
    while pending or running:
        free = [g for g in range(n_gpus) if g not in {v[2] for v in running.values()}]
        while pending and free:
            part, extra = pending.pop(0)
            gpu = free.pop(0)
            proc, log = launch(name, part, extra, job, f"cuda:{gpu}" if gpu_count() else "cpu", out)
            running[part] = (proc, log, gpu)
        time.sleep(10)
        for part in [p for p, (proc, _, _) in running.items() if proc.poll() is not None]:
            proc, log, _ = running.pop(part)
            finished[part] = (proc.returncode, log)
            print(f"{time.strftime('%H:%M')} {part} finished with exit code {proc.returncode}")
        if time.time() - last_print >= PRINT_EVERY_S:
            last_print = time.time()
            for part, (_, log, _) in running.items():
                print(f"{time.strftime('%H:%M')} {part:8s} | {last_line(log)}", flush=True)
    print(f"all parts done in {(time.time() - t0) / 60:.0f} min")
    return finished


def summarise(name, job, out):
    files = [out / f"{name}_{part}.csv" for part in job["parts"]]
    found = [f for f in files if f.exists()]
    if not found:
        print("no results found")
        return
    res = pd.concat([pd.read_csv(f) for f in found], ignore_index=True)
    res = res[res.tag == name]
    res.to_csv(out / f"{name}_all.csv", index=False)
    print(f"\n{len(res)} runs in {out / f'{name}_all.csv'}")
    agg = ["mean", "std"] if res.seed.nunique() > 1 else ["mean"]
    table = res.groupby(["init", "lr", "frac"])[SUMMARY_METRICS].agg(agg)
    print(table.round(3).to_string())
    if res.lr.nunique() > 1:
        best = res.groupby(["init", "lr"]).amazon_val_iou_t.mean().round(4)
        print("\nmean Amazon validation IoU per learning rate:\n" + best.to_string())
        print("\nchosen learning rate per initialisation:")
        for init, s in best.groupby(level="init"):
            print(f"  {init}: {s.droplevel('init').idxmax():g}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in JOBS:
        raise SystemExit(f"usage: python src/run_jobs.py JOB   (JOB is one of: {', '.join(JOBS)})")
    name = sys.argv[1]
    job = JOBS[name]
    out = C.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    finished = run_parallel(name, job, out)
    for part, (code, log) in finished.items():
        if code != 0:
            print(f"\n--- {part} failed; last lines of {log.name} ---")
            print("\n".join(Path(log).read_text().strip().splitlines()[-25:]))
    summarise(name, job, out)


if __name__ == "__main__":
    main()
