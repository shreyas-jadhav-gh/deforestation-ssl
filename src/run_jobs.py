"""
Run a named job on Kaggle: optional preparation steps first, then one 'part' per GPU
(Kaggle's "GPU T4 x2" gives two) running side by side. Progress is printed every 5 minutes
and a summary at the end.

    python -u src/run_jobs.py lrsearch

Each part is a list of steps (scripts with arguments) run one after another on its GPU.
train_seg.py steps write results/<job>_<part>.csv (combined into results/<job>_all.csv);
pretrain_moco.py steps write results/pretrain/<name>/. Logs: log_<job>_<part>.txt.
If a job is interrupted, running it again skips work that already finished.
"""

import shlex
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import common as C  # noqa: E402

LRSEARCH = "--fracs 0.05 1.0 --seeds 0 --lrs 5e-5 1e-4 2e-4 5e-4"

JOBS = {
    # Learning-rate search: every initialisation tries the same 4 learning rates at 5% and
    # 100% labels (seed 0). Rule, fixed in advance: per initialisation, pick the learning rate
    # with the highest mean Amazon-validation IoU (tuned threshold) over the two fractions.
    "lrsearch": {
        "parts": {"scratch": [("train_seg.py", f"--inits scratch {LRSEARCH}")],
                  "ssl4eo": [("train_seg.py", f"--inits ssl4eo {LRSEARCH}")]},
    },
    # Short MoCo pretraining test (both variants side by side): speed, memory, loss going down.
    "moco_test": {
        "before": [("pretrain_moco.py", "--build-cache")],
        "parts": {"both": [("pretrain_moco.py", "--name test_both --regions amazon_u congo_u --steps 300")],
                  "amazon": [("pretrain_moco.py", "--name test_amazon --regions amazon_u --steps 300")]},
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


def command(script, args, job, part, device):
    cmd = [sys.executable, "-u", str(SRC / script), *args.split()]
    if script == "train_seg.py":
        cmd += ["--tag", job, "--results-file", f"{job}_{part}.csv"]
    return cmd + ["--device", device]


def run_parallel(job_name, job, out):
    """One process per part, each on its own GPU; parts beyond the GPU count wait their turn."""
    n_gpus = gpu_count()
    slots = max(1, n_gpus)
    pending = list(job["parts"].items())
    running, finished = {}, {}
    t0, last_print = time.time(), 0.0
    while pending or running:
        busy = {gpu for _, _, gpu in running.values()}
        free = [g for g in range(slots) if g not in busy]
        while pending and free:
            part, steps = pending.pop(0)
            gpu = free.pop(0)
            device = f"cuda:{gpu}" if n_gpus else "cpu"
            chain = " && ".join(shlex.join(command(s, a, job_name, part, device)) for s, a in steps)
            log = out.parent / f"log_{job_name}_{part}.txt"
            print(f"starting {part} on {device}: {len(steps)} step(s)", flush=True)
            proc = subprocess.Popen(chain, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT)
            running[part] = (proc, log, gpu)
        time.sleep(10)
        for part in [p for p, (proc, _, _) in running.items() if proc.poll() is not None]:
            proc, log, _ = running.pop(part)
            finished[part] = (proc.returncode, log)
            print(f"{time.strftime('%H:%M')} {part} finished with exit code {proc.returncode}", flush=True)
        if time.time() - last_print >= PRINT_EVERY_S:
            last_print = time.time()
            for part, (_, log, _) in running.items():
                print(f"{time.strftime('%H:%M')} {part:8s} | {last_line(log)}", flush=True)
    print(f"all parts done in {(time.time() - t0) / 60:.0f} min")
    return finished


def summarise_training(job_name, job, out):
    files = [out / f"{job_name}_{part}.csv" for part in job["parts"]]
    found = [f for f in files if f.exists()]
    if not found:
        return
    res = pd.concat([pd.read_csv(f) for f in found], ignore_index=True)
    res = res[res.tag == job_name]
    res.to_csv(out / f"{job_name}_all.csv", index=False)
    print(f"\n{len(res)} fine-tuning runs in {out / f'{job_name}_all.csv'}")
    agg = ["mean", "std"] if res.seed.nunique() > 1 else ["mean"]
    print(res.groupby(["init", "lr", "frac"])[SUMMARY_METRICS].agg(agg).round(3).to_string())
    if res.lr.nunique() > 1:
        best = res.groupby(["init", "lr"]).amazon_val_iou_t.mean().round(4)
        print("\nmean Amazon validation IoU per learning rate:\n" + best.to_string())
        print("\nchosen learning rate per initialisation:")
        for init, s in best.groupby(level="init"):
            print(f"  {init}: {s.droplevel('init').idxmax():g}")


def summarise_pretraining(job):
    for steps in job["parts"].values():
        for script, args in steps:
            if script != "pretrain_moco.py":
                continue
            words = args.split()
            name = words[words.index("--name") + 1] if "--name" in words else "moco_both"
            log = C.results_dir() / "pretrain" / name / "log.csv"
            if log.exists():
                df = pd.read_csv(log)
                print(f"\npretraining '{name}': first and last logged steps")
                shown = df if len(df) <= 6 else pd.concat([df.head(2), df.tail(4)])
                print(shown.to_string(index=False))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in JOBS:
        raise SystemExit(f"usage: python src/run_jobs.py JOB   (JOB is one of: {', '.join(JOBS)})")
    name = sys.argv[1]
    job = JOBS[name]
    out = C.results_dir()
    out.mkdir(parents=True, exist_ok=True)
    for script, args in job.get("before", []):
        cmd = command(script, args, name, "before", "cuda:0" if gpu_count() else "cpu")
        print(f"preparing: {script} {args}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit("preparation step failed")
    finished = run_parallel(name, job, out)
    for part, (code, log) in finished.items():
        if code != 0:
            print(f"\n--- {part} failed; last lines of {log.name} ---")
            print("\n".join(Path(log).read_text().strip().splitlines()[-25:]))
    summarise_training(name, job, out)
    summarise_pretraining(job)


if __name__ == "__main__":
    main()
