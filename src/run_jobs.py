"""
Run a named job on Kaggle: optional preparation steps first, then one 'part' per GPU
(Kaggle's "GPU T4 x2" gives two) running side by side. Progress is printed every 5 minutes
and a summary at the end.

    python -u src/run_jobs.py full

Each part is a list of steps (scripts with arguments) run one after another on its GPU. If a
step fails, the part carries on with its next step (steps that depend on it fail quickly).
train_seg.py steps write results/<job>_<part>.csv (combined into results/<job>_all.csv);
pretrain_moco.py steps write results/pretrain/<name>/. Logs: log_<job>_<part>.txt.
'{results}' in step arguments is replaced by the results folder.
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

FRACS = "0.01 0.05 0.1 0.25 1.0"
GRID = f"--fracs {FRACS} --seeds 0 1 2"
LR_CANDIDATES = "--lrs 5e-5 1e-4 2e-4 5e-4"
SEARCH = f"--fracs 0.05 1.0 --seeds 0 {LR_CANDIDATES}"
MOCO_BOTH = "moco_both={results}/pretrain/moco_both/encoder.pt"
MOCO_AMAZON = "moco_amazon={results}/pretrain/moco_amazon/encoder.pt"

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
}

PRINT_EVERY_S = 300
SUMMARY_METRICS = ["amazon_val_iou_t", "amazon_test_iou_t", "congo_test_iou_t",
                   "amazon_test_ap", "congo_test_ap", "congo_test_pred_cleared_share_t"]


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
    words = args.replace("{results}", str(C.results_dir())).split()
    cmd = [sys.executable, "-u", str(SRC / script), *words]
    if script == "train_seg.py":
        if "--tag" not in words:
            cmd += ["--tag", job]
        cmd += ["--results-file", f"{job}_{part}.csv"]
    return cmd + ["--device", device]


def start_step(part, state, job_name, device):
    script, args = state["steps"].pop(0)
    cmd = command(script, args, job_name, part, device)
    with open(state["log"], "a") as f:
        f.write(f"\n### step: {shlex.join(cmd[2:])}\n")
    print(f"{time.strftime('%H:%M')} {part}: starting {script} on {device}", flush=True)
    state["proc"] = subprocess.Popen(cmd, stdout=open(state["log"], "a"), stderr=subprocess.STDOUT)
    state["current"] = f"{script} {args[:60]}"


def run_parallel(job_name, job, out):
    """One part per GPU; inside a part, steps run one after another."""
    n_gpus = gpu_count()
    slots = max(1, n_gpus)
    pending = list(job["parts"].items())
    running, results = {}, {}
    t0, last_print = time.time(), time.time()
    while pending or running:
        busy = {s["gpu"] for s in running.values()}
        for gpu in [g for g in range(slots) if g not in busy]:
            if not pending:
                break
            part, steps = pending.pop(0)
            log = out.parent / f"log_{job_name}_{part}.txt"
            log.write_text("")
            running[part] = {"steps": list(steps), "gpu": gpu, "log": log, "codes": []}
            start_step(part, running[part], job_name, f"cuda:{gpu}" if n_gpus else "cpu")
        time.sleep(10)
        for part in list(running):
            state = running[part]
            code = state["proc"].poll()
            if code is None:
                continue
            state["codes"].append((state["current"], code))
            print(f"{time.strftime('%H:%M')} {part}: step finished with exit code {code}", flush=True)
            if state["steps"]:
                start_step(part, state, job_name, f"cuda:{state['gpu']}" if n_gpus else "cpu")
            else:
                results[part] = (state["codes"], state["log"])
                del running[part]
        if time.time() - last_print >= PRINT_EVERY_S:
            last_print = time.time()
            for part, state in running.items():
                print(f"{time.strftime('%H:%M')} {part:11s} | {last_line(state['log'])}", flush=True)
    print(f"all parts done in {(time.time() - t0) / 60:.0f} min")
    return results


def summarise_training(job_name, job, out):
    files = [out / f"{job_name}_{part}.csv" for part in job["parts"]]
    found = [f for f in files if f.exists()]
    if not found:
        return
    res = pd.concat([pd.read_csv(f) for f in found], ignore_index=True)
    res.to_csv(out / f"{job_name}_all.csv", index=False)
    print(f"\n{len(res)} fine-tuning runs in {out / f'{job_name}_all.csv'}")
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
    results = run_parallel(name, job, out)
    print("\nstep results (exit code 0 = success):")
    for part, (codes, log) in results.items():
        for step, code in codes:
            print(f"  {part:11s} {'ok    ' if code == 0 else f'FAILED ({code})'} {step}")
        if any(code != 0 for _, code in codes):
            print(f"\n--- last lines of {log.name} ---")
            print("\n".join(Path(log).read_text().strip().splitlines()[-25:]))
    summarise_training(name, job, out)
    summarise_pretraining(job)


if __name__ == "__main__":
    main()
