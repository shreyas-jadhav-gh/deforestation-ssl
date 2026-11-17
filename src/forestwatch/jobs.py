"""
Jobs, steps and the job runner used on Kaggle.

A job is made of parts, and a part is a list of steps (a script in src/ with its arguments)
that run one after another. Parts run side by side, one per GPU (Kaggle's "GPU T4 x2" gives
two), each writing its own log. A step that fails is recorded and its part carries on with
the next step. Placeholders in the arguments are filled in before anything runs:

    {results}     the results folder
    {best_moco}   the MoCo variant that did better in an earlier 'full' job
    {lr:NAME}     the learning rate NAME used in that earlier job
"""

from __future__ import annotations

import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

SRC = Path(__file__).resolve().parents[1]     # the folder with the training scripts
_LR = re.compile(r"\{lr:(\w+)\}")


class JobError(Exception):
    """A job cannot be planned, e.g. because the earlier results it builds on are missing."""


@dataclass(frozen=True)
class Step:
    """One script run: e.g. Step("train_seg.py", "--inits scratch --fracs 0.05")."""

    script: str
    args: str = ""

    def command(self, job: str, part: str, device: str, results_dir, python: str = sys.executable,
                src_dir=SRC) -> list[str]:
        """The command line for this step. Training steps also get the job name as their tag
        (unless they set one) and a results file of their own, so parts never write to the
        same file."""
        words = self.args.replace("{results}", str(results_dir)).split()
        cmd = [python, "-u", str(Path(src_dir) / self.script), *words]
        if self.script == "train_seg.py":
            if "--tag" not in words:
                cmd += ["--tag", job]
            cmd += ["--results-file", f"{job}_{part}.csv"]
        return cmd + ["--device", device]

    def with_args(self, args: str) -> "Step":
        return Step(self.script, args)

    def option(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """The value given for a command-line option, e.g. step.option("--name")."""
        words = self.args.split()
        return words[words.index(name) + 1] if name in words[:-1] else default


def best_moco_variant(runs: pd.DataFrame, log: Callable = print) -> str:
    """The MoCo variant with the higher mean Amazon-validation IoU, compared over the runs
    (label fraction and seed) that both variants have."""
    rows = runs[runs.init.str.startswith("moco")]
    table = rows.pivot_table(index=["frac", "seed"], columns="init", values="amazon_val_iou_t").dropna()
    if table.empty:
        raise JobError("no MoCo runs found in the earlier results")
    means = table.mean()
    log(f"MoCo variant: mean Amazon-validation IoU over the same {len(table)} runs each: "
        + ", ".join(f"{k} {v:.4f}" for k, v in means.items()) + f"  -> {means.idxmax()}")
    return str(means.idxmax())


def grid_learning_rate(runs: pd.DataFrame, init: str, log: Callable = print) -> float:
    """The single learning rate that `init` used in the earlier grid."""
    lrs = runs[runs.init == init].lr.unique()
    if len(lrs) != 1:
        raise JobError(f"expected one learning rate for {init} in the earlier grid, found {list(lrs)}")
    log(f"learning rate for {init}: {lrs[0]:g}")
    return float(lrs[0])


@dataclass
class Job:
    """A named job: optional preparation steps, then parts that run side by side."""

    name: str
    parts: dict[str, list[Step]]
    before: list[Step] = field(default_factory=list)

    def __post_init__(self):
        if not self.parts:
            raise ValueError(f"job {self.name} has no parts")

    @classmethod
    def from_spec(cls, name: str, spec: dict) -> "Job":
        """From the plain form used in run_jobs.py: {"before": [(script, args), ...],
        "parts": {part: [(script, args), ...]}}."""
        return cls(name,
                   {part: [Step(*s) for s in steps] for part, steps in spec["parts"].items()},
                   [Step(*s) for s in spec.get("before", [])])

    def steps(self) -> list[Step]:
        return [s for steps in self.parts.values() for s in steps] + list(self.before)

    @property
    def needs_earlier_runs(self) -> bool:
        return any("{best_moco}" in s.args or "{lr:" in s.args for s in self.steps())

    def resolve(self, earlier: pd.DataFrame, log: Callable = print) -> "Job":
        """A copy with {best_moco} and {lr:NAME} filled in from an earlier job's runs."""
        if not self.needs_earlier_runs:
            return self
        best = (best_moco_variant(earlier, log)
                if any("{best_moco}" in s.args for s in self.steps()) else None)
        lrs: dict[str, float] = {}

        def lr(match):
            name = match.group(1)
            if name not in lrs:
                lrs[name] = grid_learning_rate(earlier, name, log)
            return f"{lrs[name]:g}"

        def fill(step: Step) -> Step:
            args = step.args.replace("{best_moco}", best) if best else step.args
            return step.with_args(_LR.sub(lr, args))

        before = [fill(s) for s in self.before]
        return Job(self.name, {p: [fill(s) for s in steps] for p, steps in self.parts.items()}, before)

    def plan(self, results_dir, device: str = "cuda:0", python: str = sys.executable,
             src_dir=SRC) -> list[tuple[str, list[str]]]:
        """(part, command) for every step, preparation steps first, without running anything."""
        out = [("before", s.command(self.name, "before", device, results_dir, python, src_dir))
               for s in self.before]
        for part, steps in self.parts.items():
            out += [(part, s.command(self.name, part, device, results_dir, python, src_dir)) for s in steps]
        return out


@dataclass
class PartResult:
    """What happened in one part: the exit code of every step, and the log file."""

    part: str
    steps: list[tuple[str, int]]
    log: Path

    @property
    def ok(self) -> bool:
        return all(code == 0 for _, code in self.steps)


@dataclass
class _Running:
    steps: list[Step]
    gpu: int
    log: Path
    codes: list[tuple[str, int]] = field(default_factory=list)
    proc: Optional[subprocess.Popen] = None
    handle: object = None
    current: str = ""


class JobRunner:
    """Runs a job's parts side by side, one per GPU, or one after another without a GPU.
    Prints the last line of every running part's log every `print_every_s` seconds."""

    def __init__(self, job: Job, results_dir, log_dir=None, n_gpus: int = 0, poll_s: float = 10,
                 print_every_s: float = 300, python: str = sys.executable, src_dir=SRC,
                 log: Callable = print):
        self.job = job
        self.results_dir = Path(results_dir)
        self.log_dir = Path(log_dir) if log_dir is not None else self.results_dir.parent
        self.n_gpus = n_gpus
        self.poll_s = poll_s
        self.print_every_s = print_every_s
        self.python = python
        self.src_dir = src_dir
        self.log = log

    @property
    def slots(self) -> int:
        return max(1, self.n_gpus)

    def device(self, gpu: int) -> str:
        return f"cuda:{gpu}" if self.n_gpus else "cpu"

    def log_file(self, part: str) -> Path:
        return self.log_dir / f"log_{self.job.name}_{part}.txt"

    @staticmethod
    def last_line(path) -> str:
        try:
            lines = Path(path).read_text().strip().splitlines()
            return lines[-1] if lines else "(starting)"
        except FileNotFoundError:
            return "(starting)"

    def _start_next(self, part: str, state: _Running):
        step = state.steps.pop(0)
        cmd = step.command(self.job.name, part, self.device(state.gpu), self.results_dir,
                           self.python, self.src_dir)
        with open(state.log, "a") as f:
            f.write(f"\n### step: {shlex.join(cmd[2:])}\n")
        self.log(f"{time.strftime('%H:%M')} {part}: starting {step.script} on {self.device(state.gpu)}")
        state.handle = open(state.log, "a")
        state.proc = subprocess.Popen(cmd, stdout=state.handle, stderr=subprocess.STDOUT)
        state.current = f"{step.script} {step.args[:60]}"

    def run(self) -> dict[str, PartResult]:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        pending = list(self.job.parts.items())
        running: dict[str, _Running] = {}
        results: dict[str, PartResult] = {}
        t0 = last_print = time.time()
        while pending or running:
            busy = {s.gpu for s in running.values()}
            for gpu in [g for g in range(self.slots) if g not in busy]:
                if not pending:
                    break
                part, steps = pending.pop(0)
                log = self.log_file(part)
                log.write_text("")
                running[part] = _Running(list(steps), gpu, log)
                self._start_next(part, running[part])
            time.sleep(self.poll_s)
            for part in list(running):
                state = running[part]
                code = state.proc.poll()
                if code is None:
                    continue
                state.handle.close()
                state.codes.append((state.current, code))
                self.log(f"{time.strftime('%H:%M')} {part}: step finished with exit code {code}")
                if state.steps:
                    self._start_next(part, state)
                else:
                    results[part] = PartResult(part, state.codes, state.log)
                    del running[part]
            if time.time() - last_print >= self.print_every_s:
                last_print = time.time()
                for part, state in running.items():
                    self.log(f"{time.strftime('%H:%M')} {part:11s} | {self.last_line(state.log)}")
        self.log(f"all parts done in {(time.time() - t0) / 60:.0f} min")
        return results
