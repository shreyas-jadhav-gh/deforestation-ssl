import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tests import SRC
from forestwatch.jobs import Job, JobError, JobRunner, Step, best_moco_variant, grid_learning_rate

SPEC = {
    "before": [("pretrain_moco.py", "--build-cache")],
    "parts": {
        "a": [("train_seg.py", "--inits scratch --lrs {lr:scratch}"),
              ("train_seg.py", "--inits {best_moco}=find:pretrain/{best_moco}/encoder.pt "
                               "--lrs {lr:{best_moco}} --tag search")],
        "b": [("pretrain_moco.py", "--name moco_both --out {results}/pretrain")],
    },
}


def earlier():
    """A small stand-in for an earlier 'full' job's runs."""
    rows = []
    for init, lr, iou in (("scratch", 2e-4, 0.50), ("moco_both", 5e-4, 0.52), ("moco_amazon", 5e-4, 0.51)):
        for frac in (0.05, 1.0):
            for seed in (0, 1):
                rows.append(dict(init=init, lr=lr, frac=frac, seed=seed, amazon_val_iou_t=iou + frac / 10))
    return pd.DataFrame(rows)


def quiet(_message):
    pass


class StepTests(unittest.TestCase):
    def test_training_steps_get_tag_and_own_results_file(self):
        cmd = Step("train_seg.py", "--inits scratch").command("full", "a", "cuda:1", "/res", python="py")
        self.assertEqual(cmd[:3], ["py", "-u", str(SRC / "train_seg.py")])
        self.assertEqual(cmd[3:], ["--inits", "scratch", "--tag", "full",
                                   "--results-file", "full_a.csv", "--device", "cuda:1"])

    def test_own_tag_is_kept(self):
        cmd = Step("train_seg.py", "--tag search").command("full", "a", "cpu", "/res")
        self.assertEqual(cmd.count("--tag"), 1)

    def test_results_placeholder(self):
        cmd = Step("pretrain_moco.py", "--out {results}/pretrain").command("full", "b", "cpu", "/res")
        self.assertIn("/res/pretrain", cmd)
        self.assertNotIn("--results-file", cmd)

    def test_option(self):
        step = Step("pretrain_moco.py", "--name moco_both --steps 6000")
        self.assertEqual(step.option("--name"), "moco_both")
        self.assertEqual(step.option("--regions", "all"), "all")


class JobTests(unittest.TestCase):
    def setUp(self):
        self.job = Job.from_spec("cd", SPEC)

    def test_from_spec(self):
        self.assertEqual(list(self.job.parts), ["a", "b"])
        self.assertEqual(len(self.job.steps()), 4)
        self.assertTrue(self.job.needs_earlier_runs)
        with self.assertRaises(ValueError):
            Job("empty", {})

    def test_resolve_fills_in_choices(self):
        job = self.job.resolve(earlier(), log=quiet)
        args = job.parts["a"][1].args
        self.assertIn("moco_both=find:pretrain/moco_both/encoder.pt", args)
        self.assertIn("--lrs 0.0005", args)
        self.assertEqual(job.parts["a"][0].args, "--inits scratch --lrs 0.0002")
        self.assertFalse(job.needs_earlier_runs)
        self.assertIn("{lr:", self.job.parts["a"][0].args)       # the original is unchanged

    def test_best_moco_compares_the_same_runs(self):
        runs = earlier()
        self.assertEqual(best_moco_variant(runs, log=quiet), "moco_both")
        # drop moco_both's best runs: the comparison then uses only the runs both variants have
        # (moco_amazon is better on those, though worse on average over all of its runs)
        runs = runs[~((runs.init == "moco_both") & (runs.frac == 1.0))].copy()
        runs.loc[runs.init == "moco_amazon", "amazon_val_iou_t"] += [0.1, 0.1, -0.5, -0.5]
        self.assertEqual(best_moco_variant(runs, log=quiet), "moco_amazon")

    def test_errors(self):
        with self.assertRaises(JobError):
            best_moco_variant(earlier().query("init == 'scratch'"), log=quiet)
        two_rates = earlier()
        two_rates.loc[0, "lr"] = 1e-3
        with self.assertRaises(JobError):
            grid_learning_rate(two_rates, "scratch", log=quiet)

    def test_plan(self):
        plan = self.job.resolve(earlier(), log=quiet).plan("/res", device="cuda:0")
        self.assertEqual([part for part, _ in plan], ["before", "a", "a", "b"])
        self.assertTrue(all(cmd[-2:] == ["--device", "cuda:0"] for _, cmd in plan))


class JobRunnerTests(unittest.TestCase):
    """Runs tiny stand-in scripts instead of training, on the CPU, in a temporary folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        self.src = root / "src"
        self.src.mkdir()
        (self.src / "ok.py").write_text("import sys\nprint('done', ' '.join(sys.argv[1:]))\n")
        (self.src / "fail.py").write_text("import sys\nprint('broken')\nsys.exit(3)\n")
        self.results = root / "results"

    def tearDown(self):
        self.tmp.cleanup()

    def run_job(self, parts, n_gpus=0):
        job = Job("test", {p: [Step(s, "--x 1") for s in steps] for p, steps in parts.items()})
        runner = JobRunner(job, self.results, n_gpus=n_gpus, poll_s=0.05,
                           python=sys.executable, src_dir=self.src, log=quiet)
        return runner, runner.run()

    def test_parts_run_and_failures_are_recorded(self):
        runner, results = self.run_job({"a": ["ok.py", "fail.py", "ok.py"], "b": ["ok.py"]})
        self.assertEqual(set(results), {"a", "b"})
        self.assertEqual([code for _, code in results["a"].steps], [0, 3, 0])  # carried on
        self.assertFalse(results["a"].ok)
        self.assertTrue(results["b"].ok)
        log = runner.log_file("a").read_text()
        self.assertIn("broken", log)
        self.assertEqual(log.count("### step:"), 3)
        self.assertIn("--device cpu", log)

    def test_two_gpus_run_two_parts_at_once(self):
        runner, results = self.run_job({"a": ["ok.py"], "b": ["ok.py"]}, n_gpus=2)
        self.assertIn("cuda:0", runner.log_file("a").read_text())
        self.assertIn("cuda:1", runner.log_file("b").read_text())

    def test_last_line(self):
        self.assertEqual(JobRunner.last_line(Path(self.tmp.name) / "missing.txt"), "(starting)")


if __name__ == "__main__":
    unittest.main()
