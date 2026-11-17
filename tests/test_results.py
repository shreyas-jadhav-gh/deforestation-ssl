import unittest

import numpy as np
import pandas as pd

from tests import RESULTS
from forestwatch.config import RunConfig
from forestwatch.metrics import ProbabilityHistogram
from forestwatch.results import RunResults


def toy_runs():
    rows = []
    for init, base in (("scratch", 0.40), ("ssl4eo", 0.46)):
        for frac, gain in ((0.05, 0.0), (0.1, 0.05), (1.0, 0.2)):
            for seed in range(3):
                rows.append(dict(tag="full", init=init, input="post", lr=2e-4, frac=frac,
                                 seed=seed, amazon_test_iou_t=base + gain + 0.01 * seed))
    return pd.DataFrame(rows)


class RunResultsTests(unittest.TestCase):
    def setUp(self):
        self.runs = RunResults(toy_runs())

    def test_filter(self):
        self.assertEqual(len(self.runs), 18)
        self.assertEqual(len(self.runs.filter(init="ssl4eo")), 9)
        self.assertEqual(len(self.runs.filter(frac=[0.05, 1.0], seed=0)), 4)
        self.assertEqual(len(self.runs.filter(lr=0.0002)), 18)   # float compared with tolerance
        with self.assertRaises(KeyError):
            self.runs.filter(colour="green")

    def test_summary_and_table(self):
        s = self.runs.summary("amazon_test_iou_t")
        self.assertAlmostEqual(s.loc[("scratch", 0.05), "mean"], 0.41)
        self.assertEqual(s.loc[("scratch", 0.05), "runs"], 3)
        table = self.runs.label_efficiency("amazon_test_iou_t")
        self.assertEqual(table.shape, (2, 3))
        self.assertAlmostEqual(table.loc["ssl4eo", 1.0], 0.67)

    def test_labels_needed(self):
        target = self.runs.filter(init="scratch", frac=0.1).summary("amazon_test_iou_t")["mean"].iloc[0]
        self.assertEqual(self.runs.labels_needed("ssl4eo", target, "amazon_test_iou_t"), 0.05)
        self.assertEqual(self.runs.labels_needed("scratch", target, "amazon_test_iou_t"), 0.1)
        self.assertIsNone(self.runs.labels_needed("scratch", 0.99, "amazon_test_iou_t"))

    def test_missing_input_column_means_single_date(self):
        runs = RunResults(toy_runs().drop(columns="input"))
        self.assertEqual(set(runs.table.input), {"post"})

    def test_errors(self):
        with self.assertRaises(ValueError):
            RunResults(pd.DataFrame({"init": ["scratch"]}))
        with self.assertRaises(KeyError):
            self.runs.summary("congo_test_ap")
        with self.assertRaises(KeyError):
            self.runs.run(RunConfig("scratch", 2e-4, 0.25, 0))


@unittest.skipUnless((RESULTS["full"] / "full_all.csv").exists(), "saved results not present")
class SavedResultsTests(unittest.TestCase):
    """The numbers quoted in the report, recomputed from the results kept in the repository."""

    @classmethod
    def setUpClass(cls):
        cls.full = RunResults.from_folders(RESULTS["full"])
        cls.cd = RunResults.from_folders(RESULTS["cd"])
        cls.search = RunResults.from_folders(RESULTS["lrsearch"])

    def mean(self, runs, init, frac, metric="amazon_test_iou_t"):
        return runs.filter(init=init, frac=frac).summary(metric)["mean"].iloc[0]

    def test_141_runs(self):
        self.assertEqual(len(self.full) + len(self.cd) + len(self.search), 141)
        grid = self.full.filter(tag="full")
        self.assertEqual(len(grid), 4 * 5 * 3)          # starting points x fractions x seeds

    def test_pretraining_halves_the_labels(self):
        grid = self.full.filter(tag="full")
        self.assertAlmostEqual(self.mean(grid, "ssl4eo", 0.05), 0.538, places=3)
        self.assertAlmostEqual(self.mean(grid, "scratch", 0.05), 0.477, places=3)
        self.assertAlmostEqual(self.mean(grid, "scratch", 0.1), 0.535, places=3)
        target = self.mean(grid, "scratch", 0.1)
        self.assertEqual(grid.labels_needed("ssl4eo", target, "amazon_test_iou_t"), 0.05)

    def test_change_detection_gain(self):
        best_single = self.full.filter(tag="full", frac=1.0).summary("amazon_test_iou_t")["mean"].max()
        best_cd = self.cd.filter(tag="cd", frac=1.0).summary("amazon_test_iou_t")["mean"].max()
        self.assertAlmostEqual(best_single, 0.66, places=2)
        self.assertAlmostEqual(best_cd, 0.85, places=2)
        # change detection with 5% of the labels beats every single-date model with all of them
        worst_cd_5 = self.cd.filter(tag="cd", frac=0.05).summary("amazon_test_iou_t")["mean"].min()
        self.assertGreater(worst_cd_5, best_single)

    def test_learning_rate_search_picks(self):
        self.assertEqual(self.search.best_learning_rates(), {"scratch": 2e-4, "ssl4eo": 1e-4})

    def test_scores_rebuilt_from_saved_histograms(self):
        """Every run with a saved histogram: tuning the threshold on Amazon validation and
        scoring the test sets again gives the numbers in the results CSV."""
        checked = 0
        for name, runs in (("full", self.full), ("cd", self.cd), ("lrsearch", self.search)):
            for _, row in runs.table.iterrows():
                config = RunConfig(row.init, float(row.lr), float(row.frac), int(row.seed),
                                   row.input, row.tag)
                path = RESULTS[name] / "hists" / f"{config.name}.npz"
                if not path.exists():
                    continue
                with self.subTest(run=config.name):
                    t = ProbabilityHistogram.from_npz(path, "amazon_val").best_threshold()
                    self.assertAlmostEqual(t, row.threshold, places=6)
                    for s in ("amazon_val", "amazon_test", "congo_test"):
                        h = ProbabilityHistogram.from_npz(path, s)
                        self.assertAlmostEqual(h.metrics_at(t)["iou"], row[f"{s}_iou_t"], places=4)
                        self.assertAlmostEqual(h.average_precision(), row[f"{s}_ap"], places=4)
                checked += 1
        self.assertGreater(checked, 100)


if __name__ == "__main__":
    unittest.main()
