import math
import unittest

import numpy as np

from tests import SRC  # noqa: F401  (puts src/ on the path)
import common as C  # the original helper functions, kept as the reference
from forestwatch.metrics import ConfusionCounts, ProbabilityHistogram


class ConfusionCountsTests(unittest.TestCase):
    def test_worked_example(self):
        # 100 pixels predicted cleared, 100 truly cleared, 50 of them the same:
        # IoU = 50 / (100 + 100 - 50) = 1/3
        c = ConfusionCounts(tp=50, fp=50, fn=50, tn=850)
        self.assertAlmostEqual(c.iou, 1 / 3)
        self.assertAlmostEqual(c.precision, 0.5)
        self.assertAlmostEqual(c.recall, 0.5)
        self.assertAlmostEqual(c.f1, 0.5)
        self.assertAlmostEqual(c.accuracy, 0.9)

    def test_accuracy_hides_a_useless_model(self):
        # Calling nothing cleared when 1.35% is cleared: high accuracy, IoU 0
        c = ConfusionCounts(tp=0, fp=0, fn=135, tn=9865)
        self.assertGreater(c.accuracy, 0.98)
        self.assertEqual(c.iou, 0.0)

    def test_from_masks_skips_ignored_pixels(self):
        label = np.array([[1, 0, 255], [1, 0, 255]], dtype=np.uint8)
        pred = np.array([[True, True, True], [False, False, True]])
        c = ConfusionCounts.from_masks(pred, label)
        self.assertEqual((c.tp, c.fp, c.fn, c.tn), (1, 1, 1, 1))
        self.assertEqual(c.total, 4)

    def test_from_probabilities(self):
        label = np.array([1, 1, 0, 0])
        c = ConfusionCounts.from_probabilities(np.array([0.9, 0.4, 0.6, 0.1]), label, threshold=0.5)
        self.assertEqual((c.tp, c.fp, c.fn, c.tn), (1, 1, 1, 1))

    def test_empty_scores_are_nan(self):
        c = ConfusionCounts(0, 0, 0, 0)
        self.assertTrue(math.isnan(c.iou))
        self.assertTrue(math.isnan(c.precision))

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            ConfusionCounts.from_masks(np.zeros((2, 2), bool), np.zeros((3, 3), np.uint8))

    def test_negative_counts_rejected(self):
        with self.assertRaises(ValueError):
            ConfusionCounts(-1, 0, 0, 0)

    def test_adding_patches_equals_counting_them_together(self):
        rng = np.random.default_rng(0)
        labels = rng.choice([0, 1, 255], size=(5, 32, 32), p=[0.8, 0.1, 0.1])
        preds = rng.random((5, 32, 32)) < 0.2
        per_patch = [ConfusionCounts.from_masks(p, l) for p, l in zip(preds, labels)]
        self.assertEqual(sum(per_patch), ConfusionCounts.from_masks(preds, labels))

    def test_matches_original_function(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            tp, fp, fn, tn = (int(v) for v in rng.integers(0, 10_000, 4))
            new = ConfusionCounts(tp, fp, fn, tn).as_dict()
            old = C.metrics_from_counts(tp, fp, fn, tn)
            for k in new:
                self.assertTrue(math.isclose(new[k], old[k], rel_tol=1e-12), k)


def random_histogram(seed, bins=1000):
    """Counts like a decent model's: forest pixels mostly at low probabilities, cleared high."""
    rng = np.random.default_rng(seed)
    p = np.linspace(0, 1, bins)
    forest = rng.poisson(5000 * np.exp(-p * 8))
    cleared = rng.poisson(300 * np.exp(-(1 - p) * 4))
    return np.stack([forest, cleared]).astype(np.int64)


class ProbabilityHistogramTests(unittest.TestCase):
    def test_add_puts_pixels_in_the_right_bins(self):
        h = ProbabilityHistogram(bins=10)
        h.add(np.array([0.05, 0.95, 0.95, 0.5, 1.0]), np.array([0, 1, 255, 1, 1]))
        self.assertEqual(h.counts[0, 0], 1)    # forest pixel at 0.05
        self.assertEqual(h.counts[1, 9], 2)    # cleared at 0.95, and 1.0 goes in the top bin
        self.assertEqual(h.counts[1, 5], 1)    # cleared at 0.5
        self.assertEqual(h.total, 4)           # the 255 pixel is not counted

    def test_counts_agree_with_direct_counting(self):
        rng = np.random.default_rng(2)
        prob = rng.random((4, 64, 64))
        label = rng.choice([0, 1, 255], size=prob.shape, p=[0.7, 0.2, 0.1])
        h = ProbabilityHistogram().add(prob, label)
        for t in (0.0, 0.25, 0.5, 0.731, 1.0):
            with self.subTest(threshold=t):
                self.assertEqual(h.counts_at(t), ConfusionCounts.from_probabilities(prob, label, t))

    def test_histograms_add_up(self):
        a, b = ProbabilityHistogram(random_histogram(3)), ProbabilityHistogram(random_histogram(4))
        self.assertEqual((a + b).counts.sum(), a.total + b.total)
        with self.assertRaises(ValueError):
            a + ProbabilityHistogram(bins=10)

    def test_threshold_must_be_a_bin_edge(self):
        with self.assertRaises(ValueError):
            ProbabilityHistogram(bins=10).counts_at(0.55)

    def test_perfect_model(self):
        h = ProbabilityHistogram(bins=10)
        h.add(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
        self.assertEqual(h.metrics_at(h.best_threshold())["iou"], 1.0)
        self.assertAlmostEqual(h.average_precision(), 1.0)

    def test_no_cleared_pixels_gives_nan_ap(self):
        h = ProbabilityHistogram(bins=10).add(np.array([0.3, 0.6]), np.array([0, 0]))
        self.assertTrue(math.isnan(h.average_precision()))

    def test_matches_original_functions(self):
        for seed in range(5):
            raw = random_histogram(seed)
            h = ProbabilityHistogram(raw)
            k = C.best_threshold(raw)
            self.assertAlmostEqual(h.best_threshold(), k / C.NBINS)
            self.assertAlmostEqual(h.average_precision(), C.average_precision(raw), places=12)
            old = C.metrics_at_threshold(raw, k)
            new = h.metrics_at(k / C.NBINS)
            for key in new:
                self.assertAlmostEqual(new[key], old[key], places=12, msg=key)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            ProbabilityHistogram(np.zeros((3, 10)))


if __name__ == "__main__":
    unittest.main()
