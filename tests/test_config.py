import unittest

from tests import REPO  # noqa: F401  (puts src/ on the path)
from forestwatch.config import BANDS, REGIONS, Region, RunConfig


class RegionTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(len(BANDS), 13)
        self.assertEqual(len(set(BANDS)), 13)

    def test_bad_boxes_are_rejected(self):
        with self.assertRaises(ValueError):
            Region("x", 10, 0, 5, 1, "EPSG:4326")          # west east of east
        with self.assertRaises(ValueError):
            Region("x", 0, 5, 1, 2, "EPSG:4326")           # south north of north
        with self.assertRaises(ValueError):
            Region("x", 0, 0, 1, 1, "EPSG:4326", "other")  # unknown role

    def test_contains(self):
        amazon = REGIONS["amazon"]
        self.assertTrue(amazon.contains(-63.0, -9.5))
        self.assertFalse(amazon.contains(25.0, -2.0))

    def test_unlabelled_boxes_keep_a_buffer(self):
        # pretraining images must never come from the labelled (validation/test) areas
        labelled = [r for r in REGIONS.values() if r.role == "labelled"]
        for r in REGIONS.values():
            if r.role == "unlabelled":
                for l in labelled:
                    self.assertFalse(r.overlaps(l), f"{r.name} overlaps {l.name}")
                self.assertAlmostEqual(min(r.gap_to(l) for l in labelled), 0.1)

    def test_regions_are_immutable(self):
        with self.assertRaises(Exception):
            REGIONS["amazon"].west = 0


class RunConfigTests(unittest.TestCase):
    NAMES = ["full_scratch_lr0.0002_f0.1_s0", "full_moco_amazon_lr0.0005_f0.01_s2",
             "cd_moco_both_cd_lr0.0005_f1_s0", "cd_ssl4eo_cd_lr0.0001_f0.25_s1",
             "lrsearch_ssl4eo_lr5e-05_f0.05_s0", "search_moco_both_lr0.0005_f0.05_s0"]

    def test_names_round_trip(self):
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertEqual(RunConfig.from_name(name).name, name)

    def test_change_detection_name(self):
        c = RunConfig.from_name("cd_moco_both_cd_lr0.0005_f1_s0")
        self.assertEqual((c.tag, c.init, c.input, c.lr, c.frac, c.seed),
                         ("cd", "moco_both", "both", 0.0005, 1.0, 0))

    def test_label_seed(self):
        self.assertEqual(RunConfig("scratch", 2e-4, 0.1, 2).label_seed, 1002)

    def test_invalid_settings(self):
        for kwargs in [dict(init="imagenet21k"), dict(frac=0.0), dict(frac=1.5), dict(lr=0),
                       dict(seed=-1), dict(seed=1.0), dict(input="three")]:
            settings = dict(init="scratch", lr=2e-4, frac=0.1, seed=0) | kwargs
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                RunConfig(**settings)

    def test_not_a_run_name(self):
        with self.assertRaises(ValueError):
            RunConfig.from_name("results_summary")


if __name__ == "__main__":
    unittest.main()
