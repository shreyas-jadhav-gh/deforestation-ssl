import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests import SAMPLE_DATA
import common as C
from forestwatch.data import InMemoryLoader, LabelBudget, Patch, PatchCatalog, SharedCacheLoader


def write_patches(folder, n, size=16, seed=0):
    """n small synthetic patch files in folder; returns their paths."""
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        p = Path(folder) / f"amazon_r{i:03d}_c000.npz"
        np.savez(p, post=rng.integers(0, 5000, (13, size, size), dtype=np.uint16),
                 pre=rng.integers(0, 5000, (13, size, size), dtype=np.uint16),
                 label=rng.choice(np.array([0, 1, 255], dtype=np.uint8), (size, size)))
        paths.append(p)
    return paths


class PatchTests(unittest.TestCase):
    def test_shapes_are_checked(self):
        with self.assertRaises(ValueError):
            Patch("x", np.zeros((12, 8, 8), np.uint16), np.zeros((8, 8), np.uint8))
        with self.assertRaises(ValueError):
            Patch("x", np.zeros((13, 8, 8), np.uint16), np.zeros((4, 4), np.uint8))

    def test_counts_and_share(self):
        label = np.array([[0, 0, 1], [1, 255, 255]], dtype=np.uint8)
        p = Patch("congo_r001_c002", np.zeros((13, 2, 3), np.uint16), label)
        self.assertEqual(p.pixel_counts(), {"forest_px": 2, "cleared_px": 2, "ignore_px": 2})
        self.assertAlmostEqual(p.cleared_share, 0.5)
        self.assertEqual(p.region, "congo")

    def test_rgb_and_reflectance(self):
        post = np.full((13, 4, 4), 1000, np.uint16)
        p = Patch("x", post, np.zeros((4, 4), np.uint8))
        self.assertAlmostEqual(float(p.reflectance()[0, 0, 0]), 0.1, places=6)
        rgb = p.rgb()
        self.assertEqual(rgb.shape, (4, 4, 3))
        self.assertTrue(0.0 <= rgb.min() and rgb.max() <= 1.0)
        with self.assertRaises(ValueError):
            p.rgb("pre")       # this patch has no 2019 image


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.paths = write_patches(self.tmp.name, 7)

    def tearDown(self):
        gc.collect()          # close memory-mapped files before the folder is removed (Windows)
        self.tmp.cleanup()

    def test_in_memory_loader_stacks_every_patch(self):
        out = InMemoryLoader(workers=3).load(self.paths, ["post", "label"])
        self.assertEqual(out["post"].shape, (7, 13, 16, 16))
        for i, path in enumerate(self.paths):
            with np.load(path) as d:
                np.testing.assert_array_equal(out["post"][i], d["post"])
                np.testing.assert_array_equal(out["label"][i], d["label"])

    def test_both_loaders_give_the_same_arrays(self):
        a = InMemoryLoader().load(self.paths, ["post", "pre", "label"])
        loader = SharedCacheLoader(Path(self.tmp.name) / "cache")
        b = loader.load(self.paths, ["post", "pre", "label"])
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertIsInstance(b["post"], np.memmap)          # shared, on disk
        self.assertNotIsInstance(b["label"], np.memmap)      # a private copy
        b["label"][0, 0, 0] = 7                              # which the caller may change
        del a, b

    def test_cache_is_built_once(self):
        loader = SharedCacheLoader(Path(self.tmp.name) / "cache")
        self.assertFalse(loader.is_cached(self.paths, ["post"]))
        loader.load(self.paths, ["post"])
        self.assertTrue(loader.is_cached(self.paths, ["post"]))
        file = loader.cache_files(self.paths, ["post"])["post"]
        before = file.stat().st_mtime_ns
        loader.load(self.paths, ["post"])
        self.assertEqual(file.stat().st_mtime_ns, before)
        # a different set of patches gets its own cache file
        self.assertNotEqual(loader.cache_files(self.paths[:3], ["post"])["post"], file)

    def test_errors(self):
        with self.assertRaises(ValueError):
            InMemoryLoader().load([], ["post"])
        with self.assertRaises(KeyError):
            InMemoryLoader().load(self.paths, ["image"])
        with self.assertRaises(ValueError):
            InMemoryLoader(workers=0)


class LabelBudgetTests(unittest.TestCase):
    FRACS = (0.01, 0.05, 0.1, 0.25, 1.0)

    def test_sizes_for_the_amazon_training_set(self):
        sizes = LabelBudget(1515, seed=0).sizes(self.FRACS)
        self.assertEqual(list(sizes.values()), [16, 76, 152, 379, 1515])

    def test_budgets_are_nested(self):
        for seed in range(3):
            b = LabelBudget(1515, seed)
            for small, large in zip(self.FRACS, self.FRACS[1:]):
                self.assertTrue(set(b.indices(small)) <= set(b.indices(large)))

    def test_same_as_original_function(self):
        for seed in range(3):
            for frac in self.FRACS:
                np.testing.assert_array_equal(LabelBudget(1515, seed).indices(frac),
                                              C.nested_subset(1515, frac, seed))

    def test_seeds_draw_different_patches(self):
        self.assertFalse(np.array_equal(LabelBudget(1515, 0).indices(0.05),
                                        LabelBudget(1515, 1).indices(0.05)))

    def test_never_empty_and_validated(self):
        self.assertEqual(LabelBudget(10, 0).size(0.01), 1)
        with self.assertRaises(ValueError):
            LabelBudget(10, 0).size(0)
        with self.assertRaises(ValueError):
            LabelBudget(0, 0)


@unittest.skipUnless((SAMPLE_DATA / "manifests").exists(), "sample_data/ not present")
class SampleDataTests(unittest.TestCase):
    """The 16 test patches kept in the repository."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = PatchCatalog.from_folder(SAMPLE_DATA)

    def test_contents(self):
        self.assertEqual(len(self.catalog), 16)
        self.assertEqual(self.catalog.missing_files(), [])
        self.assertEqual(len(self.catalog.select(region="amazon")), 8)
        self.assertEqual(len(self.catalog.select(region="congo", split="test")), 8)
        self.assertEqual(len(self.catalog.select(split="train")), 0)

    def test_files_match_the_manifest(self):
        table = self.catalog.table.set_index("patch_id")
        for patch in self.catalog:
            with self.subTest(patch=patch.patch_id):
                self.assertEqual(patch.post.shape, (13, 256, 256))
                self.assertEqual(patch.pre.shape, (13, 256, 256))
                for k, v in patch.pixel_counts().items():
                    self.assertEqual(v, table.loc[patch.patch_id, k])

    def test_indexing(self):
        first = self.catalog[0]
        self.assertEqual(self.catalog[first.patch_id].patch_id, first.patch_id)
        self.assertIn(first.patch_id, self.catalog)
        with self.assertRaises(KeyError):
            self.catalog["nowhere_r000_c000"]

    def test_summary(self):
        s = self.catalog.summary()
        self.assertEqual(int(s.patches.sum()), 16)
        self.assertTrue(((s.cleared_share > 0) & (s.cleared_share < 1)).all())

    def test_load(self):
        data = self.catalog.select(region="congo").load(["post", "label"])
        self.assertEqual(data["post"].shape, (8, 13, 256, 256))

    def test_bad_table(self):
        with self.assertRaises(ValueError):
            PatchCatalog(pd.DataFrame({"patch_id": ["a"]}), ".")
        with self.assertRaises(ValueError):
            PatchCatalog(pd.DataFrame({"patch_id": ["a", "a"], "region": ["x", "x"],
                                       "split": ["test", "test"]}), ".")


if __name__ == "__main__":
    unittest.main()
