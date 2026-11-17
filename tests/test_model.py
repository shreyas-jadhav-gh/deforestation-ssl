"""Tests of the PyTorch parts of train_seg.py. Skipped when PyTorch or
segmentation-models-pytorch is not installed (e.g. on a computer set up only for analysis)."""

import unittest

import numpy as np

from tests import SRC  # noqa: F401  (puts src/ on the path)

try:
    import segmentation_models_pytorch as smp
    import torch
    import train_seg
except ImportError:            # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "needs PyTorch and segmentation-models-pytorch")
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.unet = smp.Unet("resnet18", encoder_weights=None, in_channels=13, classes=2)

    def test_single_date_output_shape(self):
        x = torch.rand(2, 13, 64, 64)
        self.assertEqual(tuple(self.unet(x).shape), (2, 2, 64, 64))

    def test_siamese_model(self):
        model = train_seg.SiameseDiffUnet(self.unet).eval()
        same = torch.rand(1, 13, 64, 64)
        x = torch.cat([same, same], 1)                  # 2024 and 2019 identical: no change
        with torch.no_grad():
            out = model(x)
            # the decoder only sees feature differences, so identical dates give the same
            # output as an all-zero input, whatever the image is
            zero = model(torch.cat([torch.rand(1, 13, 64, 64)] * 2, 1))
        self.assertEqual(tuple(out.shape), (1, 2, 64, 64))
        self.assertTrue(torch.allclose(out, zero, atol=1e-5))

    def test_loss_ignores_unscored_pixels(self):
        logits = torch.randn(2, 2, 8, 8)
        y = torch.full((2, 8, 8), 255, dtype=torch.long)
        self.assertEqual(float(train_seg.seg_loss(logits, y)), 0.0)

    def test_loss_rewards_right_answers(self):
        y = torch.randint(0, 2, (2, 8, 8))
        right = torch.stack([(y == 0).float(), (y == 1).float()], 1) * 10
        wrong = torch.stack([(y == 1).float(), (y == 0).float()], 1) * 10
        self.assertLess(float(train_seg.seg_loss(right, y)), float(train_seg.seg_loss(wrong, y)))

    def test_augment_keeps_image_and_label_together(self):
        x = torch.arange(2 * 16, dtype=torch.float32).reshape(2, 1, 4, 4)
        y = x[:, 0].long()
        xa, ya = train_seg.augment(x, y, np.random.default_rng(0))
        self.assertTrue(torch.equal(xa[:, 0].long(), ya))

    def test_input_scaling(self):
        x = train_seg.to_input(np.full((1, 13, 2, 2), 10000, np.uint16), torch.device("cpu"))
        self.assertTrue(torch.allclose(x, torch.ones_like(x)))


if __name__ == "__main__":
    unittest.main()
