"""Geometry and failure guards for a heuristic, not a medical accuracy test."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glaucoma_training.classic_roi import classic_options, locate_classic_roi
from glaucoma_training.preprocessing import prepare_image


class ClassicRoiChecks(unittest.TestCase):
    def fixture(self, mirror=False, twin=False):
        image = Image.new("RGB", (256, 256), "black")
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 10, 246, 246), fill=(100, 50, 20))
        draw.ellipse((57, 100, 91, 134), fill=(230, 195, 100))
        draw.line((74, 117, 160, 155), fill=(60, 20, 10), width=2)
        if twin: draw.ellipse((165, 100, 199, 134), fill=(230, 195, 100))
        if mirror: image = ImageOps.mirror(image)
        return prepare_image(image, 256)

    def test_localized_bright_structure_and_mirror(self):
        centers = []
        for mirror in (False, True):
            image, info = self.fixture(mirror)
            selection, scores = locate_classic_roi(image, info, {"probe_size": 256})
            self.assertFalse(selection["fallback"], selection)
            centers.append(selection["center_probe"])
            self.assertTrue(np.isfinite(scores).all())
            l, t, r, b = selection["roi_box"]
            point = (181 if mirror else 74, 117)
            self.assertTrue(l <= point[0] < r and t <= point[1] < b)
        self.assertLess(abs(centers[0][0] + centers[1][0] - 255), 3)

    def test_multiple_peaks_are_not_automatic_confident_localization(self):
        image, info = self.fixture(twin=True)
        selection, _ = locate_classic_roi(image, info, {"probe_size": 256})
        self.assertTrue(selection["fallback"])
        self.assertEqual(selection["reason"], "multiple_or_diffuse_bright_regions")

    def test_uniform_dark_and_metadata_fallback(self):
        for color in ((0, 0, 0), (12, 8, 4), (100, 50, 30)):
            image, info = prepare_image(Image.new("RGB", (256, 256), color), 256)
            selection, _ = locate_classic_roi(image, info, {"probe_size": 256})
            self.assertTrue(selection["fallback"])
        image, info = self.fixture()
        info["reason"] = "dark_image_fallback"
        selection, _ = locate_classic_roi(image, info, {"probe_size": 256})
        self.assertEqual(selection["reason"], "preprocessing_dark_image_fallback")

    def test_invalid_options_and_size(self):
        for config in ({"probe_size": True}, {"min_peak_ratio": float("nan")}, {"size_fraction": .1}, {"unknown": 1}):
            with self.assertRaises(ValueError): classic_options(config)
        image, info = self.fixture()
        with self.assertRaises(ValueError): locate_classic_roi(image, info)


if __name__ == "__main__":
    unittest.main()
