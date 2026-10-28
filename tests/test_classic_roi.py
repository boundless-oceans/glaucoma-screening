"""Geometry and failure guards for a heuristic, not a medical accuracy test."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glaucoma_training.classic_roi import classic_options, locate_classic_roi, square_extrema
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

    def test_fast_extrema_matches_pillow_including_edges(self):
        rng = np.random.default_rng(20261028)
        for shape in ((31, 27), (32, 32)):
            pixels = rng.integers(0, 256, shape, dtype=np.uint8)
            for radius in (0, 1, 3, 6):
                for maximum in (False, True):
                    filter_type = ImageFilter.MaxFilter if maximum else ImageFilter.MinFilter
                    # Radius zero is the identity; some Pillow builds cannot rank-filter size 1.
                    expected = pixels if radius == 0 else np.asarray(Image.fromarray(pixels).filter(filter_type(radius * 2 + 1)))
                    np.testing.assert_array_equal(square_extrema(pixels, radius, maximum), expected)
                    binary = pixels > 127
                    expected_binary = binary if radius == 0 else np.asarray(Image.fromarray(binary.astype(np.uint8) * 255).filter(
                        filter_type(radius * 2 + 1))) > 0
                    np.testing.assert_array_equal(square_extrema(binary, radius, maximum), expected_binary)

    def test_fast_backend_preserves_v1_score_and_geometry(self):
        image, info = self.fixture()
        policy = dict(probe_size=256, classic_roi_version=1)
        first, heat = locate_classic_roi(image, info, policy, backend="pillow")
        second, other = locate_classic_roi(image, info, policy, backend="fast")
        self.assertEqual(first, second)
        np.testing.assert_array_equal(heat, other)

    def test_v2_rim_guard_does_not_reject_every_bright_structure(self):
        for point, near_rim in (((22, 128), True), ((80, 128), False)):
            original = Image.new("RGB", (256, 256), "black")
            draw = ImageDraw.Draw(original)
            draw.ellipse((10, 10, 246, 246), fill=(100, 50, 20))
            x, y = point
            draw.ellipse((x - 14, y - 16, x + 14, y + 16), fill=(240, 210, 120))
            image, info = prepare_image(original, 256)
            old, _ = locate_classic_roi(image, info, dict(probe_size=256, classic_roi_version=1))
            new, _ = locate_classic_roi(image, info, dict(probe_size=256, classic_roi_version=2))
            self.assertFalse(old["fallback"], old)
            self.assertEqual(new["fallback"], near_rim, new)
            if near_rim: self.assertEqual(new["reason"], "rim_brightness_without_structure")


if __name__ == "__main__":
    unittest.main()
