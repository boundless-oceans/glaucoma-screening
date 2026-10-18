"""Synthetic image tests for cropping; no PyTorch import or GPU execution."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glaucoma_training.preprocess_config import preprocessing_options
from glaucoma_training.preprocessing import crop_black_border, prepare_image
from glaucoma_training.model_config import canonical_config, default_run_name, load_config

ON = {"remove_black_border": True}


class PreprocessingChecks(unittest.TestCase):
    def fundus(self):
        image = Image.new("RGB", (400, 300), (0, 0, 0))
        ImageDraw.Draw(image).ellipse((80, 30, 320, 270), fill=(120, 35, 20))
        return image

    def test_crop_preserves_detectable_foreground(self):
        image = self.fundus()
        cropped, info = crop_black_border(image, ON)
        self.assertTrue(info["cropped"])
        self.assertLess(cropped.width, image.width)
        self.assertLess(cropped.height, image.height)
        ys, xs = np.nonzero(np.asarray(image).max(axis=2) > 8)
        left, top, right, bottom = info["crop_box"]
        self.assertLessEqual(left, int(xs.min()))
        self.assertLessEqual(top, int(ys.min()))
        self.assertGreater(right, int(xs.max()))
        self.assertGreater(bottom, int(ys.max()))

    def test_disabled_is_legacy_letterbox(self):
        image = self.fundus()
        output, info = prepare_image(image, 64, {})
        from PIL import ImageOps
        expected = ImageOps.pad(image, (64, 64), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
        self.assertTrue(np.array_equal(np.asarray(output), np.asarray(expected)))
        self.assertEqual(info["reason"], "disabled")

    def test_dark_and_black_images_fall_back(self):
        for color in ((0, 0, 0), (20, 10, 5)):
            image = Image.new("RGB", (100, 100), color)
            output, info = crop_black_border(image, ON)
            self.assertEqual(output.size, image.size)
            self.assertFalse(info["cropped"])

    def test_bright_image_without_borders_is_unchanged(self):
        image = Image.new("RGB", (200, 100), (100, 60, 30))
        output, info = crop_black_border(image, ON)
        self.assertEqual(info["reason"], "no_outer_border")
        self.assertEqual(output.size, image.size)

    def test_small_bright_area_not_mistaken_for_entire_fundus(self):
        image = Image.new("RGB", (200, 200), (0, 0, 0))
        ImageDraw.Draw(image).rectangle((70, 70, 125, 125), fill=(120, 40, 20))
        output, info = crop_black_border(image, ON)
        self.assertFalse(info["cropped"])
        self.assertEqual(output.size, image.size)

    def test_far_edge_signal_is_not_discarded_as_noise(self):
        image = self.fundus()
        ImageDraw.Draw(image).rectangle((0, 100, 10, 200), fill=(120, 40, 20))
        _, info = crop_black_border(image, ON)
        self.assertEqual(info["crop_box"][0], 0)

    def test_repeatable_and_rgb_output(self):
        image = self.fundus().convert("L")
        one, meta_one = prepare_image(image, 80, ON)
        two, meta_two = prepare_image(image, 80, ON)
        self.assertEqual(one.mode, "RGB")
        self.assertEqual(one.size, (80, 80))
        self.assertEqual(meta_one, meta_two)
        self.assertTrue(np.array_equal(np.asarray(one), np.asarray(two)))

    def test_invalid_options(self):
        for key, value in (("remove_black_border", "false"), ("black_border_threshold", 100),
                           ("black_border_margin", -1), ("black_border_version", 2)):
            with self.assertRaises(ValueError):
                preprocessing_options({key: value})

    def test_legacy_checkpoint_and_new_crop_config_cannot_silently_mix(self):
        old = canonical_config({"model_name": "efficientnet_b2"})
        new = canonical_config({"model_name": "efficientnet_b2", **ON})
        self.assertFalse(old["remove_black_border"])
        self.assertNotEqual(old, new)
        self.assertNotEqual(default_run_name("efficientnet_b2"),
                            default_run_name("efficientnet_b2", remove_black_border=True))

    def test_switch_is_the_only_difference_in_ablation(self):
        cropped = load_config()
        baseline = load_config(ROOT / "configs" / "no_blackborder.json")
        self.assertTrue(cropped["remove_black_border"])
        self.assertFalse(baseline["remove_black_border"])
        self.assertEqual({k: v for k, v in cropped.items() if k != "remove_black_border"},
                         {k: v for k, v in baseline.items() if k != "remove_black_border"})


if __name__ == "__main__":
    unittest.main()
