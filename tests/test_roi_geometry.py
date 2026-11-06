"""Synthetic ROI geometry/fallback checks; no model framework or GPU."""
import io
import json
from pathlib import Path
import sys
import subprocess
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glaucoma_training.preprocessing import prepare_image
from glaucoma_training.roi_geometry import content_box, map_box_to_original, roi_options, select_roi, preview_rows, preview_strip


class RoiGeometryChecks(unittest.TestCase):
    def fixture(self, dimensions=(200, 100), size=64):
        original = Image.new("RGB", dimensions, (110, 70, 30))
        processed, info = prepare_image(original, size)
        return original, processed, info

    def test_letterbox_matches_pillow_and_full_content_maps_to_original(self):
        for dimensions in ((200, 100), (100, 200), (333, 201), (201, 333), (201, 201)):
            original, processed, info = self.fixture(dimensions)
            box = content_box(info, 64)
            self.assertEqual(box, list(processed.getbbox()))
            self.assertEqual(map_box_to_original(box, info, 64), [0, 0, *dimensions])

    def test_content_mapping_includes_crop_offset(self):
        info = dict(original_size=[1000, 800], decoded_size=[500, 400],
                    decoded_crop_box=[50, 40, 450, 360])
        box = content_box(info, 100)
        self.assertEqual(map_box_to_original(box, info, 100), [100, 80, 900, 720])
        cl, ct, cr, cb = box
        middle = [(cl + cr) / 2 - 10, (ct + cb) / 2 - 10, (cl + cr) / 2 + 10, (ct + cb) / 2 + 10]
        self.assertEqual(map_box_to_original(middle, info, 100), [420, 320, 580, 480])

    def test_jpeg_draft_exif_and_crop_use_same_coordinate_contract(self):
        original = Image.new("RGB", (2048, 1536), "black")
        ImageDraw.Draw(original).ellipse((250, 100, 1800, 1400), fill=(120, 40, 20))
        exif = Image.Exif(); exif[274] = 6
        data = io.BytesIO(); original.save(data, format="JPEG", exif=exif)
        with Image.open(io.BytesIO(data.getvalue())) as source:
            _, info = prepare_image(source, 64, {"remove_black_border": True, "jpeg_draft_size": 640})
        self.assertEqual(info["original_size"], [1536, 2048])
        mapped = map_box_to_original(content_box(info, 64), info, 64)
        np.testing.assert_allclose(mapped, info["crop_box"], atol=1)

    def test_localized_attention_selects_candidate_near_center(self):
        _, processed, info = self.fixture()
        heat = np.zeros((64, 64)); heat[30:34, 30:34] = 1
        result = select_roi(heat, processed, info)
        self.assertFalse(result["fallback"])
        l, t, r, b = result["model_box"]
        self.assertLessEqual(abs((l + r) / 2 - 32), 1)
        self.assertLessEqual(abs((t + b) / 2 - 32), 1)
        self.assertTrue(0 <= result["roi_box"][0] < result["roi_box"][2] <= 200)
        self.assertTrue(0 <= result["roi_box"][1] < result["roi_box"][3] <= 100)

    def test_heat_in_padding_cannot_create_roi(self):
        _, processed, info = self.fixture()
        heat = np.zeros((64, 64)); heat[:10, :] = 1
        result = select_roi(heat, processed, info)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["reason"], "empty_foreground_attention")
        self.assertIsNone(result["roi_box"])

    def test_uniform_and_separated_attention_fall_back(self):
        _, processed, info = self.fixture((200, 200))
        heat = np.zeros((64, 64)); heat[10, 10] = heat[54, 54] = 1
        for values in (np.ones((64, 64)), heat):
            result = select_roi(values, processed, info)
            self.assertTrue(result["fallback"])
            self.assertEqual(result["reason"], "diffuse_or_ambiguous_attention")

    def test_dark_preprocessing_fallback_is_respected(self):
        processed, info = prepare_image(Image.new("RGB", (200, 200), (10, 10, 10)), 64,
                                        {"remove_black_border": True})
        result = select_roi(np.ones((64, 64)), processed, info)
        self.assertEqual(result["reason"], "preprocessing_dark_image_fallback")

    def test_low_foreground_nonfinite_and_empty_maps(self):
        _, processed, info = self.fixture((200, 200))
        result = select_roi(np.full((64, 64), np.nan), processed, info)
        self.assertEqual(result["reason"], "nonfinite_attention")
        json.dumps(result, allow_nan=False)
        self.assertEqual(select_roi(np.zeros((64, 64)), processed, info)["reason"], "empty_foreground_attention")
        image = Image.new("RGB", (64, 64), "black"); image.putpixel((32, 32), (255, 255, 255))
        heat = np.zeros((64, 64)); heat[32, 32] = 1
        self.assertEqual(select_roi(heat, image, info)["reason"], "insufficient_foreground")

    def test_invalid_policy_and_mapping_rejected(self):
        for options in ([], {"unknown": 1}, {"size_fraction": True}, {"size_fraction": 1},
                        {"min_attention_mass": float("nan")}, {"roi_version": 2}):
            with self.assertRaises(ValueError): roi_options(options)
        _, processed, info = self.fixture()
        with self.assertRaises(ValueError): map_box_to_original([0, 0, 10, 10], info, 64)
        with self.assertRaises(ValueError): select_roi(np.ones((10, 10)), processed, info)

    def test_balanced_preview_selection_is_fixed_and_is_not_training_sampling(self):
        rows = [dict(Image_name=f"train_{i}.jpg", True_Label=str(i % 2)) for i in range(20)]
        selected = preview_rows(rows, 8, 42)
        self.assertEqual(selected, preview_rows(rows, 8, 42))
        self.assertEqual(sum(r["True_Label"] == "1" for r in selected), 4)
        self.assertEqual(len(preview_rows(rows, 8, 42, include_names=["TRAIN_19.JPG"])),
                         8 + int(rows[19] not in selected))
        with self.assertRaises(ValueError): preview_rows(rows, 8, 42, include_names=["test_1.jpg"])

    def test_preview_strip_has_all_panels_without_mutating_images(self):
        original, processed, info = self.fixture()
        original_bytes, processed_bytes = original.tobytes(), processed.tobytes()
        heat = np.zeros((64, 64)); heat[30:34, 30:34] = 1
        for values in (heat, np.zeros((64, 64))):
            result = select_roi(values, processed, info)
            strip = preview_strip(original, processed, values, result, "synthetic", tile=128)
            self.assertEqual(strip.size, (384, 170))
        self.assertEqual(original.tobytes(), original_bytes)
        self.assertEqual(processed.tobytes(), processed_bytes)

    def test_cli_rejects_test_split_before_model_loading(self):
        command = [sys.executable, "-B", str(ROOT / "scripts/preview_local_roi.py"),
                   "--checkpoint", "does-not-exist.pt", "--split", "test"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
