"""Synthetic decoder compatibility tests. No real dataset, PyTorch or GPU."""
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glaucoma_training.benchmark_pipeline import compare_rows
from glaucoma_training.model_config import canonical_config, load_config, default_run_name
from glaucoma_training.preprocessing import prepare_image, crop_black_border
from glaucoma_training.preprocess_config import preprocessing_options
from glaucoma_training.finetune import validate_source
from glaucoma_training.run_utils import validate_config


class PipelineChecks(unittest.TestCase):
    def jpeg(self, orientation=1, color=(0, 0, 0), size=(2048, 1536), ellipse=True):
        image = Image.new("RGB", size, color)
        if ellipse:
            ImageDraw.Draw(image).ellipse((400, 100, 1700, 1400), fill=(120, 40, 20))
        exif = Image.Exif(); exif[274] = orientation
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=95, exif=exif)
        return stream.getvalue()

    def prepare(self, content, draft, border=True):
        with Image.open(io.BytesIO(content)) as source:
            return prepare_image(source, 320, {"remove_black_border": border, "jpeg_draft_size": draft})

    def test_disabled_is_pixel_identical_to_original_pipeline(self):
        content = self.jpeg(orientation=6)
        output, _ = self.prepare(content, 0)
        with Image.open(io.BytesIO(content)) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
            original, _ = crop_black_border(original, {"remove_black_border": True})
            expected = ImageOps.pad(original, (320, 320), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
        np.testing.assert_array_equal(np.asarray(output), np.asarray(expected))

    def test_draft_reduces_decode_and_preserves_full_coordinate_space(self):
        output, info = self.prepare(self.jpeg(), 640)
        self.assertEqual(output.size, (320, 320))
        self.assertTrue(info["jpeg_draft_applied"])
        self.assertEqual(info["original_size"], [2048, 1536])
        self.assertLess(info["decoded_size"][0], 2048)
        self.assertTrue(0 <= info["crop_box"][0] < info["crop_box"][2] <= 2048)
        self.assertTrue(0 <= info["crop_box"][1] < info["crop_box"][3] <= 1536)

    def test_exif_rotation_mapping_and_rgb_output(self):
        for orientation in (5, 6, 7, 8):
            output, info = self.prepare(self.jpeg(orientation), 640)
            self.assertEqual(info["original_size"], [1536, 2048])
            self.assertEqual(output.mode, "RGB")
            self.assertLessEqual(info["crop_box"][2], 1536)
            self.assertLessEqual(info["crop_box"][3], 2048)

    def test_dark_and_no_border_fallbacks_remain_auditable(self):
        for color, reason in [((15, 12, 10), "dark_image_fallback"), ((100, 60, 30), "no_outer_border")]:
            content = self.jpeg(color=color, ellipse=False)
            for draft in (0, 640):
                _, info = self.prepare(content, draft)
                self.assertEqual(info["reason"], reason)
                self.assertEqual(info["crop_box"], [0, 0, 2048, 1536])

    def test_small_jpeg_and_png_can_safely_skip_draft(self):
        for fmt in ("JPEG", "PNG"):
            stream = io.BytesIO(); Image.new("RGB", (201, 155), "red").save(stream, format=fmt)
            output, info = self.prepare(stream.getvalue(), 640)
            self.assertEqual(output.size, (320, 320))
            self.assertFalse(info["jpeg_draft_applied"])

    def test_invalid_draft_options(self):
        for value in (True, -1, 1, 640.5, 8192):
            with self.assertRaises(ValueError): preprocessing_options({"jpeg_draft_size": value})
        with self.assertRaises(ValueError): self.prepare(self.jpeg(), 128)
        with self.assertRaises(ValueError): validate_config({**load_config(), "jpeg_draft_size": 128})

    def test_odd_large_and_extreme_aspect_images(self):
        for dimensions in [(2049, 1537), (4096, 401), (401, 4096)]:
            output, info = self.prepare(self.jpeg(size=dimensions, ellipse=False, color=(110, 70, 20)), 640)
            self.assertEqual(output.size, (320, 320))
            self.assertEqual(info["original_size"], list(dimensions))
            self.assertEqual(info["crop_box"], [0, 0, *dimensions])

    def test_legacy_defaults_and_separate_new_run(self):
        original = load_config()
        old = {k: v for k, v in original.items() if k != "jpeg_draft_size"}
        self.assertEqual(canonical_config(old), original)
        candidate = load_config(ROOT / "configs/pipeline_draft.json")
        self.assertEqual({k for k in original if candidate[k] != original[k]}, {"jpeg_draft_size"})
        self.assertEqual(default_run_name("efficientnet_b2", False, True, False, 640), "efficientnet_b2_blackborder_draft640")

    def test_finetune_cannot_silently_change_decoder(self):
        source = dict(config=load_config(), model_name="efficientnet_b2")
        candidate = load_config(ROOT / "configs/finetune.json")
        candidate["jpeg_draft_size"] = 640
        with self.assertRaisesRegex(ValueError, "jpeg_draft_size"):
            validate_source(source, candidate, {}, ROOT / "old/best.pt", ROOT / "new")

    def test_cpu_benchmark_records_real_decoder_metadata_and_differences(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.jpg"; path.write_bytes(self.jpeg())
            rows = [dict(path=str(path), Image_name="a.jpg", True_Label="1")]
            records, summary = compare_rows(rows, 320, {"remove_black_border": True}, 640, repeats=1)
            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["draft_applied"], 1)
            self.assertGreater(summary["cpu_preprocessing_ratio"], 0)
            self.assertTrue(0 <= records[0]["fraction_pixels_gt8"] <= 1)
            self.assertEqual(path.read_bytes(), self.jpeg())


if __name__ == "__main__":
    unittest.main()
