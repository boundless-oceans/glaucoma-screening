"""CPU synthetic cache resume, source integrity and EXIF crop mapping checks."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glaucoma_training.build_roi_cache import main as build_cache
from glaucoma_training.roi_cache import load_roi_cache, cache_rows, prepare_cached_image, validate_record, snapshot_roi_cache
from glaucoma_training.run_utils import validate_config
from glaucoma_training.model_config import load_config


class RoiCacheChecks(unittest.TestCase):
    def fixture(self, root):
        data = root / "data"; data.mkdir()
        for split in ("train", "valid", "test"):
            (data / split).mkdir()
            with (data / f"{split}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["Image_name", "Age"] + ([] if split == "test" else ["True_Label"]))
                writer.writeheader()
                for i in range(2):
                    name = f"{split}_{i}.jpg"
                    Image.new("RGB", (160, 128), (80+i*20, 40, 20)).save(data / split / name)
                    row = dict(Image_name=name, Age=50+i)
                    if split != "test": row["True_Label"] = i
                    writer.writerow(row)
        policy = root / "policy.json"; policy.write_text(json.dumps(dict(classic_roi_version=2, probe_size=128)))
        return data, policy

    def build(self, data, policy, cache):
        argv = ["cache", "--data-root", str(data), "--roi-config", str(policy), "--output-dir", str(cache), "--workers", "2"]
        with patch.object(sys, "argv", argv): build_cache()

    def test_cache_resume_and_no_label_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data, policy = self.fixture(root); cache = root / "cache"
            self.build(data, policy, cache)
            original = load_roi_cache(data, cache)
            snapshot = root / "run" / "roi_cache"
            snapshot_roi_cache(cache, snapshot, original["signature"])
            self.assertEqual(load_roi_cache(data, snapshot)["signature"], original["signature"])
            snapshot_roi_cache(cache, snapshot, original["signature"])
            rows = cache_rows(original, "train", [dict(Image_name="train_0.jpg")])
            self.assertNotIn("label", rows["train_0.jpg"])
            self.assertNotIn("Age", rows["train_0.jpg"])
            with self.assertRaises(ValueError): cache_rows(original, "train", [dict(Image_name="missing.jpg")])
            # Simulate an interrupted last record and resume the same fixed selection.
            (cache / "manifest.json").unlink()
            path = cache / "train.jsonl"; first = path.read_bytes().splitlines()[0] + b"\n"
            path.rename(cache / "train.jsonl.part")
            (cache / "train.jsonl.part").write_bytes(first + b'{"interrupted":')
            self.build(data, policy, cache)
            self.assertEqual(load_roi_cache(data, cache)["signature"], original["signature"])
            with self.assertRaises(ValueError): load_roi_cache(data, cache, "wrong-checkpoint-cache")
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError): load_roi_cache(data, cache)

    def test_changed_images_are_rejected_at_decode(self):
        image = Image.new("RGB", (100, 80), (120, 40, 20))
        buffer = io.BytesIO(); image.save(buffer, format="PNG"); raw = buffer.getvalue()
        record = dict(image_name="a.png", original_size=[100, 80], box=[10, 5, 80, 70],
                      fallback=False, input_bytes=len(raw), input_sha256=hashlib.sha256(raw).hexdigest())
        expected = ImageOps.pad(image.crop(record["box"]), (64, 64), method=Image.Resampling.BILINEAR, color="black")
        np.testing.assert_array_equal(np.asarray(prepare_cached_image(raw, record, 64)), np.asarray(expected))
        with self.assertRaises(ValueError): prepare_cached_image(raw + b"changed", record, 64)
        with self.assertRaises(ValueError): validate_record({**record, "box":[0, 0, 101, 80]})

    def test_exif_and_draft_map_full_original_coordinates(self):
        image = Image.new("RGB", (1600, 1200), (120, 40, 20))
        exif = Image.Exif(); exif[274] = 6
        stream = io.BytesIO(); image.save(stream, format="JPEG", exif=exif); raw = stream.getvalue()
        record = dict(image_name="a.jpg", original_size=[1200, 1600], box=[100, 200, 900, 1200],
                      fallback=False, input_bytes=len(raw), input_sha256=hashlib.sha256(raw).hexdigest())
        output = prepare_cached_image(raw, record, 128, 640)
        self.assertEqual(output.size, (128, 128))
        self.assertTrue(np.asarray(output).max() > 0)

    def test_local_training_modes_are_guarded(self):
        config = load_config(ROOT / "configs/local_b0.json")
        validate_config(config)
        for changes in (dict(use_age=True), dict(finetune=True)):
            with self.assertRaises(ValueError): validate_config({**config, **changes})


if __name__ == "__main__": unittest.main()
