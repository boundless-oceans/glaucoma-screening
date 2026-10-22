"""Execute evaluation orchestration with CPU/model stubs, never import PyTorch."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glaucoma_training.data_utils import data_signature, json_save, metrics, read_rows, subset, write_predictions
from glaucoma_training.model_config import checkpoint_model_name, load_config
from glaucoma_training.preprocess_config import preprocessing_options
from glaucoma_training.run_utils import protect_output


class EvaluationWorkflowChecks(unittest.TestCase):
    def exercise(self, splits):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            for split in ["train", "valid", "test"]:
                (data / split).mkdir()
                (data / f"{split}.csv").write_text("Image_name,Age,True_Label\na.jpg,50,0\nb.jpg,60,1\n")
                for name in ["a.jpg", "b.jpg"]:
                    (data / split / name).write_bytes(b"image-not-decoded-by-orchestration-stub")
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checkpoint-snapshot")
            state = dict(config=load_config(), model_name="efficientnet_b2", model={}, age_stats={},
                         epoch=5, best_metrics={"auroc": 0.98}, data_signature=data_signature(data))
            cuda = Mock()
            cuda.is_available.return_value = False
            cuda.get_device_name.side_effect = AssertionError("CPU path accessed GPU name")
            framework = SimpleNamespace(cuda=cuda, __version__="stub", version=SimpleNamespace(cuda=None),
                                        load=lambda *a, **k: state, set_num_threads=lambda n: None,
                                        device=lambda name: SimpleNamespace(type=name, __str__=lambda: name))
            model = Mock()
            model.to.return_value = model
            seen = []

            def fake_evaluate(model, rows, device, amp):
                self.assertEqual(device.type, "cpu")
                seen.append(len(rows))
                return [r["Image_name"] for r in rows], [0.1, 0.9], [0, 1]

            tree = ast.parse((ROOT / "src/glaucoma_training/evaluate.py").read_text(encoding="utf-8-sig"))
            main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
            namespace = dict(argparse=argparse, Path=Path, datetime=datetime, timezone=timezone,
                             hashlib=hashlib, json=json, torch=framework, protect_output=protect_output,
                             checkpoint_model_name=checkpoint_model_name, data_signature=data_signature,
                             FusionModel=lambda **k: model, preprocessing_options=preprocessing_options,
                             json_save=json_save, read_rows=read_rows, subset=subset, metrics=metrics,
                             write_predictions=write_predictions, evaluate=fake_evaluate,
                             FundusDataset=lambda rows, *a, **k: rows,
                             make_loader=lambda dataset, *a, **k: dataset, __doc__="test")
            exec(compile(ast.Module(body=[main], type_ignores=[]), "<evaluation-main>", "exec"), namespace)
            output = root / "evaluation"
            argv = ["evaluate", "--data-root", str(data), "--checkpoint", str(checkpoint),
                    "--output-dir", str(output), "--device", "cpu", "--splits", *splits]
            with patch.object(sys, "argv", argv):
                namespace["main"]()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertIsNone(manifest["gpu"])
            self.assertEqual(len(seen), len(set(splits)))
            cuda.get_device_name.assert_not_called()
            if "test" in splits:
                prediction = output / "submission.csv"
                self.assertEqual(manifest["prediction_sha256"], hashlib.sha256(prediction.read_bytes()).hexdigest())
                self.assertEqual(manifest["prediction_rows"], 2)
            else:
                self.assertFalse((output / "submission.csv").exists())
            self.assertEqual((output / "validation_metrics.json").exists(), "valid" in splits)
            self.assertEqual(manifest["status"], "completed")

    def test_cpu_test_only_provenance(self):
        self.exercise(["test"])

    def test_cpu_validation_only(self):
        self.exercise(["valid"])

    def test_cpu_both_splits_deduplicate_requests(self):
        self.exercise(["valid", "test", "test"])


if __name__ == "__main__":
    unittest.main()
