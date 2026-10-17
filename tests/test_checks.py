"""Small CPU data/guard tests. Never import torch or train any model."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import age_value, fit_age, metrics, read_rows, subset, write_predictions
from run_utils import RunLock, protect_output, validate_config


class DataChecks(unittest.TestCase):
    def test_age_train_only_and_invalid(self):
        for raw in ("", "0", "137", "nan", "inf", None):
            self.assertIsNone(age_value(raw))
        stats = fit_age([{"Age": "40"}, {"Age": "60"}, {"Age": "0"}])
        self.assertEqual(stats["median"], 50)
        self.assertEqual(stats["mean"], 50)

    def test_metrics_ties_and_rank(self):
        scores = [0.1, 0.8, 0.8, 0.9]
        labels = [0, 1, 0, 1]
        result = metrics(labels, scores)
        # Pairwise probability that a positive outranks a negative (ties count 1/2).
        positive = [p for y, p in zip(labels, scores) if y]
        negative = [p for y, p in zip(labels, scores) if not y]
        expected = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / 4
        self.assertEqual(result["auroc"], expected)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["auprc_trapezoid"], 11 / 12)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 1)
        tied = metrics([0, 1], [0.5, 0.5])
        self.assertEqual(tied["auroc"], 0.5)
        self.assertEqual(tied["average_precision"], 0.5)
        self.assertEqual(metrics([0, 1], [0.9, 0.1])["auroc"], 0)

    def test_metric_input_guards(self):
        for labels, probs in (([0, 1], [0.2]), ([0.2, 1], [0.2, 0.8]),
                              ([0, 1], [float("nan"), 0.8]), ([0, 1], [0.2, 1.1]),
                              ([1, 1], [0.2, 0.8])):
            with self.assertRaises(ValueError):
                metrics(labels, probs)

    def test_case_sensitive_paths_and_malformed_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            (root / "train" / "a.JPG").write_bytes(b"path check only")
            path = root / "train.csv"
            path.write_text("Image_name,Age,True_Label\na.jpg,50,1\n", encoding="utf-8")
            self.assertEqual(Path(read_rows(root, "train")[0]["path"]).name, "a.JPG")
            path.write_text("Image_name,Age,True_Label\nmissing.jpg,50,1\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                read_rows(root, "train")
            path.write_text("Image_name,Age,True_Label\na.jpg,50,1,extra\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_rows(root, "train")

    def test_prediction_schema_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "submission.csv"
            write_predictions(path, ["b.jpg", "a.jpg"], [0.2, 0.8])
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["Image_name", "Probability", "Pred_Label"])
                rows = list(reader)
            self.assertEqual([row["Image_name"] for row in rows], ["b.jpg", "a.jpg"])
            self.assertEqual([row["Pred_Label"] for row in rows], ["0", "1"])
            with self.assertRaises(ValueError):
                write_predictions(path, ["a", "a"], [0.1, 0.9])

    def test_stratified_debug_sample(self):
        rows = [{"True_Label": str(int(i == 99)), "Age": "50"} for i in range(100)]
        selected = subset(rows, 8, 42)
        self.assertEqual(len(selected), 8)
        self.assertEqual({r["True_Label"] for r in selected}, {"0", "1"})

    def test_config(self):
        config = json.loads((ROOT / "config.json").read_text())
        validate_config(config)
        for key, value in (("amp", "false"), ("learning_rate", math.nan), ("batch_size", True),
                           ("seed", -1), ("weight_decay", -1)):
            with self.assertRaises(ValueError):
                validate_config({**config, key: value})

    def test_dataset_output_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for target in (root, root / "train" / "x.csv", root / "test.csv"):
                with self.assertRaises(ValueError):
                    protect_output(root, target)
            self.assertEqual(protect_output(root, root / "runs" / "x.csv"), root / "runs" / "x.csv")

    def test_run_lock_released_without_deleting_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = RunLock(temporary)
            try:
                with self.assertRaises(RuntimeError):
                    RunLock(temporary)
            finally:
                first.close()
            second = RunLock(temporary)
            second.close()
            self.assertTrue((Path(temporary) / ".run.lock").exists())


if __name__ == "__main__":
    unittest.main()
