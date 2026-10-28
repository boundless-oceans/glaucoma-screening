"""ROI audit counts are descriptive, including unequal label coverage."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glaucoma_training.roi_audit import summarize_report


class RoiAuditChecks(unittest.TestCase):
    def report(self):
        return dict(status="completed", split="train", sampling="balanced", seed=42,
                    checkpoint_sha256="abc", roi_policy={}, candidates=1,
                    images=[dict(image_name="a.jpg", label=1, preprocess_and_cam_seconds=.02,
                                 selection=dict(fallback=True, reason="diffuse_or_ambiguous_attention",
                                                attention_mass=.4, concentration_gain=1.1,
                                                policy=dict(min_attention_mass=.55, min_concentration_gain=1.2))),
                            dict(image_name="b.jpg", label=0, preprocess_and_cam_seconds=.04,
                                 selection=dict(fallback=False, reason="candidate_attention_roi", roi_box=[0, 0, 10, 10]))])

    def test_label_coverage_and_overlapping_rejections(self):
        report = self.report()
        original = copy.deepcopy(report)
        result = summarize_report(report)
        self.assertEqual(result["by_label"]["1"]["accepted"], 0)
        self.assertEqual(result["by_label"]["0"]["accepted"], 1)
        self.assertEqual(result["acceptance_fraction"], .5)
        self.assertEqual(result["failed_conditions"],
                         dict(attention_mass_below_threshold=1, concentration_gain_below_threshold=1))
        self.assertAlmostEqual(result["timing_ms"]["mean"], 30)
        self.assertEqual(report, original)

    def test_incomplete_duplicate_and_inconsistent_records_rejected(self):
        for kind in ("incomplete", "duplicate", "count", "coordinates", "timing"):
            report = self.report()
            if kind == "incomplete": report["status"] = "running"
            if kind == "duplicate": report["images"][1]["image_name"] = "A.JPG"
            if kind == "count": report["candidates"] = 2
            if kind == "coordinates": report["images"][1]["selection"]["roi_box"] = None
            if kind == "timing": report["images"][0]["preprocess_and_cam_seconds"] = float("nan")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                summarize_report(report)


if __name__ == "__main__":
    unittest.main()
