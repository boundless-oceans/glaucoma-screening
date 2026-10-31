"""Fixed fusion must align names and labels before averaging."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from glaucoma_training.fusion_utils import blend_predictions


class FusionChecks(unittest.TestCase):
    def rows(self, scores):
        return [dict(Image_name=f"test_{i}.jpg", Probability=str(p), True_Label=str(i % 2)) for i,p in enumerate(scores)]

    def test_fixed_average_and_debug_subset(self):
        global_rows = self.rows([.2, .8, .4, .6]); local = self.rows([.4, .6, .2, .8])
        names, gp, lp, fused, labels = blend_predictions(global_rows, local, labeled=True)
        self.assertEqual(labels, [0, 1, 0, 1])
        self.assertAlmostEqual(fused[0], .3)
        self.assertAlmostEqual(fused[1], .7)
        result = blend_predictions(global_rows, [local[1],local[3]], allow_subset=True)
        self.assertEqual(result[0], ["test_1.jpg", "test_3.jpg"])

    def test_mismatch_nan_duplicates_and_labels_rejected(self):
        first = self.rows([.2, .8]); second = self.rows([.4, .6])
        for bad in (list(reversed(second)), [second[0],second[0]], [dict(second[0], Probability="nan"), second[1]],
                    [dict(second[0], True_Label="1"), second[1]]):
            with self.assertRaises(ValueError): blend_predictions(first, bad, labeled=True)
        with self.assertRaises(ValueError): blend_predictions(first, second[:1])
        with self.assertRaises(ValueError): blend_predictions(first, list(reversed(second)), allow_subset=True)


if __name__ == "__main__": unittest.main()
