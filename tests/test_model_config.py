"""Check architecture metadata and construction using tiny API stubs, not PyTorch."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glaucoma_training.model_config import (canonical_config, checkpoint_model_name,
    default_run_name, load_config, model_name_from_config)


class ModelConfigChecks(unittest.TestCase):
    def test_legacy_checkpoint_uses_resnet(self):
        self.assertEqual(checkpoint_model_name({"config": {"image_size": 320}}), "resnet18")
        self.assertEqual(canonical_config({"image_size": 320})["model_name"], "resnet18")

    def test_explicit_architecture_and_inconsistent_metadata(self):
        state = {"config": {"model_name": "efficientnet_b2"}, "model_name": "efficientnet_b2"}
        self.assertEqual(checkpoint_model_name(state), "efficientnet_b2")
        with self.assertRaises(ValueError):
            checkpoint_model_name({**state, "model_name": "resnet18"})
        with self.assertRaises(ValueError):
            model_name_from_config({"model_name": "not-a-supported-model"})

    def test_run_directories_are_separate(self):
        self.assertEqual(default_run_name("resnet18"), "baseline")
        self.assertEqual(default_run_name("resnet18", True), "smoke")
        self.assertEqual(default_run_name("efficientnet_b2"), "efficientnet_b2")
        self.assertEqual(default_run_name("efficientnet_b2", True), "efficientnet_b2_smoke")

    def test_only_backbone_changes_in_full_config(self):
        candidate = load_config()
        baseline = load_config(ROOT / "configs" / "resnet18.json")
        self.assertEqual(candidate["model_name"], "efficientnet_b2")
        self.assertEqual(baseline["model_name"], "resnet18")
        self.assertEqual({k: v for k, v in candidate.items() if k != "model_name"},
                         {k: v for k, v in baseline.items() if k != "model_name"})

    def test_smoke_config_inherits_backbone(self):
        self.assertEqual(load_config(ROOT / "configs" / "smoke.json")["model_name"], "efficientnet_b2")
        self.assertEqual(load_config(ROOT / "configs" / "resnet18_smoke.json")["model_name"], "resnet18")

    def test_unknown_and_non_object_config_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for data in ({"typo": 1}, [], {"model_name": "bad"}):
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_backbone_feature_dimensions_without_loading_framework(self):
        # Execute only our constructor against known torchvision API-shaped stubs.
        # No models/weights/tensors are allocated and no framework is imported.
        class Module:
            pass

        class Linear:
            def __init__(self, inputs, outputs):
                self.in_features, self.out_features = inputs, outputs

        class Identity:
            pass

        fake_nn = SimpleNamespace(Module=Module, Linear=Linear, Identity=Identity,
                                  Sequential=lambda *layers: layers, ReLU=lambda: None,
                                  Dropout=lambda rate: None)
        calls = []

        def resnet18(weights):
            calls.append(("resnet18", weights))
            return SimpleNamespace(fc=Linear(512, 1000))

        def efficientnet_b2(weights):
            calls.append(("efficientnet_b2", weights))
            return SimpleNamespace(classifier=[None, Linear(1408, 1000)])

        tree = ast.parse((ROOT / "src" / "glaucoma_training" / "common.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusionModel")
        namespace = dict(nn=fake_nn, model_name_from_config=model_name_from_config,
                         resnet18=resnet18, efficientnet_b2=efficientnet_b2,
                         ResNet18_Weights=SimpleNamespace(DEFAULT="resnet-default"),
                         EfficientNet_B2_Weights=SimpleNamespace(DEFAULT="b2-default"))
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<constructor-check>", "exec"), namespace)
        constructor = namespace["FusionModel"]
        baseline = constructor(model_name="resnet18")
        candidate = constructor(pretrained=True, model_name="efficientnet_b2")
        self.assertEqual(baseline.head[0].in_features, 514)
        self.assertEqual(candidate.head[0].in_features, 1410)
        self.assertIsInstance(baseline.backbone.fc, Identity)
        self.assertIsInstance(candidate.backbone.classifier, Identity)
        self.assertEqual(calls, [("resnet18", None), ("efficientnet_b2", "b2-default")])


if __name__ == "__main__":
    unittest.main()
