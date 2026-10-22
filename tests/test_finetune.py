"""Fine-tuning guards and initialization wiring. Never import a model framework."""
import ast
import copy
from pathlib import Path
import sys
import unittest
import json
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glaucoma_training import launch
from glaucoma_training.finetune import validate_source, verify_baseline
from glaucoma_training.model_config import canonical_config, default_run_name, load_config
from glaucoma_training.run_utils import validate_config


class FinetuneChecks(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/finetune.json")
        self.signature = dict(train="a", valid="b", test="c")
        self.source = dict(model_name="efficientnet_b2", config=load_config(), data_signature=self.signature,
                           age_stats=dict(median=57, mean=56, std=10), epoch=5,
                           best_metrics=dict(auroc=0.981308, auprc_trapezoid=0.758498))

    def check(self, source=None, config=None):
        validate_source(source or self.source, config or self.config, self.signature,
                        ROOT / "runs/source/best.pt", ROOT / "runs/new")

    def test_finetune_changes_only_initialization_lr_and_budget(self):
        validate_config(self.config)
        differences = {k for k in self.config if self.config[k] != load_config()[k]}
        self.assertEqual(differences, {"finetune", "pretrained", "learning_rate", "epochs"})
        self.assertEqual(self.config["learning_rate"], 1e-5)
        self.assertEqual(self.config["epochs"], 3)
        self.check()

    def test_legacy_checkpoint_and_run_names_remain_compatible(self):
        old = load_config()
        old.pop("finetune")
        self.assertEqual(canonical_config(old), load_config())
        self.assertEqual(default_run_name("efficientnet_b2", False, True), "efficientnet_b2_blackborder")
        self.assertEqual(default_run_name("efficientnet_b2", False, True, True), "efficientnet_b2_blackborder_finetune")
        self.assertEqual(default_run_name("efficientnet_b2", True, True, True), "efficientnet_b2_blackborder_smoke_finetune")

    def test_source_requires_same_architecture_data_and_full_training(self):
        for section, key, value in [("config", "model_name", "resnet18"),
                                    ("config", "train_limit", 256), ("data_signature", "test", "changed")]:
            source = copy.deepcopy(self.source)
            source[section][key] = value
            with self.assertRaises(ValueError): self.check(source=source)

    def test_source_preprocessing_and_loss_are_preserved(self):
        for key, value in [("image_size", 384), ("remove_black_border", False),
                           ("black_border_threshold", 10), ("black_border_margin", 0.1),
                           ("pos_weight", 2), ("amp", False)]:
            config = {**self.config, key: value}
            with self.assertRaises(ValueError): self.check(config=config)

    def test_source_cannot_be_in_destination(self):
        with self.assertRaises(ValueError):
            validate_source(self.source, self.config, self.signature, ROOT / "runs/new/best.pt", ROOT / "runs/new")

    def test_age_and_source_validation_metadata_are_required(self):
        for section, key, value in [("age_stats", "std", 0), ("age_stats", "mean", float("nan")),
                                    ("best_metrics", "auroc", float("nan"))]:
            source = copy.deepcopy(self.source)
            source[section][key] = value
            with self.assertRaises(ValueError): self.check(source=source)

    def test_full_baseline_must_reproduce_both_metrics(self):
        saved = self.source["best_metrics"]
        verify_baseline(saved, saved, False, 1e-5)
        for key in saved:
            changed = {**saved, key: saved[key] - 0.001}
            with self.assertRaises(ValueError): verify_baseline(changed, saved, False, 1e-5)
        # A smoke subset is explicitly marked, not compared with the full score.
        verify_baseline(dict(auroc=0.5, auprc_trapezoid=0.1), saved, True, 1e-5)

    def test_smoke_preserves_image_pipeline(self):
        smoke = load_config(ROOT / "configs/finetune_smoke.json")
        self.check(config=smoke)
        self.assertEqual(smoke["image_size"], 320)
        self.assertEqual(smoke["batch_size"], 32)

    def test_model_wiring_is_strict_and_starts_with_fresh_optimizer(self):
        text = (ROOT / "src/glaucoma_training/train.py").read_text(encoding="utf-8-sig")
        tree = ast.parse(text)
        loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict"]
        source_load = next(node for node in loads if ast.unparse(node.args[0]) == "source['model']")
        self.assertTrue(next(k.value.value for k in source_load.keywords if k.arg == "strict"))
        # Optimizer restore occurs only under `if state` (resume), not source initialization.
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        restores = [node for node in main.body if isinstance(node, ast.If) and ast.unparse(node.test) == "state"
                    and any(isinstance(c, ast.Call) and ast.unparse(c.func) == "optimizer.load_state_dict" for c in ast.walk(node))]
        self.assertEqual(len(restores), 1)

    def test_launcher_forwards_initialization_and_records_budget_pause(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            run = folder / "run"
            run.mkdir()
            (run / "progress.json").write_text(json.dumps(dict(status="paused", epoch_index=0,
                                                              next_batch=0, total_epochs=3, stop_reason="time_budget")))
            commands = []

            def fake_run(session, name, command, **kwargs):
                commands.append(command)
                if name == "training":
                    self.assertIn(3, kwargs["accepted_codes"])
                    return 3
                return 0

            args = ["launch", "train", "--config", str(ROOT / "configs/finetune.json"),
                    "--run-dir", str(run), "--init-checkpoint", str(folder / "original.pt")]
            with patch.object(launch, "CODE", folder), patch.object(sys, "argv", args), \
                    patch.object(launch.signal, "signal"), patch.object(launch.LoggedSession, "run", fake_run):
                self.assertEqual(launch.main(), 3)
            self.assertIn("--init-checkpoint", commands[-1])
            status = json.loads(next((folder / "logs").glob("*/status.json")).read_text())
            self.assertEqual((status["status"], status["exit_code"]), ("paused", 3))

    def test_launcher_interruption_exception_returns_130(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)

            def stop(session, *args, **kwargs):
                session.interrupted = True
                raise InterruptedError("stop")

            args = ["launch", "train", "--config", str(ROOT / "configs/finetune.json")]
            with patch.object(launch, "CODE", folder), patch.object(sys, "argv", args), \
                    patch.object(launch.signal, "signal"), patch.object(launch.LoggedSession, "run", stop):
                self.assertEqual(launch.main(), 130)


if __name__ == "__main__":
    unittest.main()
