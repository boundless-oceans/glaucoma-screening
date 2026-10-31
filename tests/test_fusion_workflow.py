"""Exercise fixed fusion orchestration with child-evaluation stubs, never PyTorch."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glaucoma_training import evaluate_fusion
from glaucoma_training.data_utils import data_signature, json_save, write_predictions


class FusionWorkflowChecks(unittest.TestCase):
    def exercise(self, sampled, corrupt=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); data = root / "data"; data.mkdir()
            for split in ("train", "valid", "test"):
                (data / f"{split}.csv").write_text("Image_name,Age,True_Label\na.jpg,50,0\nb.jpg,60,1\nc.jpg,50,0\nd.jpg,60,1\n")
            paths = {b:root / (b + ".pt") for b in ("global", "local")}
            for b,p in paths.items(): p.write_bytes(b.encode())
            signature = data_signature(data); cache = root / "cache"
            commands = []

            def fake_run(session, title, command, **kwargs):
                branch = title.split()[0]; commands.append(command)
                self.assertEqual("--roi-cache" in command, branch == "local")
                folder = Path(command[command.index("--output-dir") + 1]); folder.mkdir()
                names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
                labels, probabilities = [0,1,0,1], [.2,.8,.3,.7] if branch == "global" else [.4,.9,.2,.6]
                if sampled and branch == "local": names,labels,probabilities = [names[0],names[3]], [0,1], [.4,.6]
                write_predictions(folder/"validation_predictions.csv", names, probabilities, labels=labels)
                filename = "DEBUG_predictions.csv" if sampled and branch == "local" else "submission.csv"
                write_predictions(folder/filename, names, probabilities)
                json_save(folder/"manifest.json", dict(status="completed", data_signature=signature,
                    checkpoint_sha256=hashlib.sha256(paths[branch].read_bytes()).hexdigest(),
                    config=dict(input_view="local" if branch == "local" else "global", use_age=branch == "global"),
                    sampled=sampled and branch == "local", roi_cache_signature="fixed-cache" if branch == "local" else None,
                    prediction_file=filename, prediction_sha256=hashlib.sha256((folder/filename).read_bytes()).hexdigest()))
                if corrupt and branch == "local": paths["global"].write_bytes(b"external replacement")
                return 0

            output = root / "fusion"
            argv = ["fusion", "--data-root", str(data), "--global-checkpoint", str(paths["global"]),
                    "--local-checkpoint", str(paths["local"]), "--roi-cache", str(cache), "--output-dir", str(output)]
            if sampled: argv.append("--allow-sample")
            with patch.object(sys, "argv", argv), patch.object(evaluate_fusion.signal, "signal"), \
                    patch.object(evaluate_fusion.LoggedSession, "run", fake_run):
                self.assertEqual(evaluate_fusion.main(), 1 if corrupt else 0)
            if corrupt:
                self.assertFalse((output/"manifest.json").exists())
                self.assertEqual(json.loads((output/"status.json").read_bytes())["status"], "failed")
                return
            manifest = json.loads((output/"manifest.json").read_bytes())
            filename = "DEBUG_predictions.csv" if sampled else "submission.csv"
            self.assertEqual(manifest["files"][filename]["rows"], 2 if sampled else 4)
            self.assertEqual(set(manifest["validation"]), {"global", "local", "fusion"})
            self.assertEqual(paths["global"].read_bytes(), b"global")
            self.assertEqual(manifest["status"], "completed")

    def test_full_global_local_fusion_and_provenance(self): self.exercise(False)

    def test_debug_subset_fusion_and_filename(self): self.exercise(True)

    def test_checkpoint_change_prevents_completion_manifest(self): self.exercise(False, corrupt=True)


if __name__ == "__main__": unittest.main()
