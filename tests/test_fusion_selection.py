"""Real selection/publication on synthetic probabilities, no torch/model execution."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glaucoma_training.data_utils import json_save, metrics, write_predictions
from glaucoma_training.fusion_selection import FusionSelection, digest
from glaucoma_training import launch


class FusionSelectionChecks(unittest.TestCase):
    def fixture(self, root):
        reference = root / "global"; reference.mkdir()
        run = root / "run"; run.mkdir()
        names = list("abcd"); labels = [0, 1, 0, 1]
        rows = [dict(Image_name=n, True_Label=str(y)) for n,y in zip(names,labels)]
        write_predictions(reference/"validation_predictions.csv", names, [.1,.3,.7,.9], labels=labels)
        json_save(reference/"manifest.json", dict(status="completed", data_signature={"train":"fixed"},
            config=dict(use_age=True, input_view="global"), sampled=False, checkpoint_sha256="a"*64,
            validation_sha256=digest(reference/"validation_predictions.csv")))
        return run, reference, rows, names, labels

    def writer(self, checkpoint, path):
        path.write_text(json.dumps(checkpoint))

    def test_selects_complementarity_instead_of_local_auc_and_records_each_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, ref, rows, names, labels = self.fixture(Path(temporary))
            selection = FusionSelection(run, ref, {"train":"fixed"}, rows)
            selection.snapshot()
            local_a, local_b = [.48,.52,.49,.51], [.05,.95,.5,.45]
            self.assertGreater(metrics(labels,local_a)["auroc"], metrics(labels,local_b)["auroc"])
            for epoch,probs in enumerate((local_a,local_b),1):
                result = {**metrics(labels, probs), "epoch":epoch}
                selection.record(names, probs, labels, result, dict(epoch=epoch), self.writer)
            self.assertEqual(selection.state()["best_epoch"], 2)
            self.assertEqual(selection.state()["best_score"][0], 1)
            self.assertEqual(json.loads((run/"best_fusion.pt").read_bytes())["epoch"],2)
            self.assertTrue((run/"validation_epochs/epoch_0001/local.csv").exists())
            self.assertTrue((run/"validation_epochs/epoch_0002/fusion.csv").exists())
            # A crash after publishing the generation but before next last.pt: recover newer best.
            saved = {**selection.state(), "best_score":[.75,.5], "best_epoch":1}
            (run/"best_fusion.pt").unlink()
            resumed = FusionSelection(run, None, {"train":"fixed"}, rows, saved)
            self.assertEqual(resumed.state(), selection.state())
            self.assertTrue((run/"best_fusion.pt").exists())
            before = digest(run/"best_fusion.pt")
            resumed.record(names,local_a,labels,dict(epoch=3),dict(epoch=3),self.writer)
            self.assertEqual(digest(run/"best_fusion.pt"),before)

    def test_rejects_changed_reference_order_labels_and_missing_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, ref, rows, names, labels = self.fixture(Path(temporary))
            for changed in (rows[::-1], [{**rows[0],"True_Label":"1"}]+rows[1:]):
                with self.assertRaises(ValueError): FusionSelection(run, ref, {"train":"fixed"}, changed)
            with self.assertRaises(ValueError): FusionSelection(run, ref, {}, rows)
            with self.assertRaises(ValueError): FusionSelection(run, ref, {"train":"fixed"}, rows, dict(reference_signature="wrong"))
            path=ref/"manifest.json"; manifest=json.loads(path.read_bytes()); manifest.pop("validation_sha256")
            json_save(path,manifest)
            with self.assertRaises(ValueError): FusionSelection(run, ref, {"train":"fixed"}, rows)

    def test_corrupted_or_missing_committed_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, ref, rows, names, labels = self.fixture(Path(temporary))
            selection=FusionSelection(run,ref,{"train":"fixed"},rows)
            selection.snapshot()
            selection.record(names,[.1,.9,.2,.8],labels,dict(epoch=1),dict(epoch=1),self.writer)
            saved=selection.state()
            committed=json.loads((run/"best_fusion_manifest.json").read_bytes())
            (run/committed["generation"]/"best_fusion.pt").write_bytes(b"corrupt")
            with self.assertRaises(ValueError): FusionSelection(run,None,{"train":"fixed"},rows,saved)
            (run/"best_fusion_manifest.json").unlink()
            with self.assertRaises(ValueError): FusionSelection(run,None,{"train":"fixed"},rows,saved)

    def test_invalid_lengths_and_nan_do_not_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, ref, rows, names, labels=self.fixture(Path(temporary))
            selection=FusionSelection(run,ref,{"train":"fixed"},rows)
            for probs in ([.1], [.1,float("nan"),.3,.5]):
                with self.assertRaises(ValueError): selection.record(names,probs,labels,dict(epoch=1),{},self.writer)
            self.assertFalse((run/"best_fusion_manifest.json").exists())

    def test_failed_generation_keeps_previous_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, ref, rows, names, labels=self.fixture(Path(temporary))
            selection=FusionSelection(run,ref,{"train":"fixed"},rows)
            selection.snapshot()
            selection.record(names,[.48,.52,.49,.51],labels,dict(epoch=1),dict(epoch=1),self.writer)
            saved=selection.state(); before=digest(run/"best_fusion.pt")
            def fail(checkpoint, path):
                path.write_bytes(b"partial")
                raise OSError("disk failure")
            with self.assertRaises(OSError):
                selection.record(names,[.05,.95,.5,.45],labels,dict(epoch=2),dict(epoch=2),fail)
            resumed=FusionSelection(run,None,{"train":"fixed"},rows,saved)
            self.assertEqual(resumed.state()["best_epoch"],1)
            self.assertEqual(digest(run/"best_fusion.pt"),before)

    def test_launcher_forwards_reference_and_uses_separate_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary); commands=[]
            def fake_run(session, title, command, **kwargs):
                if title == "training":
                    commands.append(command)
                    run=Path(command[command.index("--run-dir")+1]); run.mkdir(parents=True)
                    json_save(run/"progress.json",dict(status="paused",stop_reason="time_budget",epoch_index=0,total_epochs=8,next_batch=0))
                    return 3
                return 0
            argv=["launch","train","--config",str(ROOT/"configs/local_b0.json"),"--fusion-reference-evaluation",str(folder/"reference")]
            with patch.object(launch,"CODE",folder),patch.object(sys,"argv",argv),patch.object(launch.signal,"signal"),patch.object(launch.LoggedSession,"run",fake_run):
                self.assertEqual(launch.main(),3)
            command=commands[0]
            self.assertTrue(command[command.index("--run-dir")+1].endswith("_fusionselect"))
            self.assertEqual(command[command.index("--fusion-reference-evaluation")+1],str((folder/"reference").resolve()))


if __name__ == "__main__": unittest.main()
