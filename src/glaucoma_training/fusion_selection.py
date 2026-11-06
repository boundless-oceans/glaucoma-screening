"""Cached validation-only fusion selection and recoverable checkpoint publication.

No torch import: the training caller supplies the checkpoint writer.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

from .data_utils import json_save, metrics, write_predictions
from .fusion_utils import blend_predictions


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_alias(source, destination):
    """Atomic convenience alias; generation files remain the source of truth."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


class FusionSelection:
    def __init__(self, run, reference, signature, valid_rows, saved=None, sampled=False):
        self.run = Path(run)
        self.reference = Path(reference or self.run / "fusion_reference")
        raw = (self.reference / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        predictions = self.reference / "validation_predictions.csv"
        config = manifest["config"]
        if (manifest.get("status") != "completed" or manifest["data_signature"] != signature
                or config.get("input_view", "global") != "global" or not config.get("use_age", True)
                or manifest.get("experimental_decoder_override", False)
                or (manifest.get("sampled", False) and not sampled)):
            raise ValueError("Fusion reference must be a matching completed global image+age validation")
        if not manifest.get("validation_sha256") or digest(predictions) != manifest["validation_sha256"]:
            raise ValueError("Fusion reference needs a verified validation_sha256; reevaluate once with current code")
        self.reference_signature = hashlib.sha256(raw + predictions.read_bytes()).hexdigest()
        self.global_sha256 = manifest["checkpoint_sha256"]
        if len(self.global_sha256) != 64:
            raise ValueError("Invalid reference checkpoint hash")
        with predictions.open(encoding="utf-8", newline="") as stream:
            self.global_rows = list(csv.DictReader(stream))
        # Exact full order, or ordered smoke subset, with labels checked before any updates.
        local = [dict(Image_name=r["Image_name"], True_Label=r["True_Label"], Probability="0") for r in valid_rows]
        blend_predictions(self.global_rows, local, allow_subset=sampled, labeled=True)
        self.names = [r["Image_name"] for r in valid_rows]
        self.labels = [int(r["True_Label"]) for r in valid_rows]
        self.sampled = sampled
        self.best = None
        if saved and saved["reference_signature"] != self.reference_signature:
            raise ValueError("Fusion reference changed since last.pt")
        marker = self.run / "best_fusion_manifest.json"
        if marker.exists():
            committed = json.loads(marker.read_bytes())
            if committed.get("status") != "completed" or committed["reference_signature"] != self.reference_signature:
                raise ValueError("Committed fusion checkpoint uses another reference")
            generation = Path(committed["generation"])
            if generation.is_absolute() or len(generation.parts) != 2 or generation.parts[0] != "fusion_checkpoints" or ".." in generation.parts:
                raise ValueError("Invalid fusion generation path")
            for name, expected in committed["files"].items():
                if name not in ("best_fusion.pt", "local.csv", "fusion.csv", "metrics.json"):
                    raise ValueError("Unexpected fusion artifact")
                if digest(self.run / generation / name) != expected:
                    raise ValueError("Committed fusion artifacts are incomplete/corrupted")
            if set(committed["files"]) != {"best_fusion.pt", "local.csv", "fusion.csv", "metrics.json"}:
                raise ValueError("Incomplete fusion file manifest")
            report = json.loads((self.run / generation / "metrics.json").read_bytes())
            score = [report["fusion"][k] for k in ("auroc", "auprc_trapezoid")]
            if score != committed["score"] or report["epoch"] != committed["epoch"] or report["reference_signature"] != self.reference_signature:
                raise ValueError("Fusion pointer disagrees with committed metrics")
            self.best = committed
            if saved and saved.get("best_score") and tuple(saved["best_score"]) > tuple(committed["score"]):
                raise ValueError("Fusion artifacts are older than last.pt")
            publish_alias(self.run / generation / "best_fusion.pt", self.run / "best_fusion.pt")
        elif saved and saved.get("best_score"):
            raise ValueError("Restore missing best_fusion_manifest.json before resuming")

    def snapshot(self):
        destination = self.run / "fusion_reference"
        destination.mkdir(exist_ok=True)
        if self.reference.resolve() == destination.resolve():
            return
        for name in ("validation_predictions.csv", "manifest.json"):
            source = self.reference / name
            temporary = destination / (name + ".tmp")
            shutil.copyfile(source, temporary)
            if digest(temporary) != digest(source):
                raise ValueError("Reference changed during snapshot")
            os.replace(temporary, destination / name)
        raw = (destination / "manifest.json").read_bytes() + (destination / "validation_predictions.csv").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.reference_signature:
            raise ValueError("Reference changed after initialization")

    def state(self):
        return dict(reference_signature=self.reference_signature,
                    best_score=self.best["score"] if self.best else None,
                    best_epoch=self.best["epoch"] if self.best else None,
                    global_checkpoint_sha256=self.global_sha256)

    def record(self, names, probabilities, labels, result, checkpoint, writer):
        if names != self.names or list(labels) != self.labels or len(probabilities) != len(names):
            raise ValueError("Validation order/labels changed during training")
        rows = [dict(Image_name=n, True_Label=str(int(y)), Probability=f"{float(p):.10f}")
                for n,p,y in zip(names, probabilities, labels)]
        _, _, _, fused, _ = blend_predictions(self.global_rows, rows, self.sampled, labeled=True)
        fused_metrics = metrics(labels, fused)
        score = [fused_metrics[k] for k in ("auroc", "auprc_trapezoid")]
        epoch = result["epoch"]
        epoch_dir = self.run / "validation_epochs" / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        write_predictions(epoch_dir / "local.csv", names, probabilities, labels=labels)
        write_predictions(epoch_dir / "fusion.csv", names, fused, labels=labels)
        report = dict(epoch=epoch, local=result, fusion=fused_metrics,
                      reference_signature=self.reference_signature)
        json_save(epoch_dir / "metrics.json", report)
        if self.best is None or tuple(score) > tuple(self.best["score"]):
            # Unique immutable generation; a crash before pointer publication leaves an ignored orphan.
            import uuid
            generation = Path("fusion_checkpoints") / (f"epoch_{epoch:04d}_" + uuid.uuid4().hex)
            folder = self.run / generation
            folder.mkdir(parents=True)
            selection = dict(criterion="fusion_auroc_then_auprc", global_checkpoint_sha256=self.global_sha256,
                             reference_signature=self.reference_signature, fusion_metrics=fused_metrics)
            writer({**checkpoint, "selection": selection}, folder / "best_fusion.pt")
            for name in ("local.csv", "fusion.csv", "metrics.json"):
                shutil.copyfile(epoch_dir / name, folder / name)
            committed = dict(status="completed", generation=generation.as_posix(), epoch=epoch, score=score,
                             reference_signature=self.reference_signature,
                             files={name:digest(folder / name) for name in ("best_fusion.pt", "local.csv", "fusion.csv", "metrics.json")})
            json_save(self.run / "best_fusion_manifest.json", committed)
            self.best = committed
            publish_alias(folder / "best_fusion.pt", self.run / "best_fusion.pt")
        return fused_metrics
