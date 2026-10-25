"""Fine-tuning compatibility checks. Does not import PyTorch."""
import math
import hashlib
import json
from pathlib import Path
from .model_config import canonical_config, checkpoint_model_name


def validate_source(source, config, signature, checkpoint, run):
    checkpoint, run = Path(checkpoint).resolve(), Path(run).resolve()
    if checkpoint == run or run in checkpoint.parents:
        raise ValueError("Initial checkpoint must be outside the new run directory")
    if checkpoint_model_name(source) != config["model_name"]:
        raise ValueError("Initial checkpoint architecture differs from fine-tuning config")
    original = canonical_config(source["config"])
    if any(original.get(k, 0) for k in ("train_limit", "valid_limit", "test_limit")):
        raise ValueError("Initial checkpoint must be a full-data model, not a smoke model")
    for key in ("image_size", "remove_black_border", "black_border_threshold", "black_border_margin",
                "black_border_version", "pos_weight", "amp"):
        if original[key] != config[key]:
            raise ValueError(f"Fine-tuning must preserve source {key}")
    if source["data_signature"] != signature:
        raise ValueError("Initial checkpoint dataset signatures differ")
    age = source["age_stats"]
    if not all(type(age.get(k)) in (int, float) and math.isfinite(age[k]) for k in ("median", "mean", "std")) or age["std"] <= 0:
        raise ValueError("Invalid source age statistics")
    best = source.get("best_metrics") or {}
    if not all(type(best.get(k)) in (int, float) and math.isfinite(best[k]) and 0 <= best[k] <= 1
               for k in ("auroc", "auprc_trapezoid")):
        raise ValueError("Source checkpoint needs valid saved validation metrics")


def verify_baseline(result, saved, sampled, tolerance):
    if not sampled:
        for key in ("auroc", "auprc_trapezoid"):
            if abs(result[key] - saved[key]) > tolerance:
                raise ValueError(f"Source validation {key} was not reproduced: {result[key]} vs {saved[key]}. "
                                 "Check checkpoint, data and inference environment before training.")


def reconcile_best(run, state):
    """Recover newer committed best metadata without changing training weights/RNG.

    best.pt is committed before the next last.pt. A crash in between must never
    lower the best-score guard when last.pt is resumed.
    """
    run = Path(run)
    try:
        manifest = json.loads((run / "best_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("initialization") != state["initialization"]:
            raise ValueError("Best manifest is incomplete or belongs to another initialization")
        for name in ("best.pt", "valid_predictions.csv", "best_metrics.json", "valid_thresholds.csv"):
            if hashlib.sha256((run / name).read_bytes()).hexdigest() != manifest["files"][name]:
                raise ValueError(f"Best artifact hash mismatch: {name}")
        best = json.loads((run / "best_metrics.json").read_text(encoding="utf-8"))
        if best["epoch"] != manifest["epoch"]:
            raise ValueError("Best epoch disagrees with its manifest")
        score = tuple(best[k] for k in ("auroc", "auprc_trapezoid"))
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in score):
            raise ValueError("Invalid committed best metrics")
        if score < tuple(state["best_score"]):
            raise ValueError("Best artifacts are older/worse than last.pt; restore a consistent snapshot")
        return {**state, "best_score": list(score), "best_metrics": best}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot resume incomplete/corrupted best artifacts: {error}") from error
