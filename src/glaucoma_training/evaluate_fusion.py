"""Frozen global checkpoint + local checkpoint evaluation, fixed 50/50 probabilities."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import traceback

from . import DEFAULT_DATA_ROOT, PROJECT_ROOT
from .data_utils import data_signature, json_save, metrics, write_predictions
from .fusion_utils import blend_predictions
from .launch import LoggedSession
from .run_utils import protect_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-checkpoint", type=Path, required=True)
    parser.add_argument("--local-checkpoint", type=Path, required=True)
    parser.add_argument("--roi-cache", type=Path, help="Defaults to roi_cache beside the local checkpoint")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--allow-sample", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=["valid", "test"], default=["valid", "test"])
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.global_checkpoint = args.global_checkpoint.resolve()
    args.local_checkpoint = args.local_checkpoint.resolve()
    if args.roi_cache is not None: args.roi_cache = args.roi_cache.resolve()
    if not 0 <= args.threshold <= 1 or args.batch_size <= 0 or args.workers < 0:
        parser.error("Invalid threshold/batch-size/workers")
    if args.global_checkpoint.resolve() == args.local_checkpoint.resolve():
        parser.error("Global and local checkpoints must differ")
    hashes = {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in
              (("global", args.global_checkpoint), ("local", args.local_checkpoint))}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = protect_output(args.data_root, args.output_dir or args.local_checkpoint.parent / "fusion_evaluations" / stamp, directory=True)
    output.mkdir(parents=True, exist_ok=False)
    session = LoggedSession(output / "logs")
    prior_signals = {sig:signal.signal(sig, session.forward_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    report = dict(status="running", start_utc=datetime.now(timezone.utc).isoformat(),
                  combination="0.5 * global_probability + 0.5 * local_probability",
                  local_selection="best local validation AUROC, then AUPRC; not selected by fused AUROC",
                  checkpoint_sha256=hashes, global_checkpoint=str(args.global_checkpoint.resolve()),
                  local_checkpoint=str(args.local_checkpoint.resolve()), threshold=args.threshold,
                  sampled=False, data_signature=data_signature(args.data_root), splits=list(dict.fromkeys(args.splits)))
    json_save(output / "status.json", report)
    code = 0
    try:
        manifests = {}
        for branch, checkpoint in (("global", args.global_checkpoint), ("local", args.local_checkpoint)):
            command = [sys.executable, "-u", "-m", "glaucoma_training.evaluate", "--data-root", str(args.data_root),
                       "--checkpoint", str(checkpoint), "--output-dir", str(output / branch), "--device", args.device,
                       "--batch-size", str(args.batch_size), "--workers", str(args.workers),
                       "--threshold", str(args.threshold), "--splits", *args.splits]
            if branch == "local": command.extend(["--roi-cache", str(args.roi_cache or args.local_checkpoint.parent / "roi_cache")])
            if args.allow_sample: command.append("--allow-sample")
            session.run(branch + " evaluation", command)
            manifest = json.loads((output / branch / "manifest.json").read_bytes())
            if manifest["status"] != "completed" or manifest["checkpoint_sha256"] != hashes[branch] or \
                    manifest["data_signature"] != report["data_signature"]:
                raise ValueError("Branch provenance mismatch")
            manifests[branch] = manifest
        if manifests["global"]["config"].get("input_view", "global") != "global" or \
                not manifests["global"]["config"].get("use_age", True):
            raise ValueError("Reference must be the frozen global image+age model")
        if manifests["local"]["config"].get("input_view") != "local" or manifests["local"]["config"].get("use_age", True):
            raise ValueError("Local checkpoint must be an image-only cached-ROI model")
        selection = manifests["local"].get("selection")
        if selection:
            if selection["global_checkpoint_sha256"] != hashes["global"]:
                raise ValueError("Fusion-selected checkpoint requires its original frozen global model")
            report["local_selection"] = selection["criterion"]
            report["selection"] = selection
        sampled = manifests["local"]["sampled"]
        if manifests["global"]["sampled"] and not sampled:
            raise ValueError("A debug global model cannot produce a full fusion submission")
        report.update(sampled=sampled, roi_cache_signature=manifests["local"]["roi_cache_signature"],
                      branch_manifest_sha256={b:hashlib.sha256((output/b/"manifest.json").read_bytes()).hexdigest() for b in manifests},
                      files={})
        for split in report["splits"]:
            rows = {}
            for branch, manifest in manifests.items():
                filename = "validation_predictions.csv" if split == "valid" else manifest["prediction_file"]
                path = output / branch / filename
                if split == "test" and hashlib.sha256(path.read_bytes()).hexdigest() != manifest["prediction_sha256"]:
                    raise ValueError("Branch prediction hash mismatch")
                with path.open(encoding="utf-8", newline="") as stream: rows[branch] = list(csv.DictReader(stream))
            names, global_probs, local_probs, probabilities, labels = blend_predictions(
                rows["global"], rows["local"], allow_subset=sampled, labeled=split == "valid")
            filename = "validation_predictions.csv" if split == "valid" else "DEBUG_predictions.csv" if sampled else "submission.csv"
            write_predictions(output / filename, names, probabilities, args.threshold, labels)
            report["files"][filename] = dict(rows=len(names), sha256=hashlib.sha256((output/filename).read_bytes()).hexdigest())
            if split == "valid":
                result = {name:metrics(labels, probs, args.threshold) for name,probs in
                          (("global", global_probs), ("local", local_probs), ("fusion", probabilities))}
                json_save(output / "validation_metrics.json", result)
                report["validation"] = result
                session.log("Same-row validation metrics: " + json.dumps(result))
        if session.interrupted: raise InterruptedError("Evaluation interrupted")
        for branch, path in (("global", args.global_checkpoint), ("local", args.local_checkpoint)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[branch]:
                raise ValueError("Checkpoint changed during fusion evaluation; retry after training stops")
        report.update(status="completed", end_utc=datetime.now(timezone.utc).isoformat())
        json_save(output / "manifest.json", report)
        session.log(f"Fusion evaluation complete: {output}")
    except Exception as error:
        code = 130 if session.interrupted else error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        if code < 0: code = 128 - code
        report.update(status="interrupted" if session.interrupted else "failed", error=str(error))
        session.log(traceback.format_exc())
    finally:
        report["exit_code"] = code
        json_save(output / "status.json", report)
        session.stream.close()
        for sig, handler in prior_signals.items(): signal.signal(sig, handler)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
