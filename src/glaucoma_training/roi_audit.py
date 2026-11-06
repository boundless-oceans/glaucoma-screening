"""Summarize frozen ROI diagnostics; no model framework or localization claims."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize_report(report):
    if report.get("status") != "completed" or report.get("split") not in ("train", "valid"):
        raise ValueError("Only completed train/valid ROI reports can be audited")
    images = report.get("images", [])
    names = [row["image_name"].casefold() for row in images]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty ROI report or duplicate image names")
    by_label = {}
    times, failures = [], Counter()
    for row in images:
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError("Invalid diagnostic label")
        selection = row["selection"]
        if type(selection["fallback"]) is not bool:
            raise ValueError("Invalid fallback flag")
        if not selection["fallback"] and selection.get("roi_box") is None:
            raise ValueError("Accepted candidate lacks full-resolution coordinates")
        seconds = row["preprocess_and_cam_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid diagnostic timing")
        times.append(seconds * 1000)
        counts = by_label.setdefault(str(row["label"]), dict(images=0, accepted=0, fallback=0))
        counts["images"] += 1
        counts["fallback" if selection["fallback"] else "accepted"] += 1
        # Conditions can overlap. These are policy rejection causes, not anatomical truth.
        for key, limit in (("attention_mass", "min_attention_mass"),
                           ("concentration_gain", "min_concentration_gain"),
                           ("foreground_fraction", "min_foreground_fraction")):
            value = selection.get(key)
            if value is not None:
                if not math.isfinite(value):
                    raise ValueError("Nonfinite ROI quality diagnostic")
                if value < selection["policy"][limit]:
                    failures[key + "_below_threshold"] += 1
    accepted = sum(c["accepted"] for c in by_label.values())
    if accepted != report["candidates"]:
        raise ValueError("Candidate count disagrees with image records")
    for counts in by_label.values():
        counts["acceptance_fraction"] = counts["accepted"] / counts["images"]
    return dict(split=report["split"], sampling=report["sampling"], seed=report["seed"],
                checkpoint_sha256=report["checkpoint_sha256"], roi_policy=report["roi_policy"],
                images=len(images), accepted=accepted, fallback=len(images) - accepted,
                acceptance_fraction=accepted / len(images), by_label=by_label,
                reasons=dict(Counter(row["selection"]["reason"] for row in images)),
                failed_conditions=dict(failures),
                timing_ms=dict(mean=statistics.mean(times), median=statistics.median(times), max=max(times)),
                timing_scope="preprocessing + FP32 Grad-CAM; excludes full-resolution reopen/render; no warmup removal",
                limitation="Acceptance is a heuristic policy decision, not optic-disc localization accuracy or model performance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    results, sources = [], []
    for path in args.reports:
        raw = path.read_bytes()
        report = json.loads(raw)
        result = summarize_report(report)
        if results and (result["checkpoint_sha256"] != results[0]["checkpoint_sha256"] or
                        result["roi_policy"] != results[0]["roi_policy"]):
            parser.error("Compare reports from the same source checkpoint and fixed ROI policy")
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest()))
        results.append(result)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status="completed", created_utc=datetime.now(timezone.utc).isoformat(), sources=sources, reports=results)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ROI diagnostic audit", "", "Acceptance is NOT localization accuracy. No model or policy was changed.", "",
             "| Split | Images | Accepted | Positive accepted/total | Negative accepted/total |",
             "|---|---:|---:|---:|---:|"]
    for result in results:
        positive = result["by_label"].get("1", dict(accepted=0, images=0))
        negative = result["by_label"].get("0", dict(accepted=0, images=0))
        lines.append(f"| {result['split']} | {result['images']} | {result['accepted']} | "
                     f"{positive['accepted']}/{positive['images']} | {negative['accepted']}/{negative['images']} |")
    lines += ["", "Failed conditions can overlap. Counts are descriptive, not a threshold tuning objective.",
              "Timing excludes full-resolution image reopen/render and is not T4 end-to-end acceptance.", ""]
    (output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"ROI audit saved: {output}")


if __name__ == "__main__":
    main()
