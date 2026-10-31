"""CPU-only reproducibility and timing A/B for classic ROI versions. TRAIN only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image

from . import CONFIG_DIR, DEFAULT_DATA_ROOT
from .classic_roi import classic_options, locate_classic_roi
from .data_utils import data_signature, json_save, read_rows
from .preprocessing import prepare_image
from .run_utils import protect_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "classic_roi.json")
    parser.add_argument("--timing-count", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.timing_count <= 64 or not 1 <= args.repeats <= 5:
        parser.error("timing-count must be 1..64; repeats 1..5")
    new_policy = classic_options(json.loads(args.config.read_bytes()))
    if new_policy["classic_roi_version"] != 2:
        parser.error("Candidate policy must be version 2")
    signature = data_signature(args.data_root)
    lookup = {r["Image_name"]: r for r in read_rows(args.data_root, "train")}
    sources, records, inputs, seen = [], [], [], set()
    for path in args.reports:
        raw = path.read_bytes(); report = json.loads(raw)
        if report.get("status") != "completed" or report.get("split") != "train" or report["data_signature"] != signature:
            parser.error("Use completed TRAIN reports from the same dataset")
        old_policy = classic_options(report["policy"])
        if old_policy["classic_roi_version"] != 1:
            parser.error("Reference reports must use version 1")
        if old_policy["probe_size"] != new_policy["probe_size"]:
            parser.error("Probe sizes must agree for this policy comparison")
        sources.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest()))
        for row in report["images"]:
            name = row["image_name"]
            if name in seen: continue
            seen.add(name)
            with Image.open(lookup[name]["path"]) as source:
                image, info = prepare_image(source, old_policy["probe_size"], report["preprocessing_config"])
            slow, slow_score = locate_classic_roi(image, info, old_policy, backend="pillow")
            fast, fast_score = locate_classic_roi(image, info, old_policy)
            if slow != fast or not np.array_equal(slow_score, fast_score):
                raise ValueError(f"Fast filter changed v1 results: {name}")
            for key in ("fallback", "reason", "roi_box", "model_box"):
                if slow[key] != row["selection"][key]:
                    raise ValueError(f"Saved v1 geometry not reproduced: {name} {key}")
            candidate, _ = locate_classic_roi(image, info, new_policy)
            records.append(dict(image_name=name, label=row["label"], v1=fast, v2=candidate,
                                decision_changed=(fast["fallback"], fast["reason"], fast["roi_box"]) !=
                                                 (candidate["fallback"], candidate["reason"], candidate["roi_box"])))
            if len(inputs) < args.timing_count: inputs.append((image, info, old_policy))
            print(f"Equivalence {len(records)}: {name}", flush=True)
    # Separate decoded-input filter timing from file decode and drawing.
    timings = dict(pillow=[], fast=[])
    for repeat in range(args.repeats):
        for i, (image, info, policy) in enumerate(inputs):
            for backend in (("pillow", "fast") if (repeat + i) % 2 == 0 else ("fast", "pillow")):
                started = time.perf_counter()
                locate_classic_roi(image, info, policy, backend=backend)
                timings[backend].append((time.perf_counter() - started) * 1000)
    timing_summary = {backend:dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                                   count=len(values)) for backend, values in timings.items()}
    result = dict(status="completed", created_utc=datetime.now(timezone.utc).isoformat(), sources=sources,
                  source_sha256=hashlib.sha256(Path(__file__).with_name("classic_roi.py").read_bytes()).hexdigest(),
                  v2_policy=new_policy, v1_equivalent_images=len(records), images=records,
                  timing=timing_summary, filter_ratio=timing_summary["pillow"]["mean_ms"] / timing_summary["fast"]["mean_ms"],
                  timing_scope="Decoded input -> ROI selection, alternating backends; no decode/render/model; not T4 acceptance",
                  changed_images=[r["image_name"] for r in records if r["decision_changed"]])
    output = protect_output(args.data_root, args.output_dir, directory=True)
    output.mkdir(parents=True, exist_ok=False)
    json_save(output / "report.json", result)
    print(json.dumps({k:result[k] for k in ("v1_equivalent_images", "timing", "filter_ratio", "changed_images")}))


if __name__ == "__main__":
    main()
