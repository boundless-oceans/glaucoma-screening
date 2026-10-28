"""CPU-only classic ROI prototype on fixed TRAIN samples; no model or training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

from . import CONFIG_DIR, DEFAULT_DATA_ROOT, PROJECT_ROOT
from .classic_roi import classic_options, locate_classic_roi
from .data_utils import data_signature, json_save, read_rows
from .preprocessing import prepare_image
from .roi_geometry import preview_rows, preview_strip
from .run_utils import protect_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--reference-report", type=Path, required=True, help="Completed TRAIN Grad-CAM report")
    parser.add_argument("--roi-config", type=Path, default=CONFIG_DIR / "classic_roi.json")
    parser.add_argument("--sampling", choices=["reference", "unseen_train"], default="reference")
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 2 <= args.count <= 64 or args.seed < 0:
        parser.error("count must be 2..64; seed must be nonnegative")
    raw = args.reference_report.read_bytes()
    reference = json.loads(raw)
    if reference.get("status") != "completed" or reference.get("split") != "train":
        parser.error("Use a completed TRAIN report, never valid/test for prototype selection")
    if data_signature(args.data_root) != reference["data_signature"]:
        parser.error("Dataset CSV differs from reference report")
    policy = classic_options(json.loads(args.roi_config.read_text(encoding="utf-8-sig")))
    config = dict(reference["model_config"])
    if config.get("jpeg_draft_size", 0) and config["jpeg_draft_size"] < policy["probe_size"]:
        parser.error("Reference decoder is smaller than the diagnostic probe size")
    all_rows = read_rows(args.data_root, "train")
    lookup = {r["Image_name"].casefold(): r for r in all_rows}
    reference_names = [r["image_name"].casefold() for r in reference["images"]]
    if len(reference_names) != len(set(reference_names)) or not set(reference_names) <= set(lookup):
        parser.error("Invalid reference sample names")
    if args.sampling == "reference":
        rows = [lookup[name] for name in reference_names]
    else:
        excluded = set(reference_names)
        rows = preview_rows([r for r in all_rows if r["Image_name"].casefold() not in excluded],
                            args.count, args.seed, balanced=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = protect_output(args.data_root, args.output_dir or PROJECT_ROOT / "outputs" / ("classic_roi_" + stamp), directory=True)
    output.mkdir(parents=True, exist_ok=False)
    (output / "crops").mkdir()
    report = dict(status="running", split="train", method="classic_brightness_contrast_v1",
                  sampling=args.sampling, seed=args.seed if args.sampling != "reference" else reference["seed"],
                  policy=policy, preprocessing_config=config,
                  reference_report=str(args.reference_report.resolve()), reference_sha256=hashlib.sha256(raw).hexdigest(),
                  reference_checkpoint_sha256=reference["checkpoint_sha256"], data_signature=reference["data_signature"],
                  selection_sha256=hashlib.sha256(json.dumps([r["Image_name"] for r in rows]).encode()).hexdigest(),
                  limitation="Heuristic candidates, not verified optic-disc annotations. No model, labels or age used for localization.",
                  start_utc=datetime.now(timezone.utc).isoformat(), images=[], pages=[])
    json_save(output / "report.json", report)
    try:
        for i, row in enumerate(rows):
            started = time.perf_counter()
            with Image.open(row["path"]) as source:
                processed, info = prepare_image(source, policy["probe_size"], config)
            selection, score = locate_classic_roi(processed, info, policy)
            seconds = time.perf_counter() - started
            with Image.open(row["path"]) as source:
                original = ImageOps.exif_transpose(source).convert("RGB")
            if list(original.size) != info["original_size"]:
                raise ValueError("Full image geometry differs from probe metadata")
            name = row["Image_name"]
            # Labels are recorded only after localization, for descriptive preview counts.
            record = dict(image_name=name, label=int(row["True_Label"]), selection=selection,
                          preprocessing=info, decode_and_locate_seconds=seconds)
            if not selection["fallback"]:
                crop = "crops/" + name
                ImageOps.pad(original.crop(selection["roi_box"]), (384, 384), method=Image.Resampling.BILINEAR,
                             color="black").save(output / crop, quality=95)
                record["inspection_crop_file"] = crop
            report["images"].append(record)
            if i % 4 == 0:
                page = Image.new("RGB", (672, 266 * min(4, len(rows) - i)), "white")
            title = f"{name} label={row['True_Label']} {selection['reason']}"
            strip = preview_strip(original, processed, score, selection, title,
                                  heat_title="Brightness/contrast heuristic")
            page.paste(strip, (0, (i % 4) * 266))
            page_name = f"preview_{i // 4 + 1:02d}.jpg"
            page.save(output / page_name, quality=95)
            if page_name not in report["pages"]: report["pages"].append(page_name)
            json_save(output / "report.json", report)
            print(f"{i+1}/{len(rows)} {name}: {selection['reason']}; box={selection['roi_box']}", flush=True)
        report.update(status="completed", end_utc=datetime.now(timezone.utc).isoformat(),
                      candidates=sum(not r["selection"]["fallback"] for r in report["images"]),
                      reasons=dict(Counter(r["selection"]["reason"] for r in report["images"])))
        report["by_label"] = {str(label):dict(images=sum(r["label"] == label for r in report["images"]),
                               accepted=sum(r["label"] == label and not r["selection"]["fallback"] for r in report["images"]))
                              for label in (0, 1)}
        json_save(output / "report.json", report)
    except Exception as error:
        json_save(output / "report.json", {**report, "status": "failed", "error": str(error)})
        raise
    print(f"Classic ROI prototype report: {output}; no model/training/test predictions.")


if __name__ == "__main__":
    main()
