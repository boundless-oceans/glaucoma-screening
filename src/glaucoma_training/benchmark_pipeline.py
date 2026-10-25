"""CPU preprocessing A/B timing and image differences. Never imports PyTorch.

Read only train/valid. Repeated file reads generally use OS caches; this is not
a cold-disk benchmark, a GPU training speedup claim, or a medical quality test.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from . import DEFAULT_DATA_ROOT, PROJECT_ROOT
from .data_utils import json_save, read_rows, subset, data_signature
from .model_config import load_config
from .preprocess_config import preprocessing_options
from .preprocessing import prepare_image
from .run_utils import protect_output, validate_config


def compare_rows(rows, size, config, draft_size, repeats=3):
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("repeats must be within 1..10")
    if not rows:
        raise ValueError("No images selected")
    preprocessing_options({**config, "jpeg_draft_size": draft_size})
    if draft_size < size:
        raise ValueError("draft_size must be at least the final image size")
    modes = {"full": {**config, "jpeg_draft_size": 0}, "draft": {**config, "jpeg_draft_size": draft_size}}
    records, all_times = [], {key: [] for key in modes}
    for index, row in enumerate(rows):
        times = {key: [] for key in modes}
        images, infos = {}, {}
        for repeat in range(repeats):
            # Alternate which arm runs first; do not silently call either arm cold.
            order = ["full", "draft"] if (index + repeat) % 2 == 0 else ["draft", "full"]
            for mode in order:
                start = time.perf_counter()
                with Image.open(row["path"]) as source:
                    images[mode], infos[mode] = prepare_image(source, size, modes[mode])
                elapsed = (time.perf_counter() - start) * 1000
                times[mode].append(elapsed)
                all_times[mode].append(elapsed)
        full, draft = infos["full"], infos["draft"]
        difference = np.abs(np.asarray(images["full"], dtype=np.int16) - np.asarray(images["draft"], dtype=np.int16))
        w, h = full["original_size"]
        norm = np.array([w, h, w, h], dtype=float)
        box_delta = float(np.max(np.abs(np.array(full["crop_box"]) / norm - np.array(draft["crop_box"]) / norm)))
        records.append(dict(image_name=row["Image_name"], label=int(row["True_Label"]),
                            file_bytes=Path(row["path"]).stat().st_size,
                            full_mean_ms=float(np.mean(times["full"])), draft_mean_ms=float(np.mean(times["draft"])),
                            mean_abs_pixel_delta=float(difference.mean()), max_abs_pixel_delta=int(difference.max()),
                            fraction_pixels_gt8=float((difference.max(axis=2) > 8).mean()),
                            normalized_crop_box_max_delta=box_delta,
                            crop_decision_changed=full["cropped"] != draft["cropped"],
                            fallback_reason_changed=full["reason"] != draft["reason"],
                            full_reason=full["reason"], draft_reason=draft["reason"],
                            full_info=full, draft_info=draft))
        print(f"preprocessing A/B {index + 1}/{len(rows)}: {row['Image_name']}", flush=True)
    summary = dict(images=len(rows), repeats=repeats, final_size=size, jpeg_draft_size=draft_size,
                   timing_scope="open + decode + EXIF/RGB + crop + pad; alternating arms, OS caches uncontrolled",
                   timing={mode: dict(mean_ms=float(np.mean(values)), p50_ms=float(np.percentile(values, 50)),
                                      p95_ms=float(np.percentile(values, 95))) for mode, values in all_times.items()},
                   cpu_preprocessing_ratio=float(sum(all_times["full"]) / sum(all_times["draft"])),
                   crop_decision_changes=sum(r["crop_decision_changed"] for r in records),
                   reason_changes=sum(r["fallback_reason_changed"] for r in records),
                   full_reasons=dict(Counter(r["full_reason"] for r in records)),
                   draft_reasons=dict(Counter(r["draft_reason"] for r in records)),
                   draft_applied=sum(r["draft_info"]["jpeg_draft_applied"] for r in records),
                   mean_abs_pixel_delta=float(np.mean([r["mean_abs_pixel_delta"] for r in records])),
                   max_normalized_crop_box_delta=max(r["normalized_crop_box_max_delta"] for r in records))
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--draft-size", type=int, default=640)
    parser.add_argument("--include-name", action="append", default=[], help="Force inclusion of known dark/no-border/odd-size cases")
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.count < 2 or not 0 <= args.preview_count <= 32:
        parser.error("count must be >=2; preview-count must be 0..32")
    config = load_config(args.config)
    validate_config(config)
    all_rows = read_rows(args.data_root, args.split)
    rows = subset(all_rows, args.count, config["seed"])
    by_name = {r["Image_name"].casefold(): r for r in all_rows}
    selected = {r["Image_name"].casefold() for r in rows}
    for name in args.include_name:
        key = name.casefold()
        if key not in by_name:
            parser.error(f"Unknown image in {args.split}: {name}")
        if key not in selected:
            rows.append(by_name[key]); selected.add(key)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = protect_output(args.data_root, args.output_dir or PROJECT_ROOT / "outputs" / ("pipeline_" + stamp), directory=True)
    output.mkdir(parents=True, exist_ok=False)
    records, summary = compare_rows(rows, config["image_size"], config, args.draft_size, args.repeats)
    summary.update(split=args.split, config=config, python=platform.python_version(), pillow=pillow_version,
                   numpy=np.__version__, platform=platform.platform(), data_signature=data_signature(args.data_root),
                   selection_sha256=hashlib.sha256(json.dumps([r["Image_name"] for r in rows]).encode()).hexdigest())
    json_save(output / "report.json", dict(summary=summary, images=records))
    fields = [key for key in records[0] if key not in ("full_info", "draft_info")]
    with (output / "images.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    count = min(args.preview_count, len(rows))
    if count:
        size = config["image_size"]
        canvas = Image.new("RGB", (size * 2, (size + 24) * count), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(rows[:count]):
            y = i * (size + 24)
            draw.text((2, y), row["Image_name"] + " full / draft", fill="black")
            for col, draft_size in enumerate((0, args.draft_size)):
                with Image.open(row["path"]) as source:
                    image, _ = prepare_image(source, size, {**config, "jpeg_draft_size": draft_size})
                canvas.paste(image, (col * size, y + 24))
        canvas.save(output / "preview.jpg", quality=95)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output}; no weights/data modified. CPU ratio is NOT epoch speedup.")


if __name__ == "__main__":
    main()
