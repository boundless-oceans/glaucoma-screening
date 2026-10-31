"""CPU-only resumable coordinate generation. Frozen rules; no labels enter localization."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, __version__ as pillow_version

from . import CONFIG_DIR, DEFAULT_DATA_ROOT
from .classic_roi import classic_options, locate_classic_roi
from .data_utils import data_signature, json_save, read_rows, subset
from .model_config import load_config
from .preprocess_config import preprocessing_options
from .preprocessing import prepare_image
from .roi_cache import load_roi_cache, validate_record
from .run_utils import RunLock, protect_output, validate_config


def build_record(row, policy, preprocessing):
    try:
        raw = Path(row["path"]).read_bytes()
        with Image.open(io.BytesIO(raw)) as source:
            image, info = prepare_image(source, policy["probe_size"], preprocessing)
        selection, _ = locate_classic_roi(image, info, policy)
        return validate_record(dict(image_name=row["Image_name"], input_sha256=hashlib.sha256(raw).hexdigest(),
            input_bytes=len(raw), original_size=info["original_size"], fallback=selection["fallback"],
            reason=selection["reason"], box=info["crop_box"] if selection["fallback"] else selection["roi_box"]))
    except Exception as error:
        raise RuntimeError(f"Cannot build ROI for {row['path']}: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "local_b0.json", help="Sample selection limits/seed")
    parser.add_argument("--roi-config", type=Path, default=CONFIG_DIR / "classic_roi.json")
    parser.add_argument("--locator-config", type=Path, default=CONFIG_DIR / "pipeline_draft.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error("workers must be 1..8")
    config = load_config(args.config); validate_config(config)
    policy = classic_options(json.loads(args.roi_config.read_bytes()))
    if policy["classic_roi_version"] != 2: parser.error("Freeze version 2 for this experiment")
    preprocessing = preprocessing_options(load_config(args.locator_config))
    if preprocessing["jpeg_draft_size"] and preprocessing["jpeg_draft_size"] < policy["probe_size"]:
        parser.error("Locator decoder is smaller than probe")
    selected = {split:subset(read_rows(args.data_root, split), config[f"{split}_limit"], config["seed"])
                for split in ("train", "valid", "test")}
    header = dict(format_version=1, data_signature=data_signature(args.data_root), policy=policy,
                  locator_runtime=dict(pillow=pillow_version, numpy=np.__version__),
                  locator_preprocessing=preprocessing,
                  source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                 for name in ("classic_roi.py", "preprocessing.py", "roi_geometry.py")},
                  selection_sha256={s:hashlib.sha256(json.dumps([r["Image_name"] for r in rows]).encode()).hexdigest()
                                    for s,rows in selected.items()})
    output = protect_output(args.data_root, args.output_dir, directory=True)
    output.mkdir(parents=True, exist_ok=True)
    lock = RunLock(output)
    try:
        header_path = output / "header.json"
        if header_path.exists():
            if json.loads(header_path.read_bytes()) != header:
                raise ValueError("Cache inputs/rules/code changed; use a new output directory")
        else:
            if set(p.name for p in output.iterdir()) - {".run.lock"}:
                raise ValueError("Refusing unrelated output directory")
            json_save(header_path, header)
        if (output / "manifest.json").exists():
            load_roi_cache(args.data_root, output)
            print(f"Completed coordinate cache verified: {output}"); return
        files = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for split, rows in selected.items():
                final, partial = output / f"{split}.jsonl", output / f"{split}.jsonl.part"
                existing = []
                path = final if final.exists() else partial
                if path.exists():
                    raw = path.read_bytes()
                    if not final.exists() and raw and not raw.endswith(b"\n"):
                        raw = raw[:raw.rfind(b"\n") + 1]; partial.write_bytes(raw)
                    existing = [validate_record(json.loads(line)) for line in raw.splitlines()]
                    if [r["image_name"] for r in existing] != [r["Image_name"] for r in rows[:len(existing)]] or len(existing) > len(rows):
                        raise ValueError("Partial coordinate cache is not a prefix of selected rows")
                if final.exists() and len(existing) != len(rows): raise ValueError("Incomplete finalized cache split")
                reasons = Counter(r["reason"] for r in existing)
                if not final.exists():
                    with partial.open("ab") as stream:
                        # Bound queued work so Ctrl+C does not wait for the full dataset.
                        for start in range(len(existing), len(rows), 128):
                            jobs = executor.map(lambda row:build_record(row, policy, preprocessing), rows[start:start + 128])
                            for index, record in enumerate(jobs, start=start + 1):
                                stream.write((json.dumps(record, separators=(",", ":")) + "\n").encode()); stream.flush()
                                reasons[record["reason"]] += 1
                                if index % 200 == 0 or index == len(rows):
                                    os.fsync(stream.fileno()); print(f"{split}: {index}/{len(rows)}", flush=True)
                    os.replace(partial, final)
                files[split] = dict(count=len(rows), sha256=hashlib.sha256(final.read_bytes()).hexdigest(), reasons=dict(reasons))
        json_save(output / "manifest.json", dict(status="completed", header=header, files=files))
        load_roi_cache(args.data_root, output)
        print(f"Coordinate cache complete: {output}. No image copies, labels, model updates or predictions.")
    finally:
        lock.close()


if __name__ == "__main__":
    main()
