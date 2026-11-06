"""Coordinate cache and verified local image decoding. Pillow/NumPy only."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil

from PIL import Image, ImageOps

from .data_utils import data_signature


def validate_record(record):
    dimensions = record["original_size"]
    if len(dimensions) != 2 or any(type(v) is not int or v <= 0 for v in dimensions):
        raise ValueError("Invalid cached dimensions")
    box = record["box"]
    if len(box) != 4 or any(type(v) is not int for v in box) or not (
            0 <= box[0] < box[2] <= dimensions[0] and 0 <= box[1] < box[3] <= dimensions[1]):
        raise ValueError("Cached crop lies outside original image")
    if type(record["fallback"]) is not bool or type(record["input_bytes"]) is not int or record["input_bytes"] <= 0:
        raise ValueError("Invalid cache image metadata")
    digest = record["input_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid cached image SHA256")
    name = record["image_name"]
    if not name or name != name.strip() or "/" in name or "\\" in name:
        raise ValueError("Invalid cached image name")
    return record


def load_roi_cache(root, directory, expected_signature=None):
    if directory is None:
        raise ValueError("Local input requires --roi-cache; build a coordinate cache first")
    directory = Path(directory)
    if not (directory / "manifest.json").is_file():
        raise ValueError("ROI cache has no completion manifest; finish building the cache first")
    raw = (directory / "manifest.json").read_bytes()
    signature = hashlib.sha256(raw).hexdigest()
    if expected_signature is not None and signature != expected_signature:
        raise ValueError("ROI cache differs from checkpoint; restore the original cache")
    manifest = json.loads(raw)
    if manifest.get("status") != "completed" or manifest["header"]["format_version"] != 1:
        raise ValueError("ROI cache is incomplete or unsupported")
    if manifest["header"]["data_signature"] != data_signature(root):
        raise ValueError("ROI cache CSV signatures differ from dataset")
    splits = {}
    for split in ("train", "valid", "test"):
        metadata = manifest["files"][split]
        raw_rows = (directory / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(raw_rows).hexdigest() != metadata["sha256"]:
            raise ValueError(f"ROI cache content hash mismatch: {split}")
        rows = [validate_record(json.loads(line)) for line in raw_rows.splitlines()]
        lookup = {r["image_name"].casefold(): r for r in rows}
        if len(lookup) != len(rows) or len(rows) != metadata["count"]:
            raise ValueError(f"ROI cache duplicate/count mismatch: {split}")
        names = [r["image_name"] for r in rows]
        if hashlib.sha256(json.dumps(names).encode()).hexdigest() != manifest["header"]["selection_sha256"][split]:
            raise ValueError(f"ROI cache row selection mismatch: {split}")
        splits[split] = lookup
    return dict(signature=signature, manifest=manifest, splits=splits)


def cache_rows(cache, split, rows):
    records = cache["splits"][split]
    if any(r["Image_name"].casefold() not in records for r in rows):
        raise ValueError(f"ROI cache does not cover requested {split} rows (sample/full mismatch)")
    return records


def snapshot_roi_cache(source, destination, signature):
    """Keep small coordinate files with the run; publish completion manifest last."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        return
    names = ["train.jsonl", "valid.jsonl", "test.jsonl", "manifest.json"]
    destination.mkdir(parents=True, exist_ok=True)
    allowed = set(names) | {name + ".tmp" for name in names}
    if set(p.name for p in destination.iterdir()) - allowed:
        raise ValueError("Unexpected files in run ROI cache snapshot")
    if (destination / "manifest.json").exists():
        if hashlib.sha256((destination/"manifest.json").read_bytes()).hexdigest() != signature:
            raise ValueError("Run cache snapshot belongs to different coordinates")
        metadata = json.loads((destination/"manifest.json").read_bytes())
        for split in ("train", "valid", "test"):
            if hashlib.sha256((destination/f"{split}.jsonl").read_bytes()).hexdigest() != metadata["files"][split]["sha256"]:
                raise ValueError("Run cache snapshot is corrupted; restore it before resuming")
        return
    manifest = json.loads((source/"manifest.json").read_bytes())
    if hashlib.sha256((source/"manifest.json").read_bytes()).hexdigest() != signature:
        raise ValueError("Source cache changed before snapshot")
    for name in names:
        temporary = destination / (name + ".tmp")
        shutil.copyfile(source/name, temporary)
        expected = signature if name == "manifest.json" else manifest["files"][name.removesuffix(".jsonl")]["sha256"]
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
            raise ValueError("Cache changed while copying snapshot")
        os.replace(temporary, destination/name)


def prepare_cached_image(raw, record, size, draft_size=0):
    """Map full EXIF-oriented coordinates into a fresh JPEG decoder image."""
    validate_record(record)
    if len(raw) != record["input_bytes"] or hashlib.sha256(raw).hexdigest() != record["input_sha256"]:
        raise ValueError("Image content differs from ROI cache; rebuild rather than reusing stale coordinates")
    if type(size) is not int or size <= 0 or type(draft_size) is not int or draft_size < 0 or (draft_size and draft_size < size):
        raise ValueError("Invalid local decoder/image size")
    with Image.open(io.BytesIO(raw)) as source:
        original_size = source.size
        if source.getexif().get(274, 1) in (5, 6, 7, 8): original_size = original_size[::-1]
        if list(original_size) != record["original_size"]:
            raise ValueError("Original geometry differs from cache")
        if draft_size and source.format in ("JPEG", "MPO"):
            source.draft("RGB", (draft_size, draft_size))
        decoded = ImageOps.exif_transpose(source).convert("RGB")
    l, t, r, b = record["box"]
    sx, sy = decoded.width / original_size[0], decoded.height / original_size[1]
    mapped = [max(0, math.floor(l * sx)), max(0, math.floor(t * sy)),
              min(decoded.width, math.ceil(r * sx)), min(decoded.height, math.ceil(b * sy))]
    return ImageOps.pad(decoded.crop(mapped), (size, size), method=Image.Resampling.BILINEAR, color="black")
