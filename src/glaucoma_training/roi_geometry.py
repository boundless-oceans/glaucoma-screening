"""Label-independent candidate ROI geometry. Pillow/NumPy only, no PyTorch."""
import math
import random

import numpy as np
from PIL import Image, ImageOps, ImageDraw

DEFAULTS = dict(roi_version=1, size_fraction=0.6, min_attention_mass=0.55,
                min_concentration_gain=1.2, min_foreground_fraction=0.5)


def roi_options(config=None):
    config = {} if config is None else config
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("Unknown ROI options or invalid config")
    options = {**DEFAULTS, **config}
    if type(options["roi_version"]) is not int or options["roi_version"] != 1:
        raise ValueError("Unsupported roi_version")
    limits = dict(size_fraction=(0.2, 0.9), min_attention_mass=(0.1, 1),
                  min_concentration_gain=(1, 10), min_foreground_fraction=(0.1, 1))
    for key, (low, high) in limits.items():
        value = options[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid ROI option: {key}")
    return options


def content_box(info, size):
    """Exact Pillow letterbox footprint, including its version-specific rounding.

    A one-bit solid mask uses the same contain/pad geometry as the RGB pipeline.
    This runs only in ROI diagnostics, never in the existing training pipeline.
    """
    if type(size) is not int or size <= 0:
        raise ValueError("Invalid input size")
    left, top, right, bottom = info["decoded_crop_box"]
    if not 0 <= left < right <= info["decoded_size"][0] or not 0 <= top < bottom <= info["decoded_size"][1]:
        raise ValueError("Invalid decoded crop geometry")
    mask = Image.new("1", (right - left, bottom - top), 1)
    box = ImageOps.pad(mask, (size, size), method=Image.Resampling.BILINEAR, color=0).getbbox()
    if box is None:
        raise ValueError("Empty letterbox footprint")
    return list(box)


def map_box_to_original(box, info, size):
    """Model input -> unpadded decoded crop -> EXIF-oriented full original."""
    cl, ct, cr, cb = content_box(info, size)
    left, top, right, bottom = box
    if not cl <= left < right <= cr or not ct <= top < bottom <= cb:
        raise ValueError("ROI must lie inside unpadded input content")
    dl, dt, dr, db = info["decoded_crop_box"]
    dw, dh = info["decoded_size"]
    ow, oh = info["original_size"]
    sx, sy = (dr - dl) / (cr - cl), (db - dt) / (cb - ct)
    mapped = [math.floor((dl + (left - cl) * sx) * ow / dw),
              math.floor((dt + (top - ct) * sy) * oh / dh),
              math.ceil((dl + (right - cl) * sx) * ow / dw),
              math.ceil((dt + (bottom - ct) * sy) * oh / dh)]
    return [max(0, mapped[0]), max(0, mapped[1]), min(ow, mapped[2]), min(oh, mapped[3])]


def select_roi(heatmap, processed_image, info, config=None):
    """Fixed positive-class attention policy. Never receives a label or probability.

    Fallback means global-only, not a claimed optic-disc localization.
    Quality ratios are heuristic attention diagnostics, not calibrated confidence.
    """
    options = roi_options(config)
    pixels = np.asarray(processed_image.convert("RGB"))
    size = processed_image.width
    if processed_image.height != size:
        raise ValueError("Expected square model input")
    heat = np.asarray(heatmap, dtype=np.float64)
    if heat.shape != (size, size):
        raise ValueError("Heatmap shape differs from model input")
    content = content_box(info, size)
    result = dict(fallback=True, fallback_mode="global_only", reason=None,
                  roi_box=None, model_box=None, content_box=content, policy=options)

    def fallback(reason):
        return {**result, "reason": reason}

    if not np.isfinite(heat).all():
        return fallback("nonfinite_attention")
    if info["reason"] in ("dark_image_fallback", "sparse_foreground_fallback", "small_region_fallback"):
        return fallback("preprocessing_" + info["reason"])
    cl, ct, cr, cb = content
    mask = np.zeros((size, size), dtype=bool)
    threshold = info["preprocessing"]["black_border_threshold"]
    mask[ct:cb, cl:cr] = pixels[ct:cb, cl:cr].max(axis=2) > threshold
    heat = np.maximum(heat, 0) * mask
    total = float(heat.sum())
    if total <= 1e-12:
        return fallback("empty_foreground_attention")
    # Attention window is square in model-content space and never enters padding.
    side = max(1, round(min(cr - cl, cb - ct) * options["size_fraction"]))
    region = heat[ct:cb, cl:cr]
    integral = np.pad(region, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[side:, side:] - integral[:-side, side:] - integral[side:, :-side] + integral[:-side, :-side]
    # Several windows may contain the same isolated peak. Break mass ties by
    # proximity to the attention centroid, rather than always choosing top-left.
    ys, xs = np.nonzero(np.isclose(sums, sums.max(), rtol=1e-9, atol=total * 1e-12))
    cy = float((region.sum(axis=1) * (np.arange(region.shape[0]) + 0.5)).sum() / total)
    cx = float((region.sum(axis=0) * (np.arange(region.shape[1]) + 0.5)).sum() / total)
    nearest = int(np.argmin((ys + side / 2 - cy) ** 2 + (xs + side / 2 - cx) ** 2))
    y, x = int(ys[nearest]), int(xs[nearest])
    left, top = cl + int(x), ct + int(y)
    box = [left, top, left + side, top + side]
    selected_mask = mask[top:top + side, left:left + side]
    foreground = int(selected_mask.sum())
    fraction = foreground / side ** 2
    mass = float(sums[y, x] / total)
    expected_mass = foreground / max(int(mask.sum()), 1)
    gain = mass / max(expected_mass, 1e-12)
    result.update(candidate_model_box=box, attention_mass=mass, concentration_gain=gain,
                  foreground_fraction=fraction)
    if fraction < options["min_foreground_fraction"]:
        return fallback("insufficient_foreground")
    if mass < options["min_attention_mass"] or gain < options["min_concentration_gain"]:
        return fallback("diffuse_or_ambiguous_attention")
    original_box = map_box_to_original(box, info, size)
    return {**result, "fallback": False, "fallback_mode": None, "reason": "candidate_attention_roi",
            "model_box": box, "roi_box": original_box}


def preview_rows(rows, count, seed, balanced=True, include_names=()):
    if type(count) is not int or not 2 <= count <= 64:
        raise ValueError("Preview count must be within 2..64")
    rng = random.Random(seed)
    if balanced:
        groups = [[r for r in rows if r.get("True_Label") == str(label)] for label in (1, 0)]
        if not all(groups):
            raise ValueError("Balanced preview needs both labeled classes")
        chosen = rng.sample(groups[0], min(len(groups[0]), count // 2))
        chosen += rng.sample(groups[1], min(len(groups[1]), count - len(chosen)))
    else:
        chosen = rng.sample(rows, min(len(rows), count))
    lookup = {r["Image_name"].casefold(): r for r in rows}
    selected = {r["Image_name"].casefold() for r in chosen}
    for name in include_names:
        key = name.casefold()
        if key not in lookup:
            raise ValueError(f"Unknown image name: {name}")
        if key not in selected:
            chosen.append(lookup[key]); selected.add(key)
    if len(chosen) > 96:
        raise ValueError("Too many included images")
    return chosen


def preview_strip(original, processed, heatmap, selection, title, tile=224, heat_title="Positive-logit Grad-CAM"):
    """Original/ROI, diagnostic heatmap and local view. No image files modified."""
    canvas = Image.new("RGB", (tile * 3, tile + 42), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), title, fill="black")
    thumbnail = ImageOps.contain(original.convert("RGB"), (tile, tile), Image.Resampling.BILINEAR)
    offset = ((tile - thumbnail.width) // 2, (tile - thumbnail.height) // 2)
    first = Image.new("RGB", (tile, tile), "black")
    first.paste(thumbnail, offset)
    if selection["roi_box"] is not None:
        l, t, r, b = selection["roi_box"]
        sx, sy = thumbnail.width / original.width, thumbnail.height / original.height
        ImageDraw.Draw(first).rectangle((offset[0] + l * sx, offset[1] + t * sy,
                                         offset[0] + r * sx - 1, offset[1] + b * sy - 1), outline=(0, 255, 80), width=2)
    heat = np.maximum(np.nan_to_num(np.asarray(heatmap, dtype=float), nan=0, posinf=0, neginf=0), 0)
    heat = heat / max(float(heat.max()), 1e-12)
    rgb = np.asarray(processed.convert("RGB"), dtype=float)
    colors = np.stack([255 * heat, 128 * (1 - np.abs(2 * heat - 1)), 255 * (1 - heat)], axis=-1)
    alpha = heat[..., None] * 0.5
    overlay = Image.fromarray(np.clip(rgb * (1 - alpha) + colors * alpha, 0, 255).astype(np.uint8))
    candidate = selection.get("candidate_model_box")
    if candidate is not None:
        l, t, r, b = candidate
        color = (255, 170, 0) if selection["fallback"] else (0, 255, 80)
        ImageDraw.Draw(overlay).rectangle((l, t, r - 1, b - 1), outline=color, width=2)
    if selection["fallback"]:
        local = processed
        local_title = "Fallback: global only"
    else:
        local = original.crop(selection["roi_box"])
        local_title = "Candidate local view"
    for index, (image, label) in enumerate([(first, "Original / green=accepted ROI"),
                                           (overlay, heat_title), (local, local_title)]):
        draw.text((index * tile + 4, 20), label, fill="black")
        canvas.paste(ImageOps.pad(image, (tile, tile), method=Image.Resampling.BILINEAR, color="black"),
                     (index * tile, 42))
    return canvas
