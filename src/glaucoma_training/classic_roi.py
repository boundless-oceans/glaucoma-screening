"""Experimental brightness/contrast candidates; no model, label or medical prior weights."""
import math

import numpy as np
from PIL import Image, ImageFilter

from .roi_geometry import content_box, map_box_to_original

DEFAULTS = dict(classic_roi_version=1, probe_size=512, size_fraction=.65,
                min_peak_score=.32, min_peak_ratio=1.08)


def classic_options(config=None):
    config = {} if config is None else config
    if not isinstance(config, dict) or set(config) - set(DEFAULTS):
        raise ValueError("Invalid or unknown classic ROI options")
    options = {**DEFAULTS, **config}
    if type(options["classic_roi_version"]) is not int or options["classic_roi_version"] != 1:
        raise ValueError("Unsupported classic ROI version")
    if type(options["probe_size"]) is not int or not 128 <= options["probe_size"] <= 1024:
        raise ValueError("probe_size must be an integer in 128..1024")
    for key, (low, high) in dict(size_fraction=(.4, .9), min_peak_score=(.01, 1), min_peak_ratio=(1, 3)).items():
        value = options[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid {key}")
    return options


def locate_classic_roi(processed, info, config=None):
    """Return selection and score image. All cues are heuristic, not verified anatomy.

    Green-heavy intensity and local contrast produce candidates. Thin dark-line
    support is only a small ranking cue; it is NOT a vessel convergence detector.
    A boundary exclusion and two-scale consistency guard reject some glare.
    """
    options = classic_options(config)
    size = processed.width
    if processed.height != size or size != options["probe_size"]:
        raise ValueError("Expected configured square probe image")
    box = content_box(info, size)
    result = dict(fallback=True, fallback_mode="global_only", reason=None,
                  roi_box=None, model_box=None, content_box=box, policy=options)
    empty = np.zeros((size, size), dtype=np.float32)

    def reject(reason, score=empty):
        return {**result, "reason": reason}, score

    if info["reason"] in ("dark_image_fallback", "sparse_foreground_fallback", "small_region_fallback"):
        return reject("preprocessing_" + info["reason"])
    rgb = np.asarray(processed.convert("RGB"), dtype=np.float32)
    cl, ct, cr, cb = box
    short = min(cr - cl, cb - ct)
    foreground = rgb.max(axis=2) > info["preprocessing"]["black_border_threshold"]
    foreground[:ct] = False; foreground[cb:] = False
    foreground[:, :cl] = False; foreground[:, cr:] = False
    if foreground.mean() < .1:
        return reject("insufficient_foreground")
    # A local support mask excludes the circular rim, dark holes and letterbox.
    radius = max(2, round(short * .025))
    interior = np.asarray(Image.fromarray((foreground * 255).astype(np.uint8)).filter(
        ImageFilter.MinFilter(radius * 2 + 1))) > 0
    y, x = np.mgrid[:size, :size]
    ellipse = ((x - (cl + cr - 1) / 2) / ((cr - cl) * .46)) ** 2 + \
              ((y - (ct + cb - 1) / 2) / ((cb - ct) * .46)) ** 2 < 1
    interior &= ellipse
    if not interior.any():
        return reject("no_interior_support")
    intensity = .25 * rgb[..., 0] + .65 * rgb[..., 1] + .1 * rgb[..., 2]
    # Compact bright structures can occupy less than 5% of a fundus image.
    low, high = np.percentile(intensity[foreground], [5, 99])
    if high <= 20 or high - low < 8:
        return reject("dark_or_low_contrast")
    normalized = np.clip((intensity - low) / max(high - low, 20), 0, 1)
    gray = Image.fromarray(np.uint8(normalized * 255))
    small = np.asarray(gray.filter(ImageFilter.GaussianBlur(max(1, short * .025))), dtype=float) / 255
    large = np.asarray(gray.filter(ImageFilter.GaussianBlur(max(2, short * .075))), dtype=float) / 255
    # Bright compact structure versus its broad neighbourhood. Both scales must agree.
    contrast = np.maximum(small - large, 0)
    score = (.65 * small + .35 * np.clip(contrast / .2, 0, 1)) * interior
    # Top-hat-like dark fine structure is a weak cue, deliberately <=10% of ranking.
    green = Image.fromarray(rgb[..., 1].astype(np.uint8))
    closing = green.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))
    dark = np.maximum(np.asarray(closing, dtype=float) - rgb[..., 1], 0)
    support = np.asarray(Image.fromarray(np.uint8(np.clip(dark * 6, 0, 255))).filter(
        ImageFilter.GaussianBlur(max(1, short * .04))), dtype=float) / 255
    score *= .9 + .1 * support
    # Reject tiny saturated reflections before choosing a peak.
    saturated = (rgb.min(axis=2) > 245).astype(np.uint8) * 255
    glare = np.asarray(Image.fromarray(saturated).filter(
        ImageFilter.GaussianBlur(max(1, short * .025))), dtype=float) / 255
    score[glare > .35] = 0
    py, px = np.unravel_index(np.argmax(score), score.shape)
    peak = float(score[py, px])
    result.update(center_probe=[int(px), int(py)], peak_score=peak,
                  local_contrast=float(contrast[py, px]), thin_dark_structure_support=float(support[py, px]))
    if peak < options["min_peak_score"]:
        return reject("weak_brightness_candidate", score.astype(np.float32))
    other = score.copy()
    other[(x - px) ** 2 + (y - py) ** 2 < (short * .2) ** 2] = 0
    alternative = float(other.max())
    ratio = peak / max(alternative, 1e-8)
    result.update(alternative_peak_score=alternative, peak_ratio=ratio)
    side = max(1, round(short * options["size_fraction"]))
    left = min(max(cl, int(px) - side // 2), cr - side)
    top = min(max(ct, int(py) - side // 2), cb - side)
    candidate = [left, top, left + side, top + side]
    result["candidate_model_box"] = candidate
    if ratio < options["min_peak_ratio"]:
        return reject("multiple_or_diffuse_bright_regions", score.astype(np.float32))
    # Glare/edges can peak at one scale. Require a nearby broad-scale maximum as well.
    broad = large * interior
    by, bx = np.unravel_index(np.argmax(broad), broad.shape)
    result["broad_scale_center"] = [int(bx), int(by)]
    if math.hypot(int(bx) - px, int(by) - py) > short * .16:
        return reject("scale_disagreement", score.astype(np.float32))
    return {**result, "fallback": False, "fallback_mode": None, "reason": "classic_brightness_candidate",
            "model_box": candidate, "roi_box": map_box_to_original(candidate, info, size)}, score.astype(np.float32)
