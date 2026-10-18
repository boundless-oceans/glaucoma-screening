"""Conservative outer-background cropping with Pillow/NumPy only, no model execution."""
import math

import numpy as np
from PIL import Image, ImageOps

from .preprocess_config import preprocessing_options


def crop_black_border(image, config):
    """Return cropped RGB image and auditable original-coordinate crop metadata.

    Version 1 uses a <=512px detection image, max(R,G,B)>threshold foreground,
    and its *entire* bounding rectangle (no connected-component pruning).
    A 2%-of-original-dimension margin protects dark retinal edges by default.
    Dim/sparse/implausibly small detections fall back to the full image.
    This removes outer rectangular background, not every black circular corner.
    """
    options = preprocessing_options(config)
    width, height = image.size
    full_box = (0, 0, width, height)
    info = dict(original_size=[width, height], crop_box=list(full_box), cropped=False,
                reason="disabled", preprocessing=options)
    if not options["remove_black_border"]:
        return image, info
    probe = image.copy()
    probe.thumbnail((512, 512), Image.Resampling.BOX)
    pixels = np.asarray(probe.convert("RGB"), dtype=np.uint8)
    intensity = pixels.max(axis=2)
    if float(np.percentile(intensity, 95)) <= max(32, options["black_border_threshold"]):
        info["reason"] = "dark_image_fallback"
        return image, info
    foreground = intensity > options["black_border_threshold"]
    info["foreground_fraction"] = float(foreground.mean())
    if not foreground.any() or info["foreground_fraction"] < 0.1:
        info["reason"] = "sparse_foreground_fallback"
        return image, info
    ys, xs = np.nonzero(foreground)
    probe_width, probe_height = probe.size
    margin = options["black_border_margin"]
    left = max(0, math.floor(int(xs.min()) * width / probe_width - margin * width))
    top = max(0, math.floor(int(ys.min()) * height / probe_height - margin * height))
    right = min(width, math.ceil((int(xs.max()) + 1) * width / probe_width + margin * width))
    bottom = min(height, math.ceil((int(ys.max()) + 1) * height / probe_height + margin * height))
    retained_width, retained_height = (right - left) / width, (bottom - top) / height
    retained_area = retained_width * retained_height
    info["retained_area_fraction"] = retained_area
    if retained_width < 0.35 or retained_height < 0.35 or retained_area < 0.2:
        info["reason"] = "small_region_fallback"
        return image, info
    box = (left, top, right, bottom)
    if box == full_box:
        info["reason"] = "no_outer_border"
        return image, info
    info.update(cropped=True, crop_box=list(box), reason="cropped")
    return image.crop(box), info


def prepare_image(image, size, config=None):
    """Shared deterministic EXIF/RGB/crop/letterbox path for all dataset splits."""
    if type(size) is not int or size <= 0:
        raise ValueError("Output image size must be a positive integer")
    image = ImageOps.exif_transpose(image).convert("RGB")
    image, info = crop_black_border(image, config or {})
    image = ImageOps.pad(image, (size, size), method=Image.Resampling.BILINEAR, color=(0, 0, 0))
    return image, info
