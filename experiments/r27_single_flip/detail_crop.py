"""CPU prototype: retain the fixed ROI while reading full-resolution JPEG pixels.

Not connected to training/inference; it cannot change the current candidate.
"""
import math
from pathlib import Path
from PIL import Image, ImageOps


def source_box(roi, original_size):
    if len(roi['decoded_size']) != 2 or len(original_size) != 2:
        raise ValueError('Two source dimensions required')
    dw, dh = roi['decoded_size']
    dimensions = (dw, dh, *original_size)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v <= 0 or int(v) != v for v in dimensions):
        raise ValueError('Finite positive integer dimensions required')
    box = roi['roi_box']
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('Finite ROI required')
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('Nonempty ROI required')
    if tuple(roi['original_size']) != tuple(original_size):
        raise ValueError('ROI original size does not match oriented JPEG')
    ow, oh = original_size
    return tuple(round(v * s) for v, s in zip(box, (ow/dw, oh/dh, ow/dw, oh/dh)))


def prepare_detail(path, roi, size=512, mirrored=False):
    if Path(path).suffix.casefold() not in ('.jpg', '.jpeg') or type(size) is not int or size <= 0:
        raise ValueError('Original JPEG and positive output size required')
    if type(mirrored) is not bool:
        raise ValueError('Explicit boolean view required')
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened) if opened.getexif().get(274, 1) != 1 else opened
        if image.mode != 'RGB':
            image = image.convert('RGB')
        box = source_box(roi, image.size)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError('Mapped source ROI collapsed')
        # Do not shift a boundary ROI; retain the same outside-image black pad.
        # ROI coordinates describe the selected view. Reflect its source box
        # first, then mirror only the crop, avoiding a full-frame mirror copy.
        sampling_box = (image.width-box[2], box[1], image.width-box[0], box[3]) if mirrored else box
        crop = image.crop(sampling_box)
        if mirrored:
            crop = ImageOps.mirror(crop)
    return ImageOps.pad(crop, (size, size), method=Image.Resampling.BICUBIC, color=(0, 0, 0)), box
