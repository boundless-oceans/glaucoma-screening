"""Detached full JPEG decode for one bounded background worker; no models."""
from contextlib import contextmanager
from PIL import Image, ImageOps
from detail_crop import source_box


def decode_original(path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened) if opened.getexif().get(274, 1) != 1 else opened
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # copy forces full decoding and detaches pixels from the closed file.
        return image.copy()


@contextmanager
def background_decode(executor, path):
    future = executor.submit(decode_original, path)
    try:
        yield future
    finally:
        # Await even on a prediction exception; never leave a decode in flight.
        if not future.cancel():
            future.result().close()


def crop_decoded(image, roi, size=512, mirrored=False):
    if type(size) is not int or size <= 0 or type(mirrored) is not bool:
        raise ValueError('Positive integer size and explicit boolean view required')
    box = source_box(roi, image.size)
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError('Mapped source ROI collapsed')
    sampling_box = (image.width-box[2], box[1], image.width-box[0], box[3]) if mirrored else box
    crop = image.crop(sampling_box)
    if mirrored:
        crop = ImageOps.mirror(crop)
    return ImageOps.pad(crop, (size, size), method=Image.Resampling.BICUBIC, color=(0, 0, 0)), box
