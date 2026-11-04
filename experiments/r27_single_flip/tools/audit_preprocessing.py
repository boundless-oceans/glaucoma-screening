"""CPU-only fingerprints for the fixed five R27 reproduction outliers."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

IDS = ('valid_00163.jpg', 'valid_02776.jpg', 'valid_03407.jpg',
       'valid_05455.jpg', 'valid_05673.jpg')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materials-root', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    # Only plain image/array modules are imported; never predict or model modules.
    inference = args.materials_root.resolve() / 'inference'
    sys.path.insert(0, str(inference))
    import numpy as np
    import PIL
    from PIL import Image
    from glaucoma.local_region import prepare_shared
    from glaucoma.r24_protocol import locator_input, locator_array
    records = []
    paths = {}
    root = args.data_root.resolve() / 'valid'
    for p in root.rglob('*'):
        if p.is_file() and p.name.casefold() in IDS:
            key = p.name.casefold()
            if key in paths or not p.resolve().is_relative_to(root):
                raise ValueError('Ambiguous/unsafe image path')
            paths[key] = p
    for name in IDS:
        path = paths[name]
        with Image.open(path) as image:
            raw_size = list(image.size)
            orientation = image.getexif().get(274, 1)
        full, geometry, decoded = prepare_shared(path)
        locator, letterbox = locator_input(decoded)
        array = locator_array(locator)
        records.append(dict(image_id=name, jpeg_sha256=sha(path), raw_size=raw_size,
                            exif_orientation=orientation, decoded_size=list(decoded.size),
                            decoded_rgb_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
                            global_geometry=geometry,
                            global_rgb_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
                            locator_geometry=letterbox,
                            locator_rgb_sha256=hashlib.sha256(locator.tobytes()).hexdigest(),
                            locator_array_dtype=str(array.dtype), locator_array_shape=list(array.shape),
                            locator_array_sha256=hashlib.sha256(array.astype('<f4').tobytes()).hexdigest()))
    if 'torch' in sys.modules:
        raise RuntimeError('Unexpected model dependency imported')
    modules = ('data.py', 'local_region.py', 'r23_geometry.py', 'r24_protocol.py')
    report = dict(environment=dict(python=sys.version, pillow=PIL.__version__, numpy=np.__version__,
                                   platform=platform.platform(), byteorder=sys.byteorder),
                  source_sha256={name: sha(inference / 'glaucoma' / name) for name in modules},
                  records=records, torch_imported=False)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('CPU fingerprints saved:', args.output)


if __name__ == '__main__':
    main()
