"""File identities and experiment protocol; no model imports."""
import csv
import hashlib
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify_materials(root):
    root = Path(root).resolve()
    pins = json.loads((HERE / 'materials_sha256.json').read_text(encoding='utf-8'))
    for name, expected in pins.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing/unsafe material: ' + name)
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('Material identity mismatch: ' + name)
    return pins


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Finite continuous probability required')
    return value


def validate_reference(rows, reference):
    if len(rows) != 5849 or sum(int(r['label']) for r in rows) != 191:
        raise ValueError('Official full valid required')
    if len(reference) != len(rows):
        raise ValueError('Full reference required')
    for row, ref in zip(rows, reference):
        if row['image_id'] != ref['Image_name'] or row['label'] != ref['True_Label']:
            raise ValueError('Reference ID/order/label mismatch')
        p = probability(ref['Probability'])
        if int(ref['Pred_Label']) != int(p >= .5):
            raise ValueError('Reference threshold mismatch')


def check_indices(rows):
    rng = random.Random(20261102)
    selected = []
    for label in ('0', '1'):
        selected.extend(rng.sample([i for i, r in enumerate(rows) if r['label'] == label], 16))
    return sorted(selected)


def resolve_images(data_root, rows):
    """Index each split once, tolerate case differences, reject ambiguity/escape."""
    root = Path(data_root).resolve()
    splits = {Path(r['relative_path']).parts[0] for r in rows}
    if not splits <= {'valid', 'test'}:
        raise ValueError('Only official valid/test images are allowed')
    index = {}
    for split in sorted(splits):
        for path in (root / split).rglob('*'):
            if path.is_file() and path.suffix.casefold() in ('.jpg', '.jpeg'):
                real = path.resolve()
                if not real.is_relative_to(root):
                    raise ValueError('Image escaped data root')
                key = path.relative_to(root).as_posix().casefold()
                if key in index:
                    raise ValueError('Ambiguous image path: ' + key)
                index[key] = real
    result = []
    seen = set()
    for row in rows:
        name = row['image_id']
        relative = Path(row['relative_path'])
        if relative.is_absolute() or '..' in relative.parts or relative.name.casefold() != name.casefold():
            raise ValueError('Unsafe/mismatched metadata path')
        if name.casefold() in seen:
            raise ValueError('Duplicate image ID')
        seen.add(name.casefold())
        result.append(index[relative.as_posix().casefold()])
    return result


def audit_reference(results, refs, tolerance=.001):
    if len(results) != len(refs) or not results:
        raise ValueError('Reference audit requires paired rows')
    deltas = [abs(probability(r['probability']) - probability(ref['Probability'])) for r, ref in zip(results, refs)]
    changed = sum(int(probability(r['probability']) >= .5) != int(ref['Pred_Label']) for r, ref in zip(results, refs))
    return dict(max_probability_delta=max(deltas), label_changes=changed,
                tolerance=tolerance, passed=max(deltas) <= tolerance and changed == 0)


def write_submission(path, rows, results):
    if len(rows) != 5855 or len(results) != 5855 or len({r['image_id'].casefold() for r in rows}) != 5855:
        raise ValueError('Full unique test required')
    values = []
    for row, result in zip(rows, results):
        if row['image_id'] != result['image_id']:
            raise ValueError('Submission order mismatch')
        p = probability(result['probability'])
        values.append((row['image_id'], format(p, '.17g'), int(p >= .5)))
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('Image_name', 'Probability', 'Pred_Label'))
        writer.writerows(values)
