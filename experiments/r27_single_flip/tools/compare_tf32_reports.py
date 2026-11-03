"""Compare two fixed five-image diagnostics on CPU; no weights or torch."""
import argparse
import json
import math
from pathlib import Path

IDS = {'valid_00163.jpg', 'valid_02776.jpg', 'valid_03407.jpg',
       'valid_05455.jpg', 'valid_05673.jpg'}
ARMS = ('original', 'locator_tf32_off')


def validate(report):
    if not report.get('no_training') or not report.get('no_submission'):
        raise ValueError('Expected bounded inference diagnostic')
    rows = report['records']
    expected = {(name, arm, repeat) for name in IDS for arm in ARMS for repeat in range(3)}
    actual = [(r['image_id'], r['arm'], r['repeat']) for r in rows]
    if len(actual) != 30 or set(actual) != expected:
        raise ValueError('Full unique five-image/two-arm/repeat3 diagnostic required')
    result = {}
    for name in IDS:
        for arm in ARMS:
            group = [r for r in rows if r['image_id'] == name and r['arm'] == arm]
            first = group[0]
            for row in group:
                if row['effective_locator_tf32'] != (arm == 'original'):
                    raise ValueError('Unexpected effective TF32 scope')
                p = row['probability']
                if not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError('Invalid probability')
                for key in ('roi_box', 'locator_center', 'probability', 'top_indices'):
                    if row[key] != first[key]:
                        raise ValueError('Repeated diagnostic is unstable: ' + name + '/' + arm)
            result[name, arm] = first
    return result


def compare(left, right):
    a, b = validate(left), validate(right)
    rows = []
    for name in sorted(IDS):
        for arm in ARMS:
            x, y = a[name, arm], b[name, arm]
            rows.append(dict(image_id=name, arm=arm,
                             roi_equal=x['roi_box'] == y['roi_box'],
                             peaks_equal=x['top_indices'] == y['top_indices'],
                             left_box=x['roi_box'], right_box=y['roi_box'],
                             max_center_delta=max(abs(u-v) for u, v in zip(x['locator_center'], y['locator_center'])),
                             probability_delta=y['probability']-x['probability']))
    return dict(rows=rows,
                same_box_counts={arm: sum(r['roi_equal'] for r in rows if r['arm'] == arm) for arm in ARMS},
                note='Five-image cross-environment diagnostic only; not accuracy, full validation or latency acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', required=True, type=Path)
    parser.add_argument('--right', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.left.read_text(encoding='utf-8')),
                     json.loads(args.right.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result['same_box_counts']))


if __name__ == '__main__':
    main()
