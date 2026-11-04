"""Locator-only precision policy and auditable same-policy comparisons."""
import json

IDS = {'valid_00163.jpg', 'valid_02776.jpg', 'valid_03407.jpg',
       'valid_05455.jpg', 'valid_05673.jpg'}
KEYS = ('roi_box', 'field', 'decoded_size', 'original_size', 'crop_rgb_sha256', 'fallback')


def install_locator_policy(engine, torch):
    forward = engine.locator.forward
    def scoped(*args, **kwargs):
        before = torch.backends.cudnn.allow_tf32
        try:
            with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
                if torch.backends.cudnn.allow_tf32:
                    raise RuntimeError('Locator TF32 policy not applied')
                return forward(*args, **kwargs)
        finally:
            if torch.backends.cudnn.allow_tf32 != before:
                raise RuntimeError('Locator precision scope leaked')
    engine.locator.forward = scoped


def equivalent(a, b):
    return all(a[k] == b[k] for k in KEYS) and abs(a['probability']-b['probability']) <= 1e-7


def captain_audit(results, report):
    refs = {r['image_id']: r for r in report['records'] if r['arm'] == 'locator_tf32_off' and r['repeat'] == 0}
    if set(refs) != IDS:
        raise ValueError('Fixed five-image captain reference required')
    current = {r['image_id']: r for r in results if r['image_id'] in IDS}
    if set(current) != IDS:
        raise ValueError('Missing a cross-GPU diagnostic image')
    delta = max(abs(current[k]['probability']-refs[k]['probability']) for k in IDS)
    boxes = all(current[k]['roi_box'] == refs[k]['roi_box'] for k in IDS)
    return dict(count=5, roi_equal=boxes, max_probability_delta=delta,
                tolerance=.001, passed=boxes and delta <= .001,
                scope='Five fixed images only; not all-image cross-GPU acceptance')


def check_replay(project, originals, flips):
    saved = [json.loads(s) for s in (project/'check/trace.jsonl').read_text(encoding='utf-8').splitlines()]
    count = 0
    for arm, rows in (('original', originals), ('flip', flips)):
        lookup = {r['image_id']: r for r in rows}
        for old in saved:
            if old['view'] == arm:
                count += 1
                if old['image_id'] not in lookup or not equivalent(old, lookup[old['image_id']]):
                    return dict(passed=False, count=count, failed_image=old['image_id'], view=arm)
    if count < 64:
        raise ValueError('Full small-check replay required')
    return dict(passed=True, count=count, scope='Small-check images replayed in full valid on the same GPU')
