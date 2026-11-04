"""Five-image diagnostic only: no training, selection or submission export."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

IDS = ('valid_00163.jpg', 'valid_02776.jpg', 'valid_03407.jpg',
       'valid_05455.jpg', 'valid_05673.jpg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--materials-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run model diagnostics on the Linux GPU server only')
    sys.path.insert(0, str(args.code_root / 'experiments/r27_single_flip'))
    from core import verify_materials
    verify_materials(args.materials_root)
    import numpy as np
    import torch
    import torchvision
    if not torch.cuda.is_available():
        raise ValueError('CUDA server required')
    sys.path.insert(0, str(args.materials_root / 'inference'))
    from predict import Inference
    # Fail on existing output; this audit never changes prior stage markers.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    engine = Inference(args.materials_root / 'inference', 'cuda')
    paths = {}
    for p in (args.data_root / 'valid').iterdir():
        if p.is_file() and p.name.casefold() in IDS:
            if p.name.casefold() in paths:
                raise ValueError('Ambiguous image ID')
            paths[p.name.casefold()] = p
    if set(paths) != set(IDS):
        raise ValueError('All five original JPEGs required')
    original_forward = engine.locator.forward
    arm = 'original'
    captured = {}
    heatmaps = {}

    def forward(*inputs, **kwargs):
        # Nested scope overrides only the locator. Both classifier forwards stay
        # under the unchanged captain Inference.predict scopes.
        before = torch.backends.cudnn.allow_tf32
        allow = before if arm == 'original' else False
        with torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                       deterministic=True, allow_tf32=allow):
            captured['effective_locator_tf32'] = torch.backends.cudnn.allow_tf32
            output = original_forward(*inputs, **kwargs)
            captured['heat'] = output.detach().float().cpu().numpy().copy()
        if torch.backends.cudnn.allow_tf32 != before:
            raise RuntimeError('Locator precision scope leaked')
        return output

    initial_tf32 = torch.backends.cudnn.allow_tf32
    engine.locator.forward = forward
    records = []
    try:
        # Warm up both arms; these calls are excluded from recorded diagnostics.
        for arm in ('original', 'locator_tf32_off'):
            engine.predict(paths[IDS[0]])
        for image_id in IDS:
            for repeat in range(3):
                order = ('original', 'locator_tf32_off') if repeat % 2 == 0 else ('locator_tf32_off', 'original')
                for arm in order:
                    captured.clear()
                    result = engine.predict(paths[image_id])
                    if torch.backends.cudnn.allow_tf32 != initial_tf32:
                        raise RuntimeError('Classifier/default precision changed')
                    heat = captured.pop('heat')
                    key = f'{image_id[:-4]}_{arm}_{repeat}'
                    heatmaps[key] = heat
                    flat = heat.reshape(-1).astype(np.float64)
                    probabilities = np.exp(flat - flat.max())
                    probabilities /= probabilities.sum()
                    top = np.argsort(-probabilities, kind='stable')[:2]
                    records.append(dict(image_id=image_id, arm=arm, repeat=repeat,
                                        effective_locator_tf32=captured['effective_locator_tf32'],
                                        heatmap_key=key,
                                        top_indices=top.tolist(),
                                        top_probabilities=probabilities[top].tolist(),
                                        **result))
                print(image_id, repeat + 1, '/3', flush=True)
    finally:
        engine.locator.forward = original_forward
    np.savez_compressed(args.output_dir / 'heatmaps.npz', **heatmaps)
    report = dict(records=records, torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                  python=sys.version, gpu=torch.cuda.get_device_name(0),
                  cudnn=torch.backends.cudnn.version(), initial_cudnn_tf32=initial_tf32,
                  classifier_settings_unchanged=True, no_training=True, no_submission=True,
                  note='Diagnostic timings include heatmap CPU transfer; not inference acceptance or speed comparison.')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    verify_materials(args.materials_root)
    (args.output_dir / 'complete.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output_dir.iterdir() if p.is_file()}), encoding='utf-8')
    print('DIAGNOSTIC_COMPLETE', args.output_dir, flush=True)


if __name__ == '__main__':
    main()
