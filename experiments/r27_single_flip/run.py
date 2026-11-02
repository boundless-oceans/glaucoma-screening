"""Explicit prepare -> check -> evaluate -> optional export; never trains."""
import argparse
import importlib.util
import json
import platform
import sys
import time
import traceback
from pathlib import Path

from core import (HERE, audit_reference, check_indices, digest, probability,
                  read_csv, resolve_images, validate_reference, verify_materials,
                  write_json, write_submission)


def source_identity():
    files = sorted(HERE.glob('*.py')) + [HERE / 'materials_sha256.json']
    metrics = HERE.parents[1] / 'src/glaucoma_training/data_utils.py'
    return {str(p.relative_to(HERE)) if p.is_relative_to(HERE) else 'metrics': digest(p)
            for p in files + [metrics]}


def require_complete(project, stage):
    path = project / stage
    complete = json.loads((path / 'complete.json').read_text(encoding='utf-8'))
    for name, sha in complete['files'].items():
        if digest(path / name) != sha:
            raise ValueError('Prior stage output changed: ' + stage + '/' + name)
    return json.loads((path / 'report.json').read_text(encoding='utf-8'))


def load_metrics():
    path = HERE.parents[1] / 'src/glaucoma_training/data_utils.py'
    spec = importlib.util.spec_from_file_location('r27_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.metrics


def make_engine(materials):
    # No torch import or weight deserialization before a Linux CUDA stage.
    import torch
    import torchvision
    import numpy
    import PIL
    if sys.version_info[:2] != (3, 12) or torch.__version__ != '2.10.0+cu128' or torchvision.__version__ != '0.25.0+cu128':
        raise ValueError('Use Python3.12 / torch2.10.0+cu128 / torchvision0.25.0+cu128')
    if not torch.cuda.is_available():
        raise ValueError('This paired experiment requires a CUDA server')
    sys.path.insert(0, str(materials / 'inference'))
    from predict import Inference
    env = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
               numpy=numpy.__version__, pillow=PIL.__version__, cuda=torch.version.cuda,
               gpu=torch.cuda.get_device_name(0), platform=platform.platform())
    return Inference(materials / 'inference', 'cuda'), env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'check', 'evaluate', 'export'))
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--materials-root', type=Path)
    parser.add_argument('--data-root', type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    if args.stage == 'prepare':
        if args.materials_root is None or args.data_root is None:
            parser.error('prepare requires --materials-root and --data-root')
        materials = args.materials_root.resolve()
        verify_materials(materials)
        rows = read_csv(materials / 'metadata/official_valid.csv')
        validate_reference(rows, read_csv(materials / 'predictions/valid_branches.csv'))
        resolve_images(args.data_root, rows)
        tests = read_csv(materials / 'metadata/test_manifest.csv')
        if len(tests) != 5855 or any(r['label'] for r in tests):
            raise ValueError('Official unlabeled test manifest required')
        project.mkdir(parents=True, exist_ok=False)
        write_json(project / 'project.json', dict(materials=str(materials), data_root=str(args.data_root.resolve()),
                   source=source_identity(), policy='R27-single-flip-v1', reference_tolerance=.001))
        print('Prepared paths and hashes only; no model loaded:', project)
        return 0
    if sys.platform != 'linux':
        parser.error('Model stages run only on the Linux server; local GPU use is disabled')
    if args.materials_root is not None or args.data_root is not None:
        parser.error('Paths are fixed by prepare; do not override them during model stages')
    import fcntl
    with (project / '.stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((project / 'project.json').read_text(encoding='utf-8'))
        if config['source'] != source_identity() or config['policy'] != 'R27-single-flip-v1' or config['reference_tolerance'] != .001:
            raise ValueError('Prepared source/protocol changed; prepare a new project')
        materials = Path(config['materials'])
        verify_materials(materials)
        if args.stage in ('evaluate', 'export'):
            check = require_complete(project, 'check')
            if not check['reference_audit']['passed'] or not check['adapter_equivalence']:
                raise ValueError('Small paired check has not passed')
        if args.stage == 'export':
            evaluation = require_complete(project, 'evaluate')
            if not evaluation['reference_audit']['passed'] or evaluation['delta_auroc'] <= 0:
                raise ValueError('No validated AUROC gain; do not export this candidate')
        output = project / args.stage
        output.mkdir(exist_ok=False)  # Failed/partial stages remain inspectable.
        started = time.perf_counter()
        try:
            rows = read_csv(materials / ('metadata/test_manifest.csv' if args.stage == 'export' else 'metadata/official_valid.csv'))
            refs = read_csv(materials / 'predictions/valid_branches.csv')
            if args.stage != 'export':
                validate_reference(rows, refs)
            if args.stage == 'check':
                indices = check_indices(rows)
                rows, refs = [rows[i] for i in indices], [refs[i] for i in indices]
            paths = resolve_images(config['data_root'], rows)
            engine, env = make_engine(materials)
            write_json(output / 'environment.json', env)
            from adapter import predict_view
            # Fixed warmup, excluded from diagnostics. No batching or TTA.
            for _ in range(2):
                engine.predict(paths[0])
                predict_view(engine, paths[0], True)
            original, flipped = [], []
            equivalent = True
            with (output / 'trace.jsonl').open('x', encoding='utf-8', buffering=1) as trace:
                for i, (row, path) in enumerate(zip(rows, paths)):
                    pair = {}
                    arms = ('original', 'flip') if i % 2 == 0 else ('flip', 'original')
                    if args.stage == 'export':
                        arms = ('flip',)
                    for arm in arms:
                        result = engine.predict(path) if arm == 'original' else predict_view(engine, path, True)
                        probability(result['probability'])
                        result.update(image_id=row['image_id'], view=arm)
                        pair[arm] = result
                        trace.write(json.dumps(result, allow_nan=False) + '\n')
                    if args.stage == 'check':
                        audit = predict_view(engine, path, False)
                        # Same engine/device/pixels: changing only preparation must
                        # preserve geometry, crop bytes and original probabilities.
                        for key in ('roi_box', 'field', 'decoded_size', 'original_size', 'crop_rgb_sha256', 'fallback'):
                            equivalent &= audit[key] == pair['original'][key]
                        equivalent &= abs(audit['probability'] - pair['original']['probability']) <= 1e-7
                        audit.update(image_id=row['image_id'], view='adapter_original')
                        trace.write(json.dumps(audit, allow_nan=False) + '\n')
                    if 'original' in pair:
                        original.append(pair['original'])
                    flipped.append(pair['flip'])
                    if (i + 1) % 128 == 0 or i + 1 == len(rows):
                        print(f'{args.stage}: {i + 1}/{len(rows)}', flush=True)
            report = dict(stage=args.stage, count=len(rows), source=config['source'], environment=env,
                          elapsed_seconds=time.perf_counter() - started,
                          timing_scope='Current GPU diagnostic: decode + preprocessing + GPU forward, excludes startup/warmup; not T4 acceptance')
            if args.stage == 'export':
                write_submission(output / 'submission_single_flip.csv', rows, flipped)
                report['submission_sha256'] = digest(output / 'submission_single_flip.csv')
            else:
                metric = load_metrics()
                labels = [int(r['label']) for r in rows]
                report.update(original=metric(labels, [r['probability'] for r in original]),
                              flip=metric(labels, [r['probability'] for r in flipped]),
                              reference_audit=audit_reference(original, refs),
                              adapter_equivalence=equivalent if args.stage == 'check' else None)
                report['delta_auroc'] = report['flip']['auroc'] - report['original']['auroc']
                import numpy as np
                report['latency_ms'] = {arm: dict(mean=float(np.mean([r['elapsed_ms'] for r in results])),
                                                  p95=float(np.percentile([r['elapsed_ms'] for r in results], 95)),
                                                  p99=float(np.percentile([r['elapsed_ms'] for r in results], 99)),
                                                  max=max(r['elapsed_ms'] for r in results),
                                                  over_100ms=sum(r['elapsed_ms'] > 100 for r in results))
                                        for arm, results in (('original', original), ('flip', flipped))}
                report['reference_roi_box_changes'] = sum(
                    r['roi_box'] != [int(ref[k]) for k in ('px0', 'py0', 'px1', 'py1')]
                    for r, ref in zip(original, refs))
            write_json(output / 'report.json', report)
            if args.stage != 'export' and (not equivalent or not report['reference_audit']['passed']):
                raise ValueError('Baseline reproduction failed; preserve report/trace for audit, do not relax automatically')
            # Check input identities again; completion is written last.
            verify_materials(materials)
            if source_identity() != config['source']:
                raise ValueError('Source changed during stage')
            write_json(output / 'complete.json', dict(files={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
            print('COMPLETE:', output, flush=True)
        except BaseException:
            (output / 'failed.txt').write_text(traceback.format_exc(), encoding='utf-8')
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
