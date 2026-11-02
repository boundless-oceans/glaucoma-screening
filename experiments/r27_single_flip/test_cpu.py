"""Synthetic CPU checks; optional pinned captain source audit. No torch imports."""
import argparse
import ast
import copy
import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from core import (HERE, audit_reference, check_indices, digest, probability,
                  resolve_images, verify_materials, write_submission)

MATERIALS = None


class ProtocolTests(unittest.TestCase):
    def test_probabilities_and_reference_gate(self):
        for bad in (math.nan, math.inf, -.01, 1.01):
            with self.assertRaises(ValueError):
                probability(bad)
        ref = [{'Probability': '.49', 'Pred_Label': '0'}]
        self.assertTrue(audit_reference([{'probability': .4905}], ref)['passed'])
        self.assertFalse(audit_reference([{'probability': .492}], ref)['passed'])
        self.assertFalse(audit_reference([{'probability': .5001}], [{'Probability': '.4999', 'Pred_Label': '0'}])['passed'])

    def test_sampling_is_fixed_and_balanced(self):
        rows = [dict(label=str(i % 2)) for i in range(100)]
        ids = check_indices(rows)
        self.assertEqual(ids, check_indices(rows))
        self.assertEqual(len(set(ids)), 32)
        self.assertEqual(sum(int(rows[i]['label']) for i in ids), 16)

    def test_paths_reject_traversal_and_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'valid').mkdir()
            (root / 'valid/A.JPG').write_bytes(b'synthetic')
            row = dict(image_id='a.jpg', relative_path='valid/a.jpg')
            self.assertEqual(resolve_images(root, [row]), [(root / 'valid/A.JPG').resolve()])
            for rows in ([row, row], [dict(image_id='a.jpg', relative_path='valid/../a.jpg')]):
                with self.assertRaises(ValueError):
                    resolve_images(root, rows)

    def test_material_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'a').write_bytes(b'one')
            pins = {'a': dict(bytes=3, sha256=digest(root / 'a'))}
            (root / 'materials_sha256.json').write_text(json.dumps(pins))
            with patch('core.HERE', root):
                verify_materials(root)
                (root / 'a').write_bytes(b'two')
                with self.assertRaises(ValueError):
                    verify_materials(root)

    def test_submission_order_precision_and_no_overwrite(self):
        rows = [dict(image_id=f'synthetic_{i}.jpg') for i in range(5855)]
        results = [dict(image_id=r['image_id'], probability=.12345678912345678) for r in rows]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'submission.csv'
            write_submission(path, rows, results)
            self.assertIn(b'.12345678912345678', path.read_bytes())
            self.assertNotIn(b'\r', path.read_bytes())
            with self.assertRaises(FileExistsError):
                write_submission(path, rows, results)
            results[0]['image_id'] = 'wrong.jpg'
            with self.assertRaises(ValueError):
                write_submission(Path(temp) / 'bad.csv', rows, results)

    def test_completion_detects_changed_artifact(self):
        from run import require_complete
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'check').mkdir()
            path = root / 'check/report.json'
            path.write_text('{"passed": true}')
            (root / 'check/complete.json').write_text(json.dumps(dict(files={'report.json': digest(path)})))
            self.assertTrue(require_complete(root, 'check')['passed'])
            path.write_text('{"passed": false}')
            with self.assertRaises(ValueError):
                require_complete(root, 'check')

    def test_cli_help_does_not_load_models(self):
        result = subprocess.run([sys.executable, str(HERE / 'run.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('prepare', result.stdout)
        self.assertNotIn('torch', sys.modules)

    def test_stage_pipeline_with_synthetic_predictions(self):
        # Exercise orchestration and export gates, not a real-model integration.
        import run
        import numpy  # Initialize on the real OS before mocking Linux entry guard.
        rows = [dict(image_id=f'{i}.jpg', relative_path=f'valid/{i}.jpg', label=str(int(i < 191)))
                for i in range(5849)]
        tests = [dict(image_id=f'{i}.jpg', relative_path=f'test/{i}.jpg', label='') for i in range(5855)]

        def predict(path, mirrored=False):
            i = int(path.stem)
            p = (.9 if i < 191 else .1) if mirrored else (.05 if i == 0 else (.9 if i < 191 else .1))
            return dict(probability=p, elapsed_ms=1., roi_box=[0, 0, 1, 1], field=[0, 0, 1, 1],
                        original_size=[1, 1], decoded_size=[1, 1], crop_rgb_sha256='synthetic', fallback=False)

        refs = [dict(Image_name=r['image_id'], True_Label=r['label'], Probability=str(predict(Path(r['image_id']))['probability']),
                     Pred_Label=str(int(predict(Path(r['image_id']))['probability'] >= .5)), px0='0', py0='0', px1='1', py1='1') for r in rows]
        adapter = types.SimpleNamespace(predict_view=lambda engine, path, mirrored: predict(path, mirrored))
        fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None)
        engine = types.SimpleNamespace(predict=predict)
        def csv_rows(path):
            return refs if path.name == 'valid_branches.csv' else (tests if path.name == 'test_manifest.csv' else rows)
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run, 'verify_materials', return_value={}))
            stack.enter_context(patch.object(run, 'resolve_images', side_effect=lambda root, rr: [Path(r['image_id']) for r in rr]))
            stack.enter_context(patch.object(run, 'read_csv', side_effect=csv_rows))
            stack.enter_context(patch.object(run, 'make_engine', return_value=(engine, {'gpu': 'synthetic'})))
            stack.enter_context(patch.object(sys, 'platform', 'linux'))
            stack.enter_context(patch.dict(sys.modules, {'fcntl': fcntl, 'adapter': adapter}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            project = Path(temp) / 'project'
            for stage in ('prepare', 'check', 'evaluate', 'export'):
                argv = ['run.py', stage, '--project', str(project)]
                if stage == 'prepare':
                    argv += ['--materials-root', temp, '--data-root', temp]
                with patch.object(sys, 'argv', argv):
                    self.assertEqual(run.main(), 0)
            self.assertGreater(run.require_complete(project, 'evaluate')['delta_auroc'], 0)
            self.assertEqual(len((project / 'export/submission_single_flip.csv').read_text().splitlines()), 5856)
            # A changed earlier result must prevent another downstream stage.
            (project / 'check/report.json').write_text('{}')
            with patch.object(sys, 'argv', ['run.py', 'export', '--project', str(project)]):
                with self.assertRaises(ValueError):
                    run.main()
        self.assertNotIn('torch', sys.modules)


@unittest.skipUnless(MATERIALS is not None, 'Use --materials-root for pinned source and imaging audit')
class SourceTests(unittest.TestCase):
    def test_adapter_has_only_the_declared_changes(self):
        verify_materials(MATERIALS)
        original = ast.parse((MATERIALS / 'inference/predict.py').read_text())
        method = next(n for n in next(n for n in original.body if isinstance(n, ast.ClassDef)).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'predict')
        expected = copy.deepcopy(method)
        expected.name = 'predict_view'
        expected.args.args.append(ast.arg(arg='mirrored'))
        expected.args.defaults.append(ast.Constant(value=True))
        body = []
        for node in expected.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'glaucoma.local_region':
                node.names = [n for n in node.names if n.name != 'prepare_shared']
                body.extend([node, ast.ImportFrom(module='imaging', names=[ast.alias(name='prepare_view')], level=0)])
            else:
                body.append(node)
        expected.body = body
        for node in ast.walk(expected):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'prepare_shared':
                node.func.id = 'prepare_view'
                node.keywords.append(ast.keyword(arg='mirrored', value=ast.Name(id='mirrored', ctx=ast.Load())))
        actual = next(n for n in ast.parse((HERE / 'adapter.py').read_text()).body if isinstance(n, ast.FunctionDef))
        self.assertEqual(ast.dump(expected), ast.dump(actual))

    def test_exif_decode_and_original_preparation_equivalence(self):
        import numpy as np
        from PIL import Image, ImageOps
        sys.path.insert(0, str(MATERIALS / 'inference'))
        from glaucoma.local_region import prepare_shared
        from glaucoma.r27_geometry import uniform_roi, crop_uniform
        from imaging import prepare_view
        with tempfile.TemporaryDirectory() as temp:
            rng = np.random.default_rng(42)
            pixels = rng.integers(12, 255, (101, 203, 3), dtype=np.uint8)
            pixels[:, :17] = 0
            for orientation in (1, 2, 5, 6, 8):
                path = Path(temp) / f'exif{orientation}.jpg'
                exif = Image.Exif()
                exif[274] = orientation
                Image.fromarray(pixels).save(path, quality=95, exif=exif)
                full, geometry, decoded = prepare_shared(path)
                other, og, od = prepare_view(path, False)
                self.assertEqual(geometry, og)
                self.assertEqual(full.tobytes(), other.tobytes())
                self.assertEqual(decoded.tobytes(), od.tobytes())
                _, _, flipped = prepare_view(path, True)
                self.assertEqual(flipped.tobytes(), ImageOps.mirror(decoded).tobytes())
            decoded = Image.new('RGB', (200, 100), (255, 255, 255))
            roi = uniform_roi((2000, 1000), decoded.size, (0, 0, 200, 100), (.01, .5))
            self.assertLess(roi['px0'], 0)
            self.assertFalse(roi['fallback'])
            self.assertGreater(roi['padding_fraction'], 0)
            self.assertEqual(crop_uniform(decoded, roi).getpixel((0, 192)), (0, 0, 0))
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--materials-root', type=Path)
    args, remaining = parser.parse_known_args()
    MATERIALS = args.materials_root.resolve() if args.materials_root else None
    # The decorator is evaluated before CLI parsing; enable explicitly here.
    SourceTests.__unittest_skip__ = MATERIALS is None
    unittest.main(argv=[sys.argv[0]] + remaining)
