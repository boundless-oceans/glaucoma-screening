"""CPU policy restoration and synthetic stage orchestration; no real torch."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy
import run_stable as run
from stable_precision import install_locator_policy, captain_audit


class StableTests(unittest.TestCase):
    def test_precision_scope_restores_even_on_exception(self):
        backend = types.SimpleNamespace(allow_tf32=True)
        @contextlib.contextmanager
        def flags(**kwargs):
            old = backend.allow_tf32
            backend.allow_tf32 = kwargs['allow_tf32']
            try:
                yield
            finally:
                backend.allow_tf32 = old
        backend.flags = flags
        torch = types.SimpleNamespace(backends=types.SimpleNamespace(cudnn=backend))
        def forward(fail=False):
            self.assertFalse(backend.allow_tf32)
            if fail:
                raise ValueError('synthetic forward failure')
            return 'ok'
        engine = types.SimpleNamespace(locator=types.SimpleNamespace(forward=forward))
        install_locator_policy(engine, torch)
        self.assertEqual(engine.locator.forward(), 'ok')
        self.assertTrue(backend.allow_tf32)
        with self.assertRaises(ValueError):
            engine.locator.forward(True)
        self.assertTrue(backend.allow_tf32)

    def test_synthetic_pipeline_and_repeat_failure(self):
        rows = [dict(image_id=f'valid_{i:05}.jpg', relative_path=f'valid/valid_{i:05}.jpg', label=str(int(i < 191))) for i in range(5849)]
        tests = [dict(image_id=f'test_{i:05}.jpg', relative_path=f'test/test_{i:05}.jpg', label='') for i in range(5855)]
        def predict(path, mirrored=False):
            i = int(path.stem.rsplit('_', 1)[1])
            p = (.9 if i < 191 else .1) if mirrored else (.05 if i == 0 else (.9 if i < 191 else .1))
            return dict(probability=p, elapsed_ms=1., roi_box=[0, 0, 1, 1], field=[0, 0, 1, 1],
                        original_size=[1, 1], decoded_size=[1, 1], crop_rgb_sha256='synthetic', fallback=False)
        refs = [dict(Image_name=r['image_id'], True_Label=r['label'], Probability='.01', Pred_Label='0', px0='0', py0='0', px1='1', py1='1') for r in rows]
        captain = dict(records=[dict(image_id=n, arm='locator_tf32_off', repeat=0, **predict(Path(n))) for n in run.IDS])
        engine = types.SimpleNamespace(predict=predict)
        adapter = types.SimpleNamespace(predict_view=lambda engine, path, mirrored: predict(path, mirrored))
        fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *a: None)
        digest = run.digest
        def csv_rows(path):
            return refs if path.name == 'valid_branches.csv' else (tests if path.name == 'test_manifest.csv' else rows)
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            refpath = Path(temp)/'captain.json'
            refpath.write_text(json.dumps(captain))
            stack.enter_context(patch.object(run, 'digest', side_effect=lambda p: run.TF32_REFERENCE_SHA if Path(p) == refpath else digest(p)))
            stack.enter_context(patch.object(run, 'verify_materials', return_value={}))
            stack.enter_context(patch.object(run, 'resolve_images', side_effect=lambda root, rr: [Path(r['image_id']) for r in rr]))
            stack.enter_context(patch.object(run, 'read_csv', side_effect=csv_rows))
            stack.enter_context(patch.object(run, 'make_engine', return_value=(engine, {'gpu': 'synthetic'})))
            stack.enter_context(patch.object(run, 'install_locator_policy'))
            stack.enter_context(patch.object(sys, 'platform', 'linux'))
            stack.enter_context(patch.dict(sys.modules, {'fcntl': fcntl, 'adapter': adapter, 'torch': types.SimpleNamespace()}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            project = Path(temp)/'project'
            for stage in ('prepare', 'check', 'evaluate', 'export'):
                argv = ['run_stable.py', stage, '--project', str(project)]
                if stage == 'prepare':
                    argv += ['--materials-root', temp, '--data-root', temp, '--tf32-reference', str(refpath)]
                with patch.object(sys, 'argv', argv):
                    self.assertEqual(run.main(), 0)
            report = run.require_complete(project, 'evaluate')
            self.assertTrue(report['baseline_gate']['passed'])
            self.assertFalse(report['reference_audit']['passed'])
            self.assertEqual(len((project/'export/submission_single_flip.csv').read_text().splitlines()), 5856)
            from stable_precision import check_replay
            current = [dict(image_id=r['image_id'], **predict(Path(r['image_id']))) for r in rows]
            flips = [dict(image_id=r['image_id'], **predict(Path(r['image_id']), True)) for r in rows]
            current[163]['probability'] += .01
            self.assertFalse(check_replay(project, current, flips)['passed'])
            bad = copy.deepcopy(captain)
            bad['records'][0]['roi_box'] = [1, 0, 2, 1]
            self.assertFalse(captain_audit(current, bad)['passed'])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
