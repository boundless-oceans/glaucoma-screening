"""CPU contracts only; never import PyTorch or deserialize model files."""
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
from r16_deploy import core

spec=importlib.util.spec_from_file_location('portable_prepare',HERE/'scripts/prepare.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


class InferenceChecks(unittest.TestCase):
    def test_percentiles_boundary_and_non_t4_never_claims_acceptance(self):
        r=core.summary([0,50,100,150],'NVIDIA GeForce RTX 4090 D',True)
        self.assertEqual(r['mean_ms'],75);self.assertEqual(r['p50_ms'],75)
        self.assertAlmostEqual(r['p95_ms'],142.5);self.assertAlmostEqual(r['p99_ms'],148.5)
        self.assertEqual(r['over_100ms'],1);self.assertFalse(r['hardware_is_t4']);self.assertFalse(r['official_acceptance'])
        r=core.summary([100],'Tesla T4',False)
        self.assertTrue(r['hardware_is_t4']);self.assertTrue(r['all_measured_under_or_equal_100ms'])
        self.assertFalse(r['full_coverage']);self.assertFalse(r['official_acceptance'])

    def test_invalid_measurements_and_probabilities(self):
        for values in ([],[-1],[float('nan')],[float('inf')]):
            with self.assertRaises(ValueError):core.summary(values,'T4',True)
        for p in (-1,1.1,float('nan')):
            with self.assertRaises(ValueError):core.validate_probability(p)

    def test_timing_covers_predict_and_trailing_synchronization(self):
        events=[];ticks=iter([1.,1.012])
        def sync():events.append('sync')
        def clock():events.append('clock');return next(ticks)
        def predict(path):events.append(('predict',path));return dict(probability=.3)
        value,ms=core.timed_call(predict,'image',sync,clock)
        self.assertEqual(events,['sync','clock',('predict','image'),'sync','clock'])
        self.assertAlmostEqual(ms,12);self.assertEqual(value['probability'],.3)

    def test_inputs_preserve_order_case_and_refuse_missing_or_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);images=root/'images';images.mkdir();metadata=root/'metadata.csv'
            (images/'a.JPG').write_bytes(b'fixture');(images/'b.jpg').write_bytes(b'fixture')
            metadata.write_text('Image_name,True_Label\nb.jpg,1\na.jpg,0\n')
            rows=core.inputs(images,metadata,2)
            self.assertEqual([n for n,_ in rows],['b.jpg','a.jpg']);self.assertEqual(rows[1][1].name,'a.JPG')
            (images/'c.PNG').write_bytes(b'fixture');metadata.write_text('Image_name\nc.png\n')
            self.assertEqual(core.inputs(images,metadata,1)[0][1].name,'c.PNG')
            metadata.write_text('Image_name\na.jpg\nA.JPG\n')
            with self.assertRaises(ValueError):core.inputs(images,metadata)
            metadata.write_text('Image_name\n../a.jpg\n')
            with self.assertRaises(ValueError):core.inputs(images,metadata)
            metadata.write_text('Image_name\nmissing.jpg\n')
            with self.assertRaises(ValueError):core.inputs(images,metadata)

    def test_sampling_is_fixed_and_full_run_keeps_original_order(self):
        rows=list(range(1000))
        self.assertEqual(core.select(rows,20),core.select(rows,20));self.assertEqual(len(set(core.select(rows,20))),20)
        self.assertEqual(core.select(rows,0),rows)
        with self.assertRaises(ValueError):core.select(rows,-1)

    def test_prepare_is_portable_and_refuses_tampered_sources_or_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);weights={n:root/(n+'.pt') for n in core.WEIGHTS}
            for n,p in weights.items():p.write_bytes(('not-a-checkpoint-'+n).encode())
            expected={n:core.sha(p) for n,p in weights.items()};bundle=root/'bundle'
            argv=['prepare','--output',str(bundle),'--global-checkpoint',str(weights['global']),'--local-checkpoint',str(weights['local'])]
            with patch.object(sys,'argv',argv),patch.dict(core.WEIGHTS,expected):
                prepare.main()
                with self.assertRaises(FileExistsError):prepare.main()
                m=core.validate_bundle(bundle)
                self.assertFalse(m['data_rows_included']);self.assertFalse(list(bundle.rglob('*.csv')))
                self.assertTrue((bundle/'src/r16_tta/pipeline.py').is_file())
                source=bundle/'src/r16_tta/pipeline.py';old=source.read_bytes();source.write_bytes(old+b'\n# changed\n')
                with self.assertRaisesRegex(ValueError,'source changed'):core.validate_bundle(bundle)
                source.write_bytes(old);(bundle/'model/local.pt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'weight changed'):core.validate_bundle(bundle)

    def test_help_does_not_import_framework(self):
        for args in (['--help'],['benchmark','--help'],['predict','--help']):
            with patch.object(sys,'argv',['run.py',*args]):
                with self.assertRaises(SystemExit) as e:runpy.run_path(str(HERE/'scripts/run.py'),run_name='__main__')
                self.assertEqual(e.exception.code,0)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    temporary=HERE/'outputs/test_tmp';temporary.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temporary)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InferenceChecks))
    if 'torch' in sys.modules:raise RuntimeError('CPU checks imported PyTorch')
    raise SystemExit(not result.wasSuccessful())
