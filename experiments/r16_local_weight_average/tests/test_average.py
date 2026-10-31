"""CPU-only numerical, identity and preparation checks; no torch import."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
from r16_average import core
spec=importlib.util.spec_from_file_location('average_prepare',HERE/'scripts/prepare.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
sys.path.insert(0,str(HERE.parent/'r16_roi_consistency/tests'))
from synthetic_data import write_official


class AverageChecks(unittest.TestCase):
    def average(self,dst,a,b,names):
        core.average_parameters_(dst,a,b,names,equal=np.array_equal,finite=lambda v:bool(np.isfinite(v).all()))

    def test_fixed_average_keeps_parents_and_integer_buffers(self):
        a=dict(w=np.array([2.,-4.],dtype=np.float32),count=np.array(7,dtype=np.int64))
        b=dict(w=np.array([6.,0.],dtype=np.float32),count=np.array(7,dtype=np.int64))
        dst={k:v.copy() for k,v in a.items()}
        self.average(dst,a,b,{'w'})
        np.testing.assert_array_equal(dst['w'],[4.,-2.]);self.assertEqual(dst['w'].dtype,np.float32)
        np.testing.assert_array_equal(a['w'],[2.,-4.]);np.testing.assert_array_equal(b['w'],[6.,0.])
        self.assertEqual(dst['count'],7);self.assertEqual(dst['count'].dtype,np.int64)
        self.average(dst,a,a,{'w'});np.testing.assert_array_equal(dst['w'],a['w'])

    def test_reject_before_modification(self):
        a=dict(w=np.array([2.],dtype=np.float32),buffer=np.array(1.,dtype=np.float32))
        for b in (dict(a,buffer=np.array(2.,dtype=np.float32)),
                  dict(a,w=np.array([np.nan],dtype=np.float32)),
                  dict(a,w=np.array([2.],dtype=np.float16)),
                  dict(a,w=np.array([2.,3.],dtype=np.float32)),
                  {'w':a['w']}):
            dst={k:v.copy() for k,v in a.items()}
            with self.assertRaises(ValueError):self.average(dst,a,b,{'w'})
            np.testing.assert_array_equal(dst['w'],[2.])
        with self.assertRaises(ValueError):self.average(a,a,a,set())

    def test_parent_configuration_alignment(self):
        c=dict(model='convnext_tiny',image_size=384,use_age=False,seed=20261026,
               initializer_sha256=core.INITIALIZER,local_region=dict(teacher_sha256=core.WEIGHTS['global']))
        core.validate_parent_configs(c,dict(c))
        for key,value in [('model','efficientnet_b2'),('seed',42),('image_size',320),('use_age',True),
                          ('initializer_sha256','untrusted'),('local_region',{})]:
            with self.assertRaises(ValueError):core.validate_parent_configs(c,dict(c,**{key:value}))

    def test_fixed_sample_includes_boundary_cases_and_both_classes(self):
        names=['valid_%05d.jpg'%i for i in range(5849)];labels=[int(i>=5658) for i in range(5849)]
        indices=core.check_indices(names,labels)
        self.assertEqual(indices,core.check_indices(names,labels))
        self.assertEqual(len(indices),len(set(indices)));self.assertGreaterEqual(len(indices),32);self.assertLessEqual(len(indices),35)
        for name in ('valid_03118.jpg','valid_00296.jpg','valid_01329.jpg'):self.assertIn(names.index(name),indices)
        self.assertGreaterEqual(sum(labels[i] for i in indices),16)

    def test_reference_drift_and_threshold_gating(self):
        self.assertTrue(core.compare_reference([.1,.9],[.1,.9])['passed'])
        self.assertFalse(core.compare_reference([.1,.902],[.1,.9])['passed'])
        self.assertFalse(core.compare_reference([.5001],[.4999])['passed'])
        for values in ([],[float('nan')],[1.1]):
            with self.assertRaises(ValueError):core.compare_reference(values,values)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'ref.csv';p.write_text('image_id,probability\nb.jpg,0.1\na.jpg,0.2\n')
            with self.assertRaises(ValueError):core.reference_values(p,['a.jpg','b.jpg'],'probability')

    def test_completed_stage_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'PREPARATION.json').write_text('{}')
            out=root/'build';out.mkdir();(out/'model.pt').write_bytes(b'fixture')
            core.complete(out,root,['model.pt']);core.verify_complete(out,root,['model.pt'])
            with self.assertRaises(ValueError):core.verify_complete(out,root,['missing.json'])
            (out/'model.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):core.verify_complete(out,root,['model.pt'])

    def test_full_file_preparation_with_synthetic_rows_and_fake_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';write_official(data,'valid')
            names,labels=core.validation_rows(data)
            weights={}
            for role in core.WEIGHTS:
                path=root/(role+'.pt');path.write_bytes(('non-model-fixture-'+role).encode());weights[role]=path
            refs={}
            for role,(_,column) in core.REFERENCES.items():
                path=root/(role+'.csv')
                with path.open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['image_id',column]);w.writerows((name,.9 if y else .1) for name,y in zip(names,labels))
                refs[role]=path
            target=root/'prepared'
            argv=['prepare','--output',str(target),'--data-root',str(data)]
            for role,path in weights.items():argv+=['--'+role+'-checkpoint',str(path)]
            for role,path in refs.items():argv+=['--reference-'+role,str(path)]
            digests={role:core.sha(path) for role,path in weights.items()}
            rd={role:(core.sha(path),core.REFERENCES[role][1]) for role,path in refs.items()}
            with patch.dict(core.WEIGHTS,digests),patch.dict(core.REFERENCES,rd),patch.object(core,'VALID_SHA',core.sha(data/'valid.csv')),patch.object(sys,'argv',argv):
                prepare.main();meta,_,_=core.validate_project(target)
                self.assertFalse(meta['official_rows_copied'])
                self.assertFalse(list(target.rglob('*.pt')));self.assertFalse(list(target.rglob('*.csv')))
                for name,digest in meta['sources'].items():self.assertEqual(core.sha(target/name),digest)
                for path in target.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
                with self.assertRaises(FileExistsError):prepare.main()
                argv[2]=str(root/'failed')
                with patch.object(core,'validate_project',side_effect=ValueError('fixture failure')):
                    with self.assertRaisesRegex(ValueError,'fixture failure'):prepare.main()
                self.assertFalse((root/'failed').exists());self.assertEqual(list(root.glob('.average_prepare_*')),[])
                source=target/'src/r16_average/runtime.py';original=source.read_bytes()
                source.write_bytes(original+b'\n# altered\n')
                with self.assertRaisesRegex(ValueError,'source changed'):core.validate_project(target)
                source.write_bytes(original)
                weights['ema'].write_bytes(b'changed parent')
                with self.assertRaisesRegex(ValueError,'weight changed'):core.validate_project(target)

    def test_help_has_no_framework_or_stage_side_effects(self):
        with patch.object(sys,'argv',['run.py','--help']):
            with self.assertRaises(SystemExit) as done:runpy.run_path(str(HERE/'scripts/run.py'),run_name='__main__')
        self.assertEqual(done.exception.code,0)
        with patch.object(sys,'argv',['run.py','build']),patch.object(sys,'platform','win32'):
            with self.assertRaises(SystemExit) as blocked:runpy.run_path(str(HERE/'scripts/run.py'),run_name='__main__')
        self.assertEqual(blocked.exception.code,2)
        self.assertFalse((HERE/'.stage.lock').exists())
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    temp=HERE/'outputs/test_tmp';temp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temp)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AverageChecks))
    if 'torch' in sys.modules:raise RuntimeError('Local checks imported PyTorch')
    raise SystemExit(not result.wasSuccessful())
