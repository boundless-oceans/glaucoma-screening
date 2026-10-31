"""CPU-only arithmetic and generated-project contracts; never import torch."""
import ast
import hashlib
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
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
ema=load('ema_cpu_helper',HERE/'src/local_ema.py')
prepare=load('ema_prepare',HERE/'scripts/prepare.py')
sys.path.insert(0,str(prepare.BASE/'tests'))
from synthetic_data import write_official


class EMAChecks(unittest.TestCase):
    def test_closed_form_parameters_and_buffers(self):
        target={'w':np.array([2.,-4.],dtype=np.float32),'counter':np.array(0,dtype=np.int64),
                'running':np.array(9.,dtype=np.float32)}
        source={'w':np.array([6.,0.],dtype=np.float32),'counter':np.array(7,dtype=np.int64),
                'running':np.array(3.,dtype=np.float32)}
        unchanged={k:v.copy() for k,v in source.items()}
        ema.average_state_(target,source,{'w'},.75)
        np.testing.assert_array_equal(target['w'],[3.,-3.])
        self.assertEqual(target['counter'],7);self.assertEqual(target['running'],3.)
        # A second successful update has closed-form coefficients d²,d(1-d),1-d.
        source['w'][:]=[10.,8.]
        ema.average_state_(target,source,{'w'},.75)
        np.testing.assert_array_equal(target['w'],[4.75,-.25])
        np.testing.assert_array_equal(unchanged['w'],[6.,0.])
        np.testing.assert_array_equal(source['w'],[10.,8.])
        self.assertEqual(target['w'].dtype,np.float32)

    def test_zero_decay_and_invalid_policies(self):
        source={'w':np.array([5.],dtype=np.float32)};target={'w':np.array([-1.],dtype=np.float32)}
        ema.average_state_(target,source,{'w'},0)
        np.testing.assert_array_equal(target['w'],source['w'])
        for value in (-.1,1,1.1,float('nan'),float('inf'),True,'0.99'):
            with self.assertRaises(ValueError):ema.validate_decay(value)

    def test_structure_rejected_before_any_mutation(self):
        target={'a':np.array([1.],dtype=np.float32),'b':np.zeros(2,dtype=np.int64)}
        source={'a':np.array([9.],dtype=np.float32),'b':np.zeros(3,dtype=np.int64)}
        with self.assertRaises(ValueError):ema.average_state_(target,source,{'a'},.5)
        self.assertEqual(target['a'][0],1.)
        with self.assertRaises(ValueError):ema.average_state_(target,{'a':source['a']},{'a'},.5)
        with self.assertRaises(ValueError):ema.average_state_(target,target,{'unknown'},.5)
        with self.assertRaises(ValueError):ema.average_state_(target,dict(target,a=source['a'].astype(np.float64)),{'a'},.5)

    def test_source_patches_compile_and_are_anchored(self):
        _,original=prepare.base.patched_sources(prepare.BASE/'vendor')
        text=prepare.patch_controller(original);ast.parse(text)
        runner=prepare.patch_runner((prepare.BASE/'scripts/run.py').read_text());ast.parse(runner)
        self.assertIn('on_update=lambda:ema.update(model)',text)
        self.assertIn("stop='ema_validation_budget'",text)
        self.assertIn("best_ema.pt",text)
        self.assertIn('paired_completed_epochs=len(paired_history)',text)
        self.assertIn("if stale>=SET['patience']",text)
        self.assertNotIn('stale=ema',text)
        self.assertIn('server_self_check',runner)
        self.assertIn("checkpoint_name='best_ema.pt'",runner)
        with self.assertRaises(ValueError):prepare.patch_controller(text)
        # Frozen successful-update gating is part of the registered vendor contract.
        r08=(prepare.BASE/'vendor/original/src/glaucoma/r08.py').read_text()
        self.assertIn('if on_update is not None and updated:on_update()',r08)
        records=json.loads((prepare.BASE/'vendor/SOURCE_SHA256.json').read_text())
        for name,digest in records.items():self.assertEqual(prepare.base.sha(prepare.BASE/'vendor'/name),digest)

    def test_prepare_isolated_atomic_and_fully_hashed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';data.mkdir()
            for split in ('train','valid'):write_official(data,split)
            hashes={s:hashlib.sha256(prepare.base.manifest_data.render_manifest(data,s)).hexdigest() for s in ('train','valid')}
            weight=root/'best.pt';weight.write_bytes(b'fixture-not-a-model')
            out=root/'experiment'
            argv=['prepare','--output',str(out),'--data-root',str(data),'--global-checkpoint',str(weight)]
            with patch.object(prepare.base,'GLOBAL_SHA',prepare.base.sha(weight)),patch.dict(prepare.base.manifest_data.EXPECTED_SHA,hashes),patch.object(sys,'argv',argv):
                prepare.main()
                with self.assertRaises(FileExistsError):prepare.main()
            meta=json.loads((out/'PREPARATION.json').read_text())
            self.assertEqual(meta['variant'],'r16_local_ema')
            self.assertFalse(meta['training_started']);self.assertEqual(meta['ema']['decay'],.999)
            self.assertEqual(list((out/'runs').iterdir()),[])
            self.assertEqual(meta['runner_sha256'],prepare.base.sha(out/'run.py'))
            for name,digest in meta['source_sha256'].items():self.assertEqual(prepare.base.sha(out/name),digest)
            config=json.loads((out/'configs/r16_registration_20261026.json').read_text())
            archived=json.loads((prepare.BASE/'vendor/original/configs/r16_registration_20261026.json').read_text())
            for key in ('seed','local_epochs','patience','backbone_lr','head_lr','weight_decay','selection','fusion'):
                self.assertEqual(config[key],archived[key])
            self.assertEqual(config['local_seconds'],10800)
            self.assertTrue((out/'reports/R16-local-ema/split_registration.json').is_file())
            self.assertFalse((out/'reports/R16-roi-consistency').exists())
            self.assertEqual(list(root.glob('.ema_prepare_*')),[])
            with patch.object(sys,'argv',['run.py','--help']):
                with self.assertRaises(SystemExit) as done:runpy.run_path(str(out/'run.py'),run_name='__main__')
            self.assertEqual(done.exception.code,0)

    def test_failed_preparation_never_publishes_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'experiment'
            def fail():raise RuntimeError('staging failed')
            with patch.object(sys,'argv',['prepare','--output',str(out),'--data-root',temp,'--global-checkpoint',temp]),patch.object(prepare.base,'main',fail):
                with self.assertRaisesRegex(RuntimeError,'staging failed'):prepare.main()
            self.assertFalse(out.exists());self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':
    temp=HERE/'outputs/test_tmp';temp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temp)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(EMAChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if 'torch' in sys.modules:raise RuntimeError('Local checks imported PyTorch')
    raise SystemExit(not result.wasSuccessful())
