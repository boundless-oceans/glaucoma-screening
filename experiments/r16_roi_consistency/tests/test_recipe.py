"""CPU geometry/preparation contracts. No model imports or pretrained downloads."""
import hashlib
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_recipe',HERE/'scripts/prepare.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
sys.path.insert(0,str(HERE/'tests'))
from synthetic_data import write_official


class RecipeChecks(unittest.TestCase):
    def test_local_generator_preserves_order_and_exact_serialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            data=Path(temporary);write_official(data,'valid')
            first=prepare.manifest_data.render_manifest(data,'valid')
            with (data/'valid.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
            with (data/'valid.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(reversed(rows))
            self.assertEqual(first,prepare.manifest_data.render_manifest(data,'valid'))
            self.assertTrue(first.startswith(b'image_id,relative_path,official_split,role,label,age,patient_id\n'))
            self.assertNotIn(b'\r',first)
            with patch.dict(prepare.manifest_data.EXPECTED_SHA,{'valid':hashlib.sha256(first).hexdigest()}):
                payloads=prepare.manifest_data.manifest_payloads(data,('valid',))
                paths=prepare.manifest_data.write_manifests(data/'output',payloads)
                self.assertEqual(paths['valid'].read_bytes(),first)
                with self.assertRaises(FileExistsError):prepare.manifest_data.write_manifests(data/'output',payloads)

    def test_local_generator_refuses_changed_rows_and_untrusted_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            data=Path(temporary);write_official(data,'valid')
            # Artificial fixtures must never pass the real historical data digest.
            with self.assertRaisesRegex(ValueError,'does not reproduce'):prepare.manifest_data.manifest_payloads(data,('valid',))
            with (data/'valid.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
            def write():
                with (data/'valid.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
            rows[0]['Image_name']=rows[1]['Image_name'];write()
            with self.assertRaisesRegex(ValueError,'Duplicate'):prepare.manifest_data.render_manifest(data,'valid')
            rows[0]['Image_name']='../escape.jpg';write()
            with self.assertRaisesRegex(ValueError,'JPEG name'):prepare.manifest_data.render_manifest(data,'valid')
            rows[0]['Image_name']='valid_00000.jpg';rows[0]['True_Label']='2';write()
            with self.assertRaisesRegex(ValueError,'binary'):prepare.manifest_data.render_manifest(data,'valid')

    def test_vendored_sources_and_patches_are_narrow(self):
        records=json.loads((HERE/'vendor/SOURCE_SHA256.json').read_bytes())
        for name,expected in records.items():
            self.assertEqual(prepare.sha(HERE/'vendor'/name),expected)
            self.assertNotIn(b'\r\n',(HERE/'vendor'/name).read_bytes())
        data,controller=prepare.patched_sources(HERE/'vendor')
        original=(HERE/'vendor/original/src/glaucoma/data.py').read_text(encoding='utf-8-sig')
        self.assertEqual(data.replace(",float_precision='round_trip'",''),original)
        self.assertIn("[('train',fit,1),('valid',val,1)]",controller)
        self.assertNotIn("[('train',fit,32),('valid',val,1)]",controller)
        with self.assertRaises(ValueError):prepare.replace_once('same same','same','new')

    def test_real_boundary_counterexample_round_trips(self):
        csv='image_id,x0,y0,x1,y1,fallback\na.jpg,0.2574615478515625,0.013020833333333334,0.6954574584960938,0.597015380859375,False\n'
        old=pd.read_csv(io.StringIO(csv));new=pd.read_csv(io.StringIO(csv),float_precision='round_trip')
        self.assertEqual(new.y0.iloc[0],float('0.013020833333333334'))
        self.assertEqual(math.floor(new.y0.iloc[0]*768),10)
        # Baseline parser may differ between pandas versions; round-trip contract is stable.
        self.assertLessEqual(abs(math.floor(old.y0.iloc[0]*768)-10),1)

    def test_preparation_never_loads_models_or_overwrites_existing_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'data';data.mkdir()
            for split in ('train','valid'):write_official(data,split)
            fake_hashes={split:hashlib.sha256(prepare.manifest_data.render_manifest(data,split)).hexdigest() for split in ('train','valid')}
            weights=root/'best.pt';weights.write_bytes(b'not-deserialized-test-fixture')
            target=root/'project'
            argv=['prepare','--output',str(target),'--data-root',str(data),'--global-checkpoint',str(weights)]
            with patch.object(prepare,'GLOBAL_SHA',prepare.sha(weights)),patch.dict(prepare.manifest_data.EXPECTED_SHA,fake_hashes),patch.object(sys,'argv',argv):
                prepare.main()
                with self.assertRaises(FileExistsError):prepare.main()
            meta=json.loads((target/'PREPARATION.json').read_bytes())
            self.assertFalse(meta['training_started'])
            self.assertTrue((target/'runs').is_dir())
            self.assertEqual(list((target/'runs').iterdir()),[])
            self.assertEqual(meta['runner_sha256'],prepare.sha(target/'run.py'))
            registration=json.loads((target/'reports/R16-roi-consistency/split_registration.json').read_bytes())
            self.assertEqual((registration['train_n'],registration['valid_n']),(51396,5849))
            self.assertFalse(registration['valid_gradient_or_statistics_fit'])
            self.assertIn("float_precision='round_trip'",(target/'src/glaucoma/data.py').read_text())

    def test_same_float_coordinates_produce_same_pixels(self):
        # Load only the image/geometry modules; no dataset __getitem__ (it imports torch).
        sys.path.insert(0,str(HERE/'vendor/original/src'))
        from glaucoma.local_region import crop_decoded,prepare_local,prepare_shared
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'a.JPG'
            arr=np.random.default_rng(20261029).integers(0,255,(768,1024,3),dtype=np.uint8)
            Image.fromarray(arr).save(path,quality=95)
            box=(.2574615478515625,float('0.013020833333333334'),.6954574584960938,.597015380859375)
            _,_,decoded=prepare_shared(path)
            np.testing.assert_array_equal(np.asarray(prepare_local(path,box)),np.asarray(crop_decoded(decoded,box)))

    def test_runner_help_never_imports_framework(self):
        with patch.object(sys,'argv',['run.py','--help']):
            with self.assertRaises(SystemExit) as error:runpy.run_path(str(HERE/'scripts/run.py'),run_name='__main__')
        self.assertEqual(error.exception.code,0)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    temp=HERE/'outputs/test_tmp';temp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temp)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecipeChecks))
    if 'torch' in sys.modules:raise RuntimeError('CPU checks imported PyTorch')
    raise SystemExit(not result.wasSuccessful())
