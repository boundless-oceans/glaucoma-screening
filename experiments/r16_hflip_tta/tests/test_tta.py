"""CPU contracts only: no PyTorch import, model loading, training or download."""
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
from PIL import Image,ImageOps

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
sys.path.insert(0,str(HERE.parent/'r16_roi_consistency/vendor/original/src'))
from r16_tta import contracts
from r16_tta import evaluate
from r16_tta.diagnostics import cam_peak_summary,audit_decision
from r16_tta.imaging import prepare_view
from glaucoma.local_region import prepare_shared,crop_decoded

spec=importlib.util.spec_from_file_location('prepare_tta',HERE/'scripts/prepare.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


def fixtures(root):
    data=root/'data';data.mkdir()
    manifest=HERE.parent/'r16_roi_consistency/vendor/original/manifests/official_valid.csv'
    rows=contracts.read_csv(manifest)
    with (data/'valid.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Image_name','True_Label'])
        writer.writerows((r['image_id'],r['label']) for r in rows)
    reference=root/'reference.csv'
    with reference.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['Image_name','Label','Probability','Pred_Label'])
        writer.writerows((r['image_id'],r['label'],'.8' if r['label']=='1' else '.2',r['label']) for r in rows)
    return data,manifest,reference


class TtaContracts(unittest.TestCase):
    def test_failed_or_incomplete_audit_blocks_tta_before_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            evaluate.verify_audit_record(root,'test')
            marker=root/'baseline_audit.json';marker.write_text(json.dumps(dict(status='running')))
            with self.assertRaisesRegex(ValueError,'incomplete'):evaluate.verify_audit_record(root,'test')
            report=root/'result.json'
            report.write_text(json.dumps(dict(mode='baseline-audit',n=5849,reproduction=dict(baseline_reproduced=False))))
            marker.write_text(json.dumps(dict(status='completed',fingerprint='test',result_path=str(report),result_sha256=contracts.sha(report))))
            with self.assertRaisesRegex(ValueError,'did not reproduce'):evaluate.verify_audit_record(root,'test')
            report.write_text(report.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'report changed'):evaluate.verify_audit_record(root,'test')

    def test_cam_summary_uses_original_eligible_peak_and_absolute_heat(self):
        from glaucoma.local_region import attention_box
        cam=np.zeros((16,16));mask=np.ones((16,16),dtype=bool)
        cam[0,0]=1000  # Original ellipse must exclude this edge peak.
        cam[12,7]=-4.493499755859375;cam[12,6]=4.493049144744873
        summary=cam_peak_summary(cam,mask)
        self.assertEqual((summary['peaks'][0]['y'],summary['peaks'][0]['x']),(12,7))
        self.assertLess(summary['relative_gap'],0.001)
        self.assertFalse(summary['roi_policy_changed'])
        original=attention_box(cam,mask)
        changed=cam.copy();changed[12,6]+=.001
        self.assertNotEqual(original['box'],attention_box(changed,mask)['box'])
        self.assertEqual(cam_peak_summary(changed,mask)['peaks'][0]['x'],6)

    def test_cam_summary_ties_empty_and_nonfinite(self):
        cam=np.zeros((16,16));cam[8,7]=cam[8,8]=2
        summary=cam_peak_summary(cam,np.ones_like(cam,dtype=bool))
        self.assertEqual(summary['peaks'][0]['x'],7)
        self.assertEqual(summary['relative_gap'],0)
        self.assertEqual(cam_peak_summary(cam,np.zeros_like(cam,dtype=bool))['peaks'],[])
        cam[1,1]=np.nan
        with self.assertRaises(ValueError):cam_peak_summary(cam,np.ones_like(cam,dtype=bool))

    def test_completed_audit_does_not_accept_outlier_or_unlock_tta(self):
        guard=dict(max_abs_probability_delta=.001,max_abs_auc_delta=.0001,max_abs_ap_delta=.0005)
        metrics=dict(auc_probability=.98,ap_probability=.8,prauc_trapezoid_probability=.79)
        result=audit_decision([.0475584023,.8],[.0579711395,.8],guard,metrics,metrics)
        self.assertFalse(result['baseline_reproduced']);self.assertEqual(result['outlier_count'],1)
        self.assertFalse(result['tta_unlocked']);self.assertFalse(result['guard_changed'])
        good=audit_decision([.2,.8],[.2,.8],guard,metrics,metrics)
        self.assertTrue(good['baseline_reproduced']);self.assertFalse(good['tta_unlocked'])
        changed=dict(metrics,auc_probability=.981)
        self.assertFalse(audit_decision([.2,.8],[.2,.8],guard,changed,metrics)['baseline_reproduced'])

    def test_audit_rejects_invalid_probability_arrays(self):
        for a,b in (([],[]),([.2],[.2,.3]),([float('nan')],[.2]),([1.1],[.2])):
            with self.assertRaises(ValueError):audit_decision(a,b,{}, {}, {})

    def test_probability_mean_is_not_logit_mean(self):
        self.assertAlmostEqual(contracts.average_views(.1,.8),.45)
        self.assertEqual(contracts.average_views(0,1),.5)
        for bad in (-.01,1.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):contracts.average_views(bad,.5)

    def test_reference_is_joined_by_identity_not_row_position(self):
        with tempfile.TemporaryDirectory() as temporary:
            data,manifest,reference=fixtures(Path(temporary))
            a=contracts.validation_rows(manifest,data/'valid.csv',reference)
            rows=contracts.read_csv(reference)
            with reference.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(reversed(rows))
            self.assertEqual(a,contracts.validation_rows(manifest,data/'valid.csv',reference))
            sample=contracts.check_sample(a)
            self.assertEqual(sample,contracts.check_sample(a))
            self.assertEqual((len(sample),sum(x['label'] for x in sample)),(32,16))

    def test_reference_label_and_duplicate_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            data,manifest,reference=fixtures(Path(temporary))
            rows=contracts.read_csv(reference);rows[0]['Label']='1' if rows[0]['Label']=='0' else '0'
            with reference.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'labels differ'):contracts.validation_rows(manifest,data/'valid.csv',reference)
            rows[0]=rows[1]
            with reference.open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'Duplicate'):contracts.validation_rows(manifest,data/'valid.csv',reference)

    def test_unflipped_geometry_pixels_match_original_with_exif_and_odd_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for width,height,orientation,mode in ((1025,767,1,'RGB'),(1725,1297,6,'RGB'),
                                                  (2592,1944,8,'RGB'),(769,513,2,'RGB'),(513,769,1,'L')):
                with self.subTest(size=(width,height),exif=orientation,mode=mode):
                    shape=(height,width,3) if mode=='RGB' else (height,width)
                    arr=np.random.default_rng(width).integers(0,256,shape,dtype=np.uint8)
                    # Dark borders and an asymmetric field exercise crop/padding and mirroring.
                    arr[:height//12]=0;arr[:,:width//9]=0
                    im=Image.fromarray(arr);exif=im.getexif();exif[274]=orientation
                    path=root/f'{width}_{orientation}_{mode}.JPG';im.save(path,quality=91,exif=exif)
                    original,og,decoded=prepare_shared(path)
                    plain,pg,pixels=prepare_view(path,False)
                    self.assertEqual(og,pg)
                    np.testing.assert_array_equal(original,plain)
                    np.testing.assert_array_equal(decoded,pixels)
                    full,geometry,mirrored=prepare_view(path,True)
                    np.testing.assert_array_equal(mirrored,ImageOps.mirror(decoded))
                    # A box from the flipped CAM belongs to the flipped decoded source.
                    self.assertEqual(crop_decoded(mirrored,(.1,.2,.7,.8)).size,(384,384))
                    self.assertEqual(full.size,(512,512))
                    self.assertTrue(all(0<=x<=1 for x in geometry['field']))

    def test_probability_deltas_and_mismatched_lengths(self):
        result=contracts.baseline_deltas([.2,.5001],[.2,.4999])
        self.assertAlmostEqual(result['max_abs_probability_delta'],.0002)
        self.assertEqual(result['hard_label_changes'],1)
        with self.assertRaises(ValueError):contracts.baseline_deltas([.2],[])

    def test_preparation_pins_old_weights_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data,_,reference=fixtures(root)
            weights={name:root/(name+'.pt') for name in contracts.WEIGHTS}
            for name,path in weights.items():path.write_bytes(('undecodable-'+name).encode())
            target=root/'project'
            argv=['prepare','--output',str(target),'--data-root',str(data),
                  '--global-checkpoint',str(weights['global']),'--local-checkpoint',str(weights['local']),
                  '--reference-predictions',str(reference)]
            with patch.object(sys,'argv',argv):
                with self.assertRaisesRegex(ValueError,'original R16'):prepare.main()
                self.assertFalse(target.exists())
                fake={name:contracts.sha(path) for name,path in weights.items()}
                with patch.object(prepare,'WEIGHTS',fake),patch.object(prepare,'REFERENCE_SHA',contracts.sha(reference)):
                    prepare.main()
                    with self.assertRaises(FileExistsError):prepare.main()
                with patch.object(evaluate,'WEIGHTS',fake),patch.object(evaluate,'REFERENCE_SHA',contracts.sha(reference)):
                    evaluate.verify_project(target)
                    source=target/'src/r16_tta/pipeline.py';original=source.read_bytes()
                    source.write_bytes(original+b'\n# changed\n')
                    with self.assertRaisesRegex(ValueError,'source/reference changed'):evaluate.verify_project(target)
                    source.write_bytes(original)
                    weights['local'].write_bytes(b'changed-checkpoint')
                    with self.assertRaisesRegex(ValueError,'weight changed'):evaluate.verify_project(target)
            config=json.loads((target/'EXPERIMENT.json').read_bytes())
            self.assertEqual(config['policy'],contracts.POLICY)
            self.assertFalse(config['training']);self.assertFalse(config['test_export'])
            self.assertFalse((target/'runs').exists())
            for name,expected in config['source_sha256'].items():self.assertEqual(contracts.sha(target/name),expected)
            # --help also works in the prepared standalone project, without torch.
            with patch.object(sys,'argv',['run.py','--help']):
                with self.assertRaises(SystemExit) as result:runpy.run_path(str(target/'run.py'),run_name='__main__')
                self.assertEqual(result.exception.code,0)

    def test_help_never_imports_framework_and_rejects_local_inference(self):
        with patch.object(sys,'argv',['run.py','--help']):
            with self.assertRaises(SystemExit) as result:runpy.run_path(str(HERE/'scripts/run.py'),run_name='__main__')
            self.assertEqual(result.exception.code,0)
        if sys.platform!='linux':
            with patch.object(sys,'argv',['run.py','--mode','check']):
                with self.assertRaises(SystemExit) as result:runpy.run_path(str(HERE/'scripts/run.py'),run_name='__main__')
                self.assertEqual(result.exception.code,2)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    temporary=HERE/'outputs/test_tmp';temporary.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temporary)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TtaContracts))
    if 'torch' in sys.modules:raise RuntimeError('CPU checks imported PyTorch')
    raise SystemExit(not result.wasSuccessful())
