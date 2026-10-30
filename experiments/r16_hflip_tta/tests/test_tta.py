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
from r16_tta.paired_reference import load_reference
from r16_tta.single_contracts import write_submission,test_rows,verify_stage_marker,SINGLE_POLICY
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
    def test_single_flip_submission_order_and_near_half_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows=[dict(image_id=f'test_{i:05d}.jpg') for i in reversed(range(5855))]
            values=[.49999999999999994,.5,1e-20,1.,0.]+[.12345678901234567]*5850
            path=Path(temporary)/'submission_single_flip.csv'
            write_submission(path,rows,values);restored=contracts.read_csv(path)
            self.assertEqual(list(restored[0]),['Image_name','Probability','Pred_Label'])
            self.assertEqual([r['Image_name'] for r in restored],[r['image_id'] for r in rows])
            self.assertEqual([float(r['Probability']) for r in restored],values)
            self.assertEqual([r['Pred_Label'] for r in restored[:2]],['0','1'])
            with self.assertRaises(FileExistsError):write_submission(path,rows,values)
            self.assertEqual(SINGLE_POLICY['views'],['horizontal_flip'])

    def test_single_flip_submission_rejects_missing_duplicate_nonfinite(self):
        rows=[dict(image_id=f'test_{i:05d}.jpg') for i in range(5855)]
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'submission.csv'
            with self.assertRaises(ValueError):write_submission(path,rows,[.5])
            with self.assertRaises(ValueError):write_submission(path,rows,[float('nan')]+[.5]*5854)
            rows[0]=rows[1]
            with self.assertRaises(ValueError):write_submission(path,rows,[.5]*5855)
            self.assertFalse(path.exists())

    def test_test_metadata_case_mapping_and_official_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            data=Path(temporary);metadata=data/'test.csv'
            names=[f'test_{i:05d}.jpg' for i in reversed(range(5855))]
            metadata.write_text('Image_name,Age\n'+'\n'.join(n+',60' for n in names))
            paths=[data/'test'/n.replace('.jpg','.JPG') for n in reversed(names)]
            with patch.object(Path,'iterdir',return_value=iter(paths)),patch.object(Path,'is_file',return_value=True):
                rows=test_rows(data,contracts.sha(metadata))
            self.assertEqual([r['image_id'] for r in rows],names)
            self.assertTrue(rows[0]['relative_path'].endswith('.JPG'))
            with self.assertRaises(ValueError):test_rows(data,'changed')
            metadata.write_text('Image_name,True_Label\n'+'\n'.join(n+',0' for n in names))
            with self.assertRaisesRegex(ValueError,'labels'):test_rows(data,contracts.sha(metadata))

    def test_single_view_requires_completed_same_source_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);result=root/'result.json';marker=root/'single_valid_passed.json'
            with self.assertRaises(FileNotFoundError):verify_stage_marker(root,marker.name,'test')
            result.write_text(json.dumps(dict(status='completed',mode='single-valid',n=5849)))
            marker.write_text(json.dumps(dict(fingerprint='test',result_path=str(result),result_sha256=contracts.sha(result))))
            verify_stage_marker(root,marker.name,'test')
            with self.assertRaises(ValueError):verify_stage_marker(root,marker.name,'another')
            result.write_text(result.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'changed'):verify_stage_marker(root,marker.name,'test')

    def paired_fixture(self,root):
        data,manifest,reference=fixtures(root)
        rows=contracts.validation_rows(manifest,data/'valid.csv',reference)
        config=dict(policy=contracts.POLICY,official_valid_sha256='csv',reference_sha256='ref',weights={'global':{'sha256':'g'},'local':{'sha256':'l'}},
                    source_sha256={'src/glaucoma/models.py':'model','src/r16_tta/imaging.py':'image','src/r16_tta/pipeline.py':'pipe','valid_manifest.csv':'manifest'},
                    baseline_guard=dict(max_abs_probability_delta=.001,max_abs_auc_delta=.0001,max_abs_ap_delta=.0005))
        folder=root/'audit';folder.mkdir();(folder/'outliers').mkdir()
        predictions=[]
        for i,row in enumerate(rows):
            predictions.append(dict(image_id=row['image_id'],label=row['label'],reference_probability=row['reference_probability'],probability=row['reference_probability']-(.01 if i==0 else 0)))
        with (folder/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=predictions[0].keys());writer.writeheader();writer.writerows(predictions)
        metrics=dict(auc_probability=.98,ap_probability=.8)
        payloads={'registration.json':dict(fingerprint='audit',runtime={'gpu':'test'},config=config),
                  'results.json':dict(status='completed',mode='baseline-audit',n=5849,fingerprint='audit',original_metrics=metrics,reference_metrics=metrics,
                                      outliers=[dict(image_id=rows[0]['image_id'],report='outliers/one.json')],reproduction=dict(baseline_reproduced=False)),
                  'status.json':dict(status='completed'),'outliers/one.json':dict(repeat_equal=True,helper_equal=True)}
        for name,value in payloads.items():(folder/name).write_text(json.dumps(value),encoding='utf-8')
        (folder/'complete.json').write_text(json.dumps(dict(fingerprint='audit',files={name:contracts.sha(folder/name) for name in (*payloads,'predictions.csv')})))
        return folder,config,rows

    def test_paired_control_keeps_archive_failure_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,config,rows=self.paired_fixture(Path(temporary))
            control,provenance=load_reference(folder,config,{'gpu':'test'},rows)
            self.assertAlmostEqual(control[rows[0]['image_id']],.19)
            self.assertFalse(provenance['archival_strict_reproduction'])
            self.assertTrue(provenance['archive_is_context_not_paired_control'])

    def test_paired_control_rejects_tampering_and_runtime_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,config,rows=self.paired_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError,'runtime differs'):load_reference(folder,config,{'gpu':'other'},rows)
            (folder/'predictions.csv').write_text('modified')
            with self.assertRaisesRegex(ValueError,'artifact changed'):load_reference(folder,config,{'gpu':'test'},rows)

    def test_paired_control_rejects_different_inference_or_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,config,rows=self.paired_fixture(Path(temporary))
            config=json.loads(json.dumps(config));config['source_sha256']['src/r16_tta/pipeline.py']='changed'
            with self.assertRaisesRegex(ValueError,'inference source differs'):load_reference(folder,config,{'gpu':'test'},rows)
            config['source_sha256']['src/r16_tta/pipeline.py']='pipe';rows[0]['label']=1
            with self.assertRaisesRegex(ValueError,'labels/archive'):load_reference(folder,config,{'gpu':'test'},rows)

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
