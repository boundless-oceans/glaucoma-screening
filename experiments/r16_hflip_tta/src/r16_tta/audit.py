"""Server-only complete original-baseline audit. No TTA or ROI policy change."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

from .contracts import sha
from .diagnostics import cam_peak_summary,audit_decision


def run_audit(root,output,config,rows,args):
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    from PIL import __version__ as pillow_version
    from glaucoma.models import build_model
    from glaucoma.r08 import configure_runtime
    from glaucoma.r08_export import R08Pipeline
    from glaucoma.r13_runtime import require_official_dev_environment
    from glaucoma.local_region import prepare_shared,image_tensor,region_from_cam
    from glaucoma.evaluate import evaluate_predictions
    from .pipeline import predict_view
    from .evaluate import dump,timing,verify_project

    require_official_dev_environment(torch.__version__,torchvision.__version__)
    if not torch.cuda.is_available():raise RuntimeError('CUDA server required')
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),
                 numpy=np.__version__,pandas=pd.__version__,pillow=pillow_version,gpu=torch.cuda.get_device_name(0),
                 cuda=torch.version.cuda,platform=platform.platform())
    fingerprint=hashlib.sha256(json.dumps(dict(project=str(root.resolve()),config=config,runtime=runtime),sort_keys=True).encode()).hexdigest()
    dump(output/'registration.json',dict(mode='baseline-audit',fingerprint=fingerprint,config=config,runtime=runtime,
         purpose='Complete original-only reproduction audit; never permits TTA or claims a new baseline',
         timing_excludes='Extra outlier diagnostic forwards, CSV writes, hashing and warmup',
         image_identity_check='CSV/manifest aligned; source hashes collected for outliers, no archive JPEG hash claim'))
    configure_runtime(False);models=[]
    for name,architecture,size,epoch in (('global','efficientnet_b2',512,7),('local','convnext_tiny',384,15)):
        ck=torch.load(config['weights'][name]['path'],map_location='cpu',weights_only=True)
        c=ck['config']
        if c['model']!=architecture or c['image_size']!=size or c.get('use_age',False) or ck['epoch']!=epoch:
            raise ValueError('Wrong original checkpoint architecture/epoch')
        model=build_model(c,False);model.load_state_dict(ck['state_dict'],strict=True)
        models.append(model.cuda().eval().requires_grad_(False));del ck
    pipe=R08Pipeline(*models,'cuda');data=Path(config['data_root']).resolve()
    def image_path(row):
        path=(data/row['relative_path']).resolve()
        if not path.is_relative_to(data):raise ValueError('Image escapes data root')
        return path
    for row in rows[:3]:pipe.predict_details(image_path(row))
    torch.cuda.synchronize();started=time.monotonic();predictions=[];times=[];outliers=[]
    (output/'outliers').mkdir()
    fields=['image_id','label','reference_probability','probability','absolute_delta','global_logit','local_logit','fallback','original_ms']
    with (output/'predictions.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(rows):
            if time.monotonic()-started>args.max_minutes*60:raise TimeoutError('Baseline audit budget exceeded; partial output retained')
            path=image_path(row)
            torch.cuda.synchronize();start=time.perf_counter();original=pipe.predict_details(path)
            torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start);times.append(elapsed)
            value=float(original['probability']);reference=row['reference_probability']
            if not np.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid original probability')
            delta=abs(value-reference)
            item=dict(image_id=row['image_id'],label=row['label'],reference_probability=reference,
                      **original,absolute_delta=delta,original_ms=elapsed)
            predictions.append(item);writer.writerow(item)
            if i%25==0:stream.flush()
            if delta>config['baseline_guard']['max_abs_probability_delta']:
                full,geometry,decoded=prepare_shared(path)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    features=models[0].backbone.features(image_tensor(full)[None].cuda())
                    cam=(features.float()*models[0].head[-1].weight[0].float()[None,:,None,None]).sum(1)[0].cpu().numpy()
                roi=region_from_cam(cam,full,geometry)
                repeat=pipe.predict_details(path);helper=predict_view(pipe,path,mirrored=False)
                report=dict(image_id=row['image_id'],reference_probability=reference,original=original,
                            repeated_original=repeat,repeat_equal=repeat==original,unflipped_helper=helper,
                            helper_equal=helper==original,source_jpeg_sha256=sha(path),
                            global_rgb_sha256=hashlib.sha256(np.asarray(full).tobytes()).hexdigest(),
                            decoded_size=decoded.size,geometry=geometry,roi=roi,
                            cam_peak_summary=cam_peak_summary(cam,np.asarray(full.resize((cam.shape[1],cam.shape[0]))).max(2)>25),
                            note='Extra forward diagnostics, not a captured CAM from the timed call; no new crop/threshold policy')
                name=f'outlier_{i:05d}.json';dump(output/'outliers'/name,report)
                outliers.append(dict(image_id=row['image_id'],absolute_delta=delta,report='outliers/'+name))
            if (i+1)%100==0 or i+1==len(rows):print(f'R16_BASELINE_AUDIT {i+1}/{len(rows)} outliers={len(outliers)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    frame=pd.DataFrame(predictions);manifest=frame[['image_id','label']]
    actual=evaluate_predictions(manifest,frame[['image_id','probability']])
    reference=evaluate_predictions(manifest,frame[['image_id','reference_probability']].rename(columns={'reference_probability':'probability'}))
    decision=audit_decision(frame.probability,frame.reference_probability,config['baseline_guard'],actual,reference)
    result=dict(status='completed',mode='baseline-audit',n=len(frame),fingerprint=fingerprint,
                original_metrics=actual,reference_metrics=reference,reproduction=decision,outliers=outliers,
                timing=dict(original=timing(times),scope='Original JPEG decode + live CAM/ROI + forward + synchronization; extra diagnostic work excluded',t4_accepted=False),
                training=False,test_used=False,test_exported=False,roi_policy_changed=False,
                interpretation='Audit completion is not baseline acceptance; original valid is repeatedly researched, not blind evidence')
    verify_project(root);dump(output/'results.json',result)
    dump(output/'status.json',dict(status='completed',mode='baseline-audit',n=len(frame),complete=True,baseline_reproduced=decision['baseline_reproduced'],tta_unlocked=False))
    files=['registration.json','predictions.csv','results.json','status.json']+[r['report'] for r in outliers]
    dump(output/'complete.json',dict(fingerprint=fingerprint,files={name:sha(output/name) for name in files}))
    dump(root/'baseline_audit.json',dict(status='completed',fingerprint=fingerprint,
         result_path=str(output/'results.json'),result_sha256=sha(output/'results.json')))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    print('BASELINE_AUDIT_COMPLETE; no TTA permission marker written. Report:',output,flush=True)
