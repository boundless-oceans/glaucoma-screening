"""Explicit server-only single-flip validation and test export. Never trains."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

from .contracts import sha,probability
from .single_contracts import SINGLE_POLICY,test_rows,write_submission,verify_stage_marker


def run_single(root,output,config,rows,args):
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    from PIL import __version__ as pillow_version
    from glaucoma.models import build_model
    from glaucoma.r08 import configure_runtime
    from glaucoma.r08_export import R08Pipeline
    from glaucoma.r13_runtime import require_official_dev_environment
    from glaucoma.evaluate import evaluate_predictions
    from .paired_reference import load_reference
    from .pipeline import predict_view
    from .evaluate import dump,timing,verify_project

    require_official_dev_environment(torch.__version__,torchvision.__version__)
    if not torch.cuda.is_available():raise RuntimeError('CUDA server required')
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),
                 numpy=np.__version__,pandas=pd.__version__,pillow=pillow_version,gpu=torch.cuda.get_device_name(0),
                 cuda=torch.version.cuda,platform=platform.platform())
    fingerprint=hashlib.sha256(json.dumps(dict(project=str(root.resolve()),config=config,runtime=runtime),sort_keys=True).encode()).hexdigest()
    is_test=args.mode=='single-test';data=Path(config['data_root']).resolve();control=None;provenance=None
    if is_test:
        verify_stage_marker(root,'single_valid_passed.json',fingerprint)
        rows=test_rows(data,config.get('official_test_sha256'))
    else:
        control,provenance=load_reference(args.baseline_audit,config,runtime,rows)
        verify_stage_marker(root,'single_check_passed.json',fingerprint,provenance)
    dump(output/'registration.json',dict(mode=args.mode,fingerprint=fingerprint,config=config,runtime=runtime,
         actual_policy=SINGLE_POLICY,paired_control=provenance,
         selection_caveat='Single flip chosen after viewing prior validation diagnostics; not a prespecified blind improvement'))
    configure_runtime(False);models=[]
    for name,architecture,size,epoch in (('global','efficientnet_b2',512,7),('local','convnext_tiny',384,15)):
        ck=torch.load(config['weights'][name]['path'],map_location='cpu',weights_only=True);c=ck['config']
        if c['model']!=architecture or c['image_size']!=size or c.get('use_age',False) or ck['epoch']!=epoch:raise ValueError('Wrong original checkpoint')
        m=build_model(c,False);m.load_state_dict(ck['state_dict'],strict=True)
        models.append(m.cuda().eval().requires_grad_(False));del ck
    pipe=R08Pipeline(*models,'cuda')
    def path_for(row):
        path=(data/row['relative_path']).resolve()
        if not path.is_relative_to(data):raise ValueError('Image escapes data root')
        return path
    for row in rows[:3]:predict_view(pipe,path_for(row),mirrored=True)
    torch.cuda.synchronize();started=time.monotonic();times=[];predictions=[]
    fields=['image_id','probability','global_logit','local_logit','fallback','single_flip_ms']
    if not is_test:fields+=['label','baseline_probability','archived_probability']
    with (output/'predictions.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(rows):
            if time.monotonic()-started>args.max_minutes*60:raise TimeoutError('Single-view budget exceeded; partial predictions retained')
            path=path_for(row)
            torch.cuda.synchronize();start=time.perf_counter();details=predict_view(pipe,path,mirrored=True)
            torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start);times.append(elapsed)
            probability(details['probability'])
            item=dict(image_id=row['image_id'],**details,single_flip_ms=elapsed)
            if not is_test:item.update(label=row['label'],baseline_probability=control[row['image_id']],archived_probability=row['reference_probability'])
            predictions.append(item);writer.writerow(item)
            if i%25==0:stream.flush()
            if (i+1)%100==0 or i+1==len(rows):print(f'R16_SINGLE_FLIP {args.mode} {i+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    result=dict(status='completed',mode=args.mode,n=len(rows),fingerprint=fingerprint,actual_policy=SINGLE_POLICY,
                timing=dict(single_flip=timing(times),scope='One JPEG decode + mirror + live CAM/ROI + forwards + synchronization; three warmups excluded',t4_accepted=False),
                training=False,test_used=is_test,test_exported=is_test,paired_control=provenance)
    frame=pd.DataFrame(predictions)
    if not is_test:
        manifest=frame[['image_id','label']]
        result['metrics']={view:evaluate_predictions(manifest,frame[['image_id',column]].rename(columns={column:'probability'})) for view,column in
                           (('single_flip','probability'),('frozen_original','baseline_probability'),('archived','archived_probability'))}
        result['single_minus_frozen_original']={k:result['metrics']['single_flip'][k]-result['metrics']['frozen_original'][k] for k in ('auc_probability','ap_probability','prauc_trapezoid_probability','tp','fn','fp','tn')}
        _,end=load_reference(args.baseline_audit,config,runtime,rows)
        if end!=provenance:raise ValueError('Frozen audit changed')
    verify_project(root)
    files=['registration.json','predictions.csv','results.json','status.json']
    if is_test:
        if sha(data/'test.csv')!=config['official_test_sha256']:raise ValueError('Test metadata changed')
        write_submission(output/'submission_single_flip.csv',rows,frame.probability.tolist());files+=['submission_single_flip.csv']
    dump(output/'results.json',result);dump(output/'status.json',dict(status='completed',mode=args.mode,complete=True,n=len(rows)))
    dump(output/'complete.json',dict(fingerprint=fingerprint,files={n:sha(output/n) for n in files}))
    if not is_test:dump(root/'single_valid_passed.json',dict(fingerprint=fingerprint,result_path=str(output/'results.json'),result_sha256=sha(output/'results.json')))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True);print('Report:',output,flush=True)
