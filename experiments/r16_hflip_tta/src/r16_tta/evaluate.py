"""Explicit GPU check, then fixed full-valid comparison. No training/test path."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

from .contracts import WEIGHTS,REFERENCE_SHA,POLICY,sha,validation_rows,check_sample,average_views,baseline_deltas


def dump(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def verify_project(root):
    config=json.loads((root/'EXPERIMENT.json').read_bytes())
    if config['policy']!=POLICY:raise ValueError('Fixed TTA policy differs')
    for name,expected in config['source_sha256'].items():
        if sha(root/name)!=expected:raise ValueError('Prepared source/reference changed: '+name)
    for name,expected in WEIGHTS.items():
        item=config['weights'][name]
        if item['sha256']!=expected or sha(item['path'])!=expected:
            raise ValueError('Original R16 weight changed: '+name)
    if config['reference_sha256']!=REFERENCE_SHA or sha(root/'reference_predictions.csv')!=REFERENCE_SHA:
        raise ValueError('Original reference predictions changed')
    if sha(Path(config['data_root'])/'valid.csv')!=config['official_valid_sha256']:
        raise ValueError('Official validation CSV changed')
    if config['baseline_guard']!=dict(max_abs_probability_delta=0.001,max_abs_auc_delta=0.0001,max_abs_ap_delta=0.0005):
        raise ValueError('Baseline guard differs; diagnose mismatch before changing tolerances')
    return config


def timing(values):
    import numpy as np
    return dict(n=len(values),mean_ms=float(np.mean(values)),p50_ms=float(np.quantile(values,.5)),
                p95_ms=float(np.quantile(values,.95)),max_ms=max(values))


def verify_audit_record(root,fingerprint):
    marker=root/'baseline_audit.json'
    if not marker.exists():return  # Original strict check/valid workflow is retained.
    record=json.loads(marker.read_bytes())
    if record.get('status')!='completed' or record.get('fingerprint')!=fingerprint:
        raise ValueError('Baseline audit is incomplete or uses another runtime; do not run TTA')
    if sha(record['result_path'])!=record['result_sha256']:raise ValueError('Baseline audit report changed')
    result=json.loads(Path(record['result_path']).read_bytes())
    if result['mode']!='baseline-audit' or result['n']!=5849 or not result['reproduction']['baseline_reproduced']:
        raise ValueError('Baseline audit did not reproduce the archive; resolve differences before TTA')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('check','baseline-audit','valid'),required=True)
    parser.add_argument('--output-dir',type=Path,help='Must be NEW; default outputs/<mode>_<UTC timestamp>')
    parser.add_argument('--max-minutes',type=float,default=30)
    args=parser.parse_args()
    if not math.isfinite(args.max_minutes) or args.max_minutes<=0:parser.error('--max-minutes must be positive and finite')
    if sys.platform!='linux':parser.error('Model inference is server-only. Run the CPU tests locally instead.')
    root=Path(__file__).resolve().parents[2]
    config=verify_project(root)
    rows=validation_rows(root/'valid_manifest.csv',Path(config['data_root'])/'valid.csv',root/'reference_predictions.csv')
    output=(args.output_dir or root/'outputs'/(args.mode+'_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    # Independent outputs and an OS lock protect against concurrent GPU runs.
    import fcntl
    with (root/'.inference.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        output.mkdir(parents=True,exist_ok=False)
        if args.mode=='check':(root/'check_passed.json').unlink(missing_ok=True)
        if args.mode=='baseline-audit':dump(root/'baseline_audit.json',dict(status='running',output=str(output)))
        dump(output/'status.json',dict(status='running',mode=args.mode,started_utc=datetime.now(timezone.utc).isoformat()))
        try:
            run(root,output,config,rows,args)
        except BaseException as error:
            dump(output/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                 mode=args.mode,error=str(error),traceback=traceback.format_exc(),complete=False))
            raise
    return 0


def run(root,output,config,rows,args):
    if args.mode=='baseline-audit':
        from .audit import run_audit
        return run_audit(root,output,config,rows,args)
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
    from .pipeline import predict_view
    require_official_dev_environment(torch.__version__,torchvision.__version__)
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required; CPU inference is intentionally not enabled here')
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),
                 numpy=np.__version__,pandas=pd.__version__,pillow=pillow_version,gpu=torch.cuda.get_device_name(0),
                 cuda=torch.version.cuda,platform=platform.platform())
    fingerprint=hashlib.sha256(json.dumps(dict(project=str(root.resolve()),config=config,runtime=runtime),sort_keys=True).encode()).hexdigest()
    dump(output/'registration.json',dict(mode=args.mode,fingerprint=fingerprint,config=config,runtime=runtime,
         selection_caveat='Repeatedly researched official valid, not independent blind evidence',
         image_identity_check='Manifest/CSV alignment and archived-probability reproduction; no JPEG content hash claim'))
    if args.mode=='valid':
        verify_audit_record(root,fingerprint)
        marker=json.loads((root/'check_passed.json').read_bytes())
        if marker['fingerprint']!=fingerprint or sha(marker['result_path'])!=marker['result_sha256']:
            raise ValueError('Run the 32-image check successfully in this exact prepared project/runtime first')
    configure_runtime(False)
    models=[]
    for name,architecture,size,epoch in (('global','efficientnet_b2',512,7),('local','convnext_tiny',384,15)):
        checkpoint=torch.load(config['weights'][name]['path'],map_location='cpu',weights_only=True)
        model_config=checkpoint['config']
        if (model_config['model']!=architecture or model_config['image_size']!=size
                or model_config.get('use_age',False) or checkpoint['epoch']!=epoch):
            raise ValueError('Wrong original R16 checkpoint architecture/epoch: '+name)
        model=build_model(model_config,pretrained=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        models.append(model.cuda().eval().requires_grad_(False))
        del checkpoint
    pipe=R08Pipeline(*models,'cuda')
    selected=check_sample(rows) if args.mode=='check' else rows
    data=Path(config['data_root']).resolve()
    for row in selected[:3]:
        path=(data/row['relative_path']).resolve()
        if not path.is_relative_to(data):raise ValueError('Image escapes data root')
        pipe.predict_details(path);predict_view(pipe,path)
    torch.cuda.synchronize()
    baseline_times=[];tta_times=[];predictions=[];identity_checks=0
    fields=['image_id','label','reference_probability','original_probability','flipped_probability','tta_probability',
            'original_global_logit','original_local_logit','original_fallback',
            'flipped_global_logit','flipped_local_logit','flipped_fallback','original_ms','tta_total_ms']
    started=time.monotonic()
    with (output/'predictions.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for index,row in enumerate(selected):
            if time.monotonic()-started>args.max_minutes*60:raise TimeoutError('Inference budget exceeded; partial output retained, no completion marker')
            path=(data/row['relative_path']).resolve()
            if not path.is_relative_to(data):raise ValueError('Image escapes data root')
            torch.cuda.synchronize();start=time.perf_counter()
            original=pipe.predict_details(path)
            torch.cuda.synchronize();middle=time.perf_counter()
            flipped=predict_view(pipe,path)
            torch.cuda.synchronize();end=time.perf_counter()
            # Both decodes, CPU preprocessing, GPU passes, CAM and ROI/fallback are timed.
            baseline_ms=(middle-start)*1000;total_ms=(end-start)*1000
            baseline_times.append(baseline_ms);tta_times.append(total_ms)
            item=dict(image_id=row['image_id'],label=row['label'],reference_probability=row['reference_probability'],
                      original_probability=original['probability'],flipped_probability=flipped['probability'],
                      tta_probability=average_views(original['probability'],flipped['probability']),
                      original_ms=baseline_ms,tta_total_ms=total_ms)
            for prefix,details in (('original',original),('flipped',flipped)):
                item.update({prefix+'_'+key:details[key] for key in ('global_logit','local_logit','fallback')})
            predictions.append(item);writer.writerow(item)
            if index%25==0:stream.flush()
            if abs(original['probability']-row['reference_probability'])>config['baseline_guard']['max_abs_probability_delta']:
                raise ValueError('Archived baseline probability not reproduced: '+row['image_id']+'; diagnose weights, data and runtime before TTA')
            if args.mode=='check':
                # Strong GPU check that the copied preprocessing/forward, WITHOUT flip,
                # exactly matches the unchanged original implementation on these images.
                same=predict_view(pipe,path,mirrored=False)
                if same!=original:raise ValueError('Unflipped new helper differs from original R08Pipeline: '+row['image_id'])
                identity_checks+=1
            if (index+1)%25==0 or index+1==len(selected):
                print(f"R16_TTA {args.mode} {index+1}/{len(selected)} elapsed={time.monotonic()-started:.1f}s",flush=True)
    frame=pd.DataFrame(predictions)
    deltas=baseline_deltas(frame.original_probability,frame.reference_probability)
    result=dict(status='completed',mode=args.mode,n=len(frame),fingerprint=fingerprint,policy=POLICY,
                baseline_reproduction=deltas,unflipped_helper_identity_checks=identity_checks,
                timing=dict(original=timing(baseline_times),tta=timing(tta_times),
                            scope='Serial B1, original JPEG open/decode + preprocessing + live CAM/ROI + forward + GPU synchronization; both views for TTA',
                            warmup_images=3,arm_order='original then flip, OS caches uncontrolled',
                            t4_accepted=False,hardware=runtime['gpu']),
                fallback_counts={view:int(frame[view+'_fallback'].sum()) for view in ('original','flipped')},
                hard_label_changes_original_to_tta=int(((frame.original_probability>=.5)!=(frame.tta_probability>=.5)).sum()),
                test_used=False,test_exported=False,training=False,reference_only_not_blind=True)
    if args.mode=='valid':
        manifest=frame[['image_id','label']]
        result['metrics']={view:evaluate_predictions(manifest,frame[['image_id',column]].rename(columns={column:'probability'}))
                           for view,column in (('reference','reference_probability'),('original','original_probability'),
                                               ('flipped','flipped_probability'),('tta','tta_probability'))}
        reference=result['metrics']['reference'];original=result['metrics']['original'];tta=result['metrics']['tta']
        for metric,tolerance in (('auc_probability','max_abs_auc_delta'),('ap_probability','max_abs_ap_delta')):
            if abs(original[metric]-reference[metric])>config['baseline_guard'][tolerance]:
                dump(output/'baseline_mismatch.json',result)
                raise ValueError('Full validation baseline metric differs: '+metric)
        result['tta_minus_original']={key:tta[key]-original[key] for key in
                                      ('auc_probability','ap_probability','prauc_trapezoid_probability','tp','fn','fp','tn')}
        result['interpretation']='Single fixed comparison, no ratio/threshold search; validation change alone does not prove leaderboard gain'
    # End-of-run guard also verifies the weights/source/reference were left unchanged.
    verify_project(root)
    dump(output/'results.json',result)
    dump(output/'status.json',dict(status='completed',mode=args.mode,n=len(frame),complete=True))
    dump(output/'complete.json',dict(fingerprint=fingerprint,files={name:sha(output/name) for name in
         ('registration.json','predictions.csv','results.json','status.json')}))
    if args.mode=='check':
        dump(root/'check_passed.json',dict(fingerprint=fingerprint,result_path=str(output/'results.json'),result_sha256=sha(output/'results.json')))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    print('Report:',output,'; no training, test export or platform operation',flush=True)
