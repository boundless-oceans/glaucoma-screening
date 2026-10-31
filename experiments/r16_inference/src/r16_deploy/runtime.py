"""Explicit server inference; no training, data fitting or automatic downloads."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from .core import sha,validate_bundle,inputs,select,summary,timed_call,validate_probability


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('predict','benchmark'):
        command=sub.add_parser(name)
        command.add_argument('--view',choices=('original','single-flip'),required=True)
        for flag in ('images','metadata','output-dir'):command.add_argument('--'+flag,type=Path,required=True)
        command.add_argument('--max-minutes',type=float,default=30)
        if name=='predict':command.add_argument('--expected-count',type=int,default=5855)
        else:
            command.add_argument('--limit',type=int,default=256,help='0 = full metadata coverage; subset is diagnostic only')
            command.add_argument('--warmup',type=int,default=20)
    a=p.parse_args()
    if not math.isfinite(a.max_minutes) or a.max_minutes<=0:p.error('Positive finite budget required')
    if a.command=='benchmark' and (a.limit<0 or a.warmup<0):p.error('Limit/warmup cannot be negative')
    if a.command=='predict' and a.expected_count<=0:p.error('Positive expected count required')
    if sys.platform!='linux':p.error('Model inference is Linux-server-only; local CPU tests do not load models')
    root=Path(__file__).resolve().parents[2];bundle=validate_bundle(root)
    rows=inputs(a.images,a.metadata,a.expected_count if a.command=='predict' else None);full_n=len(rows)
    rows=select(rows,a.limit) if a.command=='benchmark' else rows
    a.output_dir=a.output_dir.resolve();a.output_dir.mkdir(parents=True,exist_ok=False)
    dump(a.output_dir/'status.json',dict(status='running',command=a.command,view=a.view))
    try:run(root,bundle,rows,full_n,a)
    except BaseException as error:
        dump(a.output_dir/'status.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),traceback=traceback.format_exc()))
        raise
    return 0


def run(root,bundle,rows,full_n,a):
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    from PIL import __version__ as pillow_version
    from glaucoma.models import build_model
    from glaucoma.r08 import configure_runtime
    from glaucoma.r08_export import R08Pipeline
    from r16_tta.pipeline import predict_view
    if (str(torch.__version__),str(torchvision.__version__))!=('2.10.0+cu128','0.25.0+cu128') or sys.version_info[:2]!=(3,12):
        raise ValueError('Requires Python3.12, torch2.10.0+cu128 and torchvision0.25.0+cu128')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    configure_runtime(False);gpu=torch.cuda.get_device_name(0)
    try:driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired):driver=None
    metadata_sha=sha(a.metadata);bundle_sha=sha(root/'bundle.json')
    registration=dict(command=a.command,view=a.view,metadata_sha256=metadata_sha,bundle_sha256=bundle_sha,weights=bundle['weights'],
        selected_image_ids_sha256=hashlib.sha256(json.dumps([n for n,_ in rows]).encode()).hexdigest(),
        model_policy=dict(global_size=512,local_size=384,jpeg_draft_size=768,
                          horizontal_flip=a.view=='single-flip',fusion='equal_raw_logit_mean',fallback='global',
                          precision='cuda_fp16_autocast',cudnn_benchmark=False,threshold=.5),
        full_metadata_n=full_n,measured_n=len(rows),runtime=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),
        numpy=np.__version__,pandas=pd.__version__,pillow=pillow_version,gpu=gpu,driver=driver,cuda=torch.version.cuda,platform=platform.platform(),
        torch_cpu_threads=torch.get_num_threads(),host_cpu_count=os.cpu_count()))
    dump(a.output_dir/'registration.json',registration)
    models=[];start=time.perf_counter()
    for name,architecture,size,epoch in (('global','efficientnet_b2',512,7),('local','convnext_tiny',384,15)):
        ck=torch.load(root/'model'/(name+'.pt'),map_location='cpu',weights_only=True);config=ck['config']
        if config['model']!=architecture or config['image_size']!=size or config.get('use_age',False) or ck['epoch']!=epoch:raise ValueError('Unexpected original R16 checkpoint')
        model=build_model(config,False);model.load_state_dict(ck['state_dict'],strict=True)
        models.append(model.cuda().eval().requires_grad_(False));del ck
    torch.cuda.synchronize();load_ms=1000*(time.perf_counter()-start)
    pipe=R08Pipeline(*models,'cuda')
    predict=pipe.predict_details if a.view=='original' else lambda path:predict_view(pipe,path,mirrored=True)
    sync=torch.cuda.synchronize;torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    def budget():
        if time.monotonic()-started>a.max_minutes*60:raise TimeoutError('Inference budget exceeded; partial predictions retained')
    cold,cold_ms=timed_call(predict,rows[0][1],sync)
    warmup=a.warmup if a.command=='benchmark' else 3
    for i in range(warmup):budget();predict(rows[i%len(rows)][1])
    sync();times=[]
    raw=a.output_dir/'predictions.csv';temporary=a.output_dir/'submission.csv.tmp'
    with raw.open('x',encoding='utf-8',newline='') as stream,temporary.open('x',encoding='utf-8',newline='') as saved:
        writer=csv.writer(stream);writer.writerow(['Image_name','Probability','Pred_Label','ms','fallback'])
        submit=csv.writer(saved);submit.writerow(['Image_name','Probability','Pred_Label'])
        for i,(name,path) in enumerate(rows):
            budget();details,ms=timed_call(predict,path,sync);value=validate_probability(details['probability']);times.append(ms)
            submit.writerow([name,format(value,'.17g'),int(value>=.5)])
            writer.writerow([name,format(value,'.17g'),int(value>=.5),ms,bool(details['fallback'])])
            if (i+1)%100==0:stream.flush();saved.flush();print(a.command,a.view,i+1,len(rows),flush=True)
    # Verify all input/model/source identities again before publishing completion.
    validate_bundle(root)
    if sha(root/'bundle.json')!=bundle_sha or sha(a.metadata)!=metadata_sha:raise ValueError('Bundle/metadata changed during run')
    with temporary.open(newline='',encoding='utf-8') as stream:restored=list(csv.DictReader(stream))
    if [r['Image_name'] for r in restored]!=[n for n,_ in rows]:raise ValueError('Prediction order differs')
    for row in restored:
        if int(row['Pred_Label'])!=int(validate_probability(row['Probability'])>=.5):raise ValueError('Serialized threshold differs')
    files=['registration.json','predictions.csv','results.json','status.json']
    if a.command=='predict':temporary.replace(a.output_dir/'submission.csv');files+=['submission.csv']
    else:temporary.unlink()
    result=dict(status='completed',command=a.command,view=a.view,threshold=.5,batch_size=1,
        weight_bytes=sum(i['bytes'] for i in bundle['weights'].values()),model_load_ms=load_ms,cold_prediction_ms=cold_ms,
        cold_probability=cold['probability'],warmup_calls=warmup,steady_state=summary(times,gpu,len(rows)==full_n),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        timing_scope='Original JPEG open/decode + EXIF/RGB + optional mirror + crop/pad + live CAM/ROI + forwards + synchronization',
        exclusions='Source/weight hashing, metadata scan, model load, first prediction, warmup and CSV writes excluded from steady state',
        cache_policy='OS caches uncontrolled; one process/view, no claim of cold filesystem cache',
        note='No official acceptance decision; subset diagnostics and non-T4 timing are not T4 acceptance',training=False,test_exported=a.command=='predict')
    dump(a.output_dir/'results.json',result);dump(a.output_dir/'status.json',dict(status='completed',command=a.command,view=a.view))
    dump(a.output_dir/'complete.json',dict(files={name:sha(a.output_dir/name) for name in files}))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
