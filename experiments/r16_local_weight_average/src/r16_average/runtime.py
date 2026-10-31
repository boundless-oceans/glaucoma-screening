"""Explicit server-only build/check/evaluate stages. No tuning or training."""
import argparse
import copy
from pathlib import Path
import platform
import sys
import time
import traceback

from . import core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('build','check','evaluate'))
    parser.add_argument('--max-minutes',type=float,default=20,
                        help='Local loading/warmup/inference budget; excludes initial validation/global loading')
    args=parser.parse_args()
    if not 0<args.max_minutes<float('inf'):parser.error('Positive finite budget required')
    if sys.platform!='linux':parser.error('Model stages run on Linux server only; local CPU tests never load models')
    root=Path(__file__).resolve().parents[2]
    # An OS-held project lock prevents simultaneous stages. A stale file is harmless.
    import fcntl
    with (root/'.stage.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta,names,labels=core.validate_project(root)
        if args.stage!='build':core.verify_complete(root/'build',root,('model.pt','build.json','status.json'))
        if args.stage=='evaluate':core.verify_complete(root/'check',root,('report.json','status.json','original.csv','ema.csv','candidate.csv'))
        output=root/('evaluation' if args.stage=='evaluate' else args.stage)
        output.mkdir(exist_ok=False)
        core.dump(output/'status.json',dict(status='running',stage=args.stage))
        try:
            import torch
            import torchvision
            if sys.version_info[:2]!=(3,12) or (str(torch.__version__),str(torchvision.__version__))!=('2.10.0+cu128','0.25.0+cu128'):
                raise ValueError('Requires Python3.12, torch2.10.0+cu128, torchvision0.25.0+cu128')
            from glaucoma.r08 import configure_runtime
            configure_runtime(False)
            if args.stage=='build':build(root,meta,torch)
            else:
                if not torch.cuda.is_available():raise RuntimeError('CUDA server required for inference')
                infer(root,meta,names,labels,args,torch)
        except BaseException as error:
            core.dump(output/'status.json',dict(status='failed',stage=args.stage,error=str(error),traceback=traceback.format_exc()))
            raise
    return 0


def load_parent(meta,role,torch):
    ck=torch.load(meta['weights'][role]['path'],map_location='cpu',weights_only=True)
    config=ck['config']
    if role=='global':
        if config['model']!='efficientnet_b2' or config['image_size']!=512 or config.get('use_age',False) or ck['epoch']!=7:
            raise ValueError('Unexpected frozen global')
    elif ck['epoch']!={'original':15,'ema':11}[role]:raise ValueError('Unexpected local source epoch')
    if role=='ema' and ck.get('branch')!='ema':raise ValueError('Expected EMA branch checkpoint')
    return ck


def build(root,meta,torch):
    from glaucoma.models import build_model
    left=load_parent(meta,'original',torch);right=load_parent(meta,'ema',torch)
    core.validate_parent_configs(left['config'],right['config'])
    # Derive parameter identity from the actual architecture, not from dtype alone.
    model=build_model(left['config'],False)
    model.load_state_dict(left['state_dict'],strict=True)
    parameters=frozenset(dict(model.named_parameters()))
    model.load_state_dict(right['state_dict'],strict=True)
    del model
    averaged={k:v.detach().clone() for k,v in left['state_dict'].items()}
    with torch.no_grad():
        core.average_parameters_(averaged,left['state_dict'],right['state_dict'],parameters,
            equal=torch.equal,finite=lambda v:bool(torch.isfinite(v).all()))
    config=dict(model='convnext_tiny',image_size=384,use_age=False,seed=20261026,
                initializer_sha256=core.INITIALIZER,local_region=dict(teacher_sha256=core.WEIGHTS['global']))
    parent_info={k:copy.deepcopy(meta['weights'][k]) for k in ('original','ema')}
    payload=dict(schema='r16-local-average-v1',state_dict=averaged,config=config,
                 branch='fixed_50_50_parameter_average',policy=core.POLICY,parents=parent_info,
                 source_epochs=dict(original=15,ema=11))
    out=root/'build';temp=out/'model.pt.tmp';torch.save(payload,temp);temp.replace(out/'model.pt')
    restored=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
    if any(not torch.equal(v,restored['state_dict'][k]) for k,v in averaged.items()):raise ValueError('Saved average changed')
    model=build_model(config,False);model.load_state_dict(restored['state_dict'],strict=True)
    total=(out/'model.pt').stat().st_size+meta['weights']['global']['bytes']
    if total>500_000_000:raise ValueError('Two deployed weights exceed size budget')
    core.validate_project(root)
    report=dict(schema='r16-average-build-v1',policy=core.POLICY,parents=parent_info,
                candidate_sha256=core.sha(out/'model.pt'),candidate_bytes=(out/'model.pt').stat().st_size,
                deployed_weight_bytes=total,parameter_tensors=len(parameters),buffer_tensors=len(averaged)-len(parameters),
                parent_configs_verified=True,checkpoint_round_trip=True,training=False)
    core.dump(out/'build.json',report)
    core.dump(out/'status.json',dict(status='completed',stage='build'))
    core.complete(out,root,('model.pt','build.json','status.json'))
    print('BUILD_COMPLETED',report['candidate_sha256'],flush=True)


def load_candidate(root,meta,torch):
    ck=torch.load(root/'build/model.pt',map_location='cpu',weights_only=True)
    if ck.get('schema')!='r16-local-average-v1' or ck.get('policy')!=core.POLICY:
        raise ValueError('Candidate policy differs')
    if ck['parents']!={k:meta['weights'][k] for k in ('original','ema')}:
        raise ValueError('Candidate parents differ')
    if ck.get('branch')!='fixed_50_50_parameter_average' or 'optimizer' in ck or 'epoch' in ck:
        raise ValueError('Candidate identity differs')
    core.validate_parent_configs(ck['config'],ck['config'])
    return ck


def infer(root,meta,names,labels,args,torch):
    import csv
    import numpy as np
    import pandas as pd
    from PIL import __version__ as pillow
    from glaucoma.models import build_model
    from glaucoma.r08_export import R08Pipeline
    from r16_deploy.core import inputs,timed_call,summary
    from .metrics import metrics

    out=root/('evaluation' if args.stage=='evaluate' else 'check')
    # Parent references were measured with these dependencies. Stop on environment drift.
    if (np.__version__,pd.__version__,pillow)!=('2.5.4','3.0.6','12.3.0'):
        raise ValueError('Reference environment requires numpy2.5.4/pandas3.0.6/Pillow12.3.0')
    all_rows=inputs(Path(meta['data_root'])/'valid',Path(meta['data_root'])/'valid.csv',5849)
    if [n for n,_ in all_rows]!=names:raise ValueError('Official image list differs')
    indices=core.check_indices(names,labels) if args.stage=='check' else list(range(len(names)))
    roles=('original','ema','candidate') if args.stage=='check' else ('candidate',)
    device=torch.device('cuda')
    gck=load_parent(meta,'global',torch)
    global_model=build_model(gck['config'],False);global_model.load_state_dict(gck['state_dict'],strict=True)
    global_model=global_model.to(device).eval().requires_grad_(False);del gck
    references={r:core.reference_values(item['path'],names,item['column']) for r,item in meta['references'].items()}
    report=dict(stage=args.stage,rows=len(indices),policy=core.POLICY,
                candidate_sha256=core.sha(root/'build/model.pt'),runtime=dict(gpu=torch.cuda.get_device_name(0),
                python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,pandas=pd.__version__,
                pillow=pillow,cpu_threads=torch.get_num_threads()),arms={},official_acceptance=False,training=False,test_exported=False)
    started=time.monotonic()
    def budget():
        if time.monotonic()-started>args.max_minutes*60:raise TimeoutError('Stage budget exceeded; partial outputs retained')
    files=[]
    for role in roles:
        ck=load_candidate(root,meta,torch) if role=='candidate' else load_parent(meta,role,torch)
        local_model=build_model(ck['config'],False);local_model.load_state_dict(ck['state_dict'],strict=True)
        local_model=local_model.to(device).eval().requires_grad_(False);del ck
        pipe=R08Pipeline(global_model,local_model,device)
        for i in indices[:3]:budget();pipe.predict_details(all_rows[i][1])
        values=[];times=[];filename=role+'.csv';files.append(filename)
        with (out/filename).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['image_id','probability','global_logit','local_logit','fallback','ms'])
            for count,i in enumerate(indices,1):
                budget();details,ms=timed_call(pipe.predict_details,all_rows[i][1],torch.cuda.synchronize)
                value=core.probability(details['probability']);values.append(value);times.append(ms)
                writer.writerow([names[i],format(value,'.17g'),details['global_logit'],details['local_logit'],details['fallback'],ms])
                if count%100==0 or args.stage=='check' and count%10==0:
                    stream.flush();print(args.stage,role,count,len(indices),flush=True)
        arm=dict(metrics=metrics([labels[i] for i in indices],values),timing=summary(times,report['runtime']['gpu'],len(indices)==5849))
        if role in references:
            arm['reference_comparison']=core.compare_reference(values,[references[role][i] for i in indices])
            # Write evidence before failing; no complete marker for failed checks.
            if not arm['reference_comparison']['passed']:
                report['arms'][role]=arm;core.dump(out/'report.json',report)
                raise ValueError('Parent reference not reproduced: '+role)
        report['arms'][role]=arm
        del pipe,local_model;torch.cuda.empty_cache()
    if args.stage=='evaluate':
        report['reference_metrics']={k:metrics(labels,v) for k,v in references.items()}
        report['comparison']='Single fixed coefficient; existing full parent references, parent inference checked on fixed panel'
    core.validate_project(root)
    core.verify_complete(root/'build',root,('model.pt','build.json','status.json'))
    core.dump(out/'report.json',report)
    core.dump(out/'status.json',dict(status='completed',stage=args.stage))
    core.complete(out,root,files+['report.json','status.json'])
    print(args.stage.upper()+'_COMPLETED; no training or test export',flush=True)
