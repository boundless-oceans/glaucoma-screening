"""Pinned epoch3 detail512 live CUDA acceptance; never trains or exports test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from core import digest, verify_materials, write_json, check_indices

BEST_SHA = '3172d1858e4382019d5abb705aad618c7c5e6a0c06d123e74bfa103932e30c52'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('check','evaluate'))
    p.add_argument('--training-project',type=Path,required=True)
    p.add_argument('--materials-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--parallel',action='store_true',help='Overlap full JPEG decode using one persistent CPU worker')
    args=p.parse_args()
    if sys.platform!='linux':p.error('Linux CUDA server required; local model execution disabled')
    import fcntl
    args.output_root.mkdir(parents=True,exist_ok=True)
    with (args.output_root/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        train=args.training_project.resolve();out=args.output_root/args.stage;out.mkdir(exist_ok=False)
        pool=None
        try:
            if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied')
            verify_materials(args.materials_root)
            best=train/'train/best.pt'
            if digest(best)!=BEST_SHA:raise ValueError('Wrong fixed epoch3 candidate')
            project=json.loads((train/'project.json').read_text())
            for name in ('detail_crop.py','imaging.py','stable_precision.py'):
                if digest(Path(__file__).parent/name)!=project['source'][name]:raise ValueError('Candidate source mismatch: '+name)
            for stage in ('assets','train','evaluate'):
                for name,sha in json.loads((train/stage/'complete.json').read_text()).items():
                    if digest(train/stage/name)!=sha:raise ValueError('Training artifact changed: '+name)
            records=[json.loads(s) for s in (train/'assets/valid.jsonl').read_text().splitlines()]
            references=json.loads((train/'evaluate/best_b1.json').read_text())['predictions']
            if len(records)!=11698 or len(references)!=11698:raise ValueError('Full dual-view valid required')
            lookup={(r['image_id'],r['mirrored']):r for r in records}
            refs={(r['image_id'],r['view']=='flip'):r for r in references}
            rows=[r for r in records if not r['mirrored']]
            if len(rows)!=5849 or len(lookup)!=11698 or set(refs)!=set(lookup):raise ValueError('Role/ID mismatch')
            source={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
            identity=dict(best_sha256=BEST_SHA,source=source,assets_sha256=digest(train/'assets/valid.jsonl'),reference_sha256=digest(train/'evaluate/best_b1.json'))
            identity['execution_policy']='parallel-nocopy-v1' if args.parallel else 'serial-v1'
            if args.stage=='evaluate':
                check=args.output_root/'check'
                for name,sha in json.loads((check/'complete.json').read_text()).items():
                    if digest(check/name)!=sha:raise ValueError('Check output changed')
                report=json.loads((check/'report.json').read_text())
                if not report['passed'] or report['identity']!=identity:raise ValueError('Live check identity changed')
            from run_stable import make_engine, load_metrics
            from stable_precision import install_locator_policy, IDS
            from detail_crop import prepare_detail
            import torch
            started=time.perf_counter();engine,env=make_engine(args.materials_root)
            ck=torch.load(best,map_location='cpu',weights_only=True)
            if ck['epoch']!=3 or ck['partial_epoch'] or ck['branch']!='detail512_candidate':raise ValueError('Ineligible checkpoint')
            engine.local_model.load_state_dict(ck['state_dict'],strict=True)
            install_locator_policy(engine,torch)
            from detail_adapter import predict_view
            if args.parallel:
                from concurrent.futures import ThreadPoolExecutor
                from parallel_detail_adapter import predict_parallel
                pool=ThreadPoolExecutor(max_workers=1)
                def predict_view(engine,path,mirrored):
                    begin=time.perf_counter()
                    result=predict_parallel(engine,path,pool,mirrored)
                    # Outer timer includes background image cleanup on return.
                    result['elapsed_ms']=1000*(time.perf_counter()-begin)
                    return result
            load_seconds=time.perf_counter()-started
            if args.stage=='check':
                selected={rows[i]['image_id'] for i in check_indices([dict(label=str(r['label'])) for r in rows])}|IDS
                work=[r for r in records if r['image_id'] in selected]
            else:
                work=[r for r in records if r['mirrored']]
            for mirrored in (False,True):
                for _ in range(2):predict_view(engine,rows[0]['path'],mirrored)
            predictions=[];max_delta=0.;max_global=0.;box_changes=0;crop_changes=0
            with (out/'trace.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
                for i,row in enumerate(work):
                    stat=Path(row['path']).stat()
                    if stat.st_size!=row['jpeg_bytes'] or stat.st_mtime_ns!=row['jpeg_mtime_ns']:raise ValueError('JPEG changed')
                    r=predict_view(engine,row['path'],row['mirrored']);r.update(image_id=row['image_id'],view='flip' if row['mirrored'] else 'original')
                    ref=refs[row['image_id'],row['mirrored']]
                    box_changes+=int(r['roi_box']!=row['roi_box'] or r['decoded_size']!=row['decoded_size'] or r['original_size']!=row['original_size'])
                    max_delta=max(max_delta,abs(r['probability']-ref['probability']))
                    max_global=max(max_global,abs(r['global_logit']-row['global_logit']))
                    if args.stage=='check':
                        crop,_=prepare_detail(row['path'],row,mirrored=row['mirrored'])
                        crop_changes+=int(hashlib.sha256(crop.tobytes()).hexdigest()!=r['crop_rgb_sha256'])
                    stream.write(json.dumps(r,allow_nan=False)+'\n');predictions.append(r)
                    if (i+1)%128==0 or i+1==len(work):print(args.stage,i+1,'/',len(work),flush=True)
            import numpy as np
            elapsed=[r['elapsed_ms'] for r in predictions]
            passed=box_changes==0 and crop_changes==0 and max_delta<=1e-7 and max_global<=1e-6
            metric=load_metrics();metrics={}
            for view in ('original','flip'):
                pp=[r for r in predictions if r['view']==view]
                if pp:metrics[view]=metric([lookup[r['image_id'],view=='flip']['label'] for r in pp],[r['probability'] for r in pp])
            report=dict(passed=passed,count=len(work),identity=identity,environment=env,checkpoint_load_seconds=load_seconds,
                max_probability_delta=max_delta,max_global_logit_delta=max_global,roi_changes=box_changes,crop_changes=crop_changes,metrics=metrics,
                latency_ms=dict(mean=float(np.mean(elapsed)),p50=float(np.percentile(elapsed,50)),p95=float(np.percentile(elapsed,95)),p99=float(np.percentile(elapsed,99)),max=max(elapsed),over_100ms=sum(v>100 for v in elapsed)),
                timing_scope=('Serial B1; parallel full JPEG decode enabled. ' if args.parallel else 'Serial B1; sequential decode. ')+'Draft shared decode + global + FP32 locator TF32-off + full JPEG decode/crop512 + local + synchronization; four warmups excluded. Diagnostic crop hash and per-image cleanup included; model loading and final pool shutdown excluded. 4090D only, NOT T4 acceptance.')
            write_json(out/'report.json',report)
            if not passed:raise ValueError('Real-time/cache mismatch; retain evidence, do not relax automatically')
            if digest(best)!=BEST_SHA or source!={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}:raise ValueError('Input changed')
            write_json(out/'complete.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
            print(json.dumps(report,allow_nan=False),flush=True)
        except BaseException:
            (out/'failed.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
        finally:
            if pool is not None:pool.shutdown(wait=True)


if __name__=='__main__':
    main()
