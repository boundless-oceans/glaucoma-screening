"""Paired diagnostic of full JPEG background decode; never train or export."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from core import digest, verify_materials, write_json
from detail_crop import prepare_detail


def stats(values):
    import numpy as np
    return dict(mean_ms=float(np.mean(values)),p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('cpu','cuda'))
    p.add_argument('--training-project',type=Path,required=True)
    p.add_argument('--materials-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if not 1<=args.samples<=128 or not 1<=args.repeats<=5:p.error('Use 1..128 samples and 1..5 repeats')
    if args.output.exists():p.error('New output directory required')
    if args.mode=='cuda' and sys.platform!='linux':p.error('CUDA profiling requires Linux server')
    verify_materials(args.materials_root)
    asset=args.training_project/'assets/valid.jsonl'
    for name,sha in json.loads((args.training_project/'assets/complete.json').read_text()).items():
        if digest(args.training_project/'assets'/name)!=sha:raise ValueError('Asset identity mismatch')
    rows=[json.loads(s) for s in asset.read_text().splitlines()];rows=[r for r in rows if r['mirrored']]
    rows=random.Random(20261103).sample(rows,args.samples)
    index={}
    for path in (args.data_root/'valid').iterdir():
        if path.is_file():
            key=path.name.casefold()
            if key in index:raise ValueError('Ambiguous filename')
            index[key]=path.resolve()
    for row in rows:
        path=index[row['image_id'].casefold()]
        if not path.is_relative_to((args.data_root/'valid').resolve()):raise ValueError('Unsafe image path')
        row['path']=str(path)
    args.output.mkdir(parents=True)
    sys.path.insert(0,str(args.materials_root/'inference'))
    report=dict(mode=args.mode,count=len(rows),repeats=args.repeats,assets_sha256=digest(asset),samples=[r['image_id'] for r in rows])
    if args.mode=='cpu':
        from imaging import prepare_view
        measurements={'shared_decode_and_global_prepare':[],'full_jpeg_decode_crop_resize':[],'crop_hash':[]}
        for row in rows:
            for repeat in range(args.repeats):
                # Alternate CPU arm order to expose some cache-order effects.
                order=('shared','detail') if repeat%2==0 else ('detail','shared')
                for arm in order:
                    start=time.perf_counter()
                    if arm=='shared':
                        full,g,decoded=prepare_view(row['path'],True)
                        if list(decoded.size)!=row['decoded_size']:raise ValueError('Decoded size mismatch')
                        measurements['shared_decode_and_global_prepare'].append(1000*(time.perf_counter()-start))
                    else:
                        crop,_=prepare_detail(row['path'],row,mirrored=True)
                        measurements['full_jpeg_decode_crop_resize'].append(1000*(time.perf_counter()-start))
                        start=time.perf_counter();hashlib.sha256(crop.tobytes()).hexdigest()
                        measurements['crop_hash'].append(1000*(time.perf_counter()-start))
        import PIL,numpy,platform
        report.update(environment=dict(python=sys.version,pillow=PIL.__version__,numpy=numpy.__version__,platform=platform.platform()),
                      statistics={k:stats(v) for k,v in measurements.items()},
                      scope='CPU components only, cached fixed ROI, OS caches uncontrolled. Not epoch speed or GPU/T4 end-to-end latency.')
        if 'torch' in sys.modules:raise RuntimeError('CPU profile unexpectedly imported torch')
    else:
        import subprocess
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
        from run_stable import make_engine
        from stable_precision import install_locator_policy
        from validate_detail_live import BEST_SHA
        import torch
        best=args.training_project/'train/best.pt'
        if digest(best)!=BEST_SHA:raise ValueError('Wrong candidate checkpoint')
        trained=json.loads((args.training_project/'project.json').read_text())['source']
        for name in ('detail_crop.py','imaging.py','stable_precision.py'):
            if digest(Path(__file__).parent/name)!=trained[name]:raise ValueError('Candidate preprocessing changed')
        for name,sha in json.loads((args.training_project/'evaluate/complete.json').read_text()).items():
            if digest(args.training_project/'evaluate'/name)!=sha:raise ValueError('B1 reference changed')
        refs={(r['image_id'],r['view']):r for r in json.loads((args.training_project/'evaluate/best_b1.json').read_text())['predictions']}
        engine,env=make_engine(args.materials_root);ck=torch.load(best,map_location='cpu',weights_only=True)
        if ck['epoch']!=3 or ck['partial_epoch']:raise ValueError('Wrong epoch/partial checkpoint')
        engine.local_model.load_state_dict(ck['state_dict'],strict=True);install_locator_policy(engine,torch)
        from detail_adapter import predict_view
        from parallel_detail_adapter import predict_parallel
        from concurrent.futures import ThreadPoolExecutor
        pool=ThreadPoolExecutor(max_workers=1)
        predict_profile=lambda engine,path,mirrored: predict_parallel(engine,path,pool,mirrored)
        for _ in range(2):
            predict_view(engine,rows[0]['path'],True);predict_profile(engine,rows[0]['path'],True)
        measurements={};records=[];max_delta=0.
        for row in rows:
            for repeat in range(args.repeats):
                output={}
                for arm in (('plain','profile') if repeat%2==0 else ('profile','plain')):
                    started=time.perf_counter()
                    output[arm]=(predict_view if arm=='plain' else predict_profile)(engine,row['path'],True)
                    output[arm]['elapsed_ms']=1000*(time.perf_counter()-started)
                a,b=output['plain'],output['profile'];delta=abs(a['probability']-b['probability']);max_delta=max(max_delta,delta)
                if delta>1e-7 or a['crop_rgb_sha256']!=b['crop_rgb_sha256'] or a['roi_box']!=b['roi_box']:raise ValueError('Parallel decode changed prediction')
                if abs(a['probability']-refs[row['image_id'],'flip']['probability'])>1e-7:raise ValueError('Plain prediction differs from B1 candidate reference')
                measurements.setdefault('uninstrumented_total',[]).append(a['elapsed_ms'])
                measurements.setdefault('parallel_total',[]).append(b['elapsed_ms'])
                # Both totals include crop hash, CUDA sync, and background decode cleanup.
                records.append(dict(image_id=row['image_id'],repeat=repeat,plain=a,profile=b))
            print('profile',row['image_id'],flush=True)
        pool.shutdown(wait=True)
        write_json(args.output/'trace.json',records)
        report.update(environment=env,statistics={k:stats(v) for k,v in measurements.items()},max_probability_delta=max_delta,
                      scope='Serial B1 paired baseline vs one background full-JPEG decode worker; same per-image work, no inter-image batching. Not T4 acceptance.')
    report['source_sha256']={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}
    write_json(args.output/'report.json',report)
    write_json(args.output/'complete.json',{f.name:digest(f) for f in args.output.iterdir() if f.is_file()})
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':main()
