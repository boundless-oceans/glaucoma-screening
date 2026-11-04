"""Standalone Linux CUDA prediction/serial-B1 timing. Never trains/downloads."""
import argparse,csv,json,platform,subprocess,sys,time,traceback
from pathlib import Path
from contracts import MODES,verify
from io_contracts import inputs,select,summary,timed_call,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('predict','benchmark'))
    p.add_argument('--mode',required=True,choices=MODES)
    p.add_argument('--images',required=True,type=Path)
    p.add_argument('--metadata',required=True,type=Path,help='Official CSV with Image_name; labels ignored')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--limit',type=int,default=256,help='Benchmark: 0 means full coverage')
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--max-seconds',type=float,default=1800)
    p.add_argument('--diagnostic',action='store_true',help='Allow non-T4 benchmark; never marks official acceptance')
    p.add_argument('--reference',type=Path,help='Optional original/flip trace JSONL for strict probability/ROI replay')
    a=p.parse_args()
    if sys.platform!='linux':p.error('Linux CUDA server required; local GPU use disabled')
    if a.warmup<0 or a.limit<0 or not 0<a.max_seconds<=7200:p.error('Invalid warmup/limit/budget')
    root=Path(__file__).resolve().parent;manifest=verify(root)
    if a.mode=='detail-flip512' and not manifest['detail']:p.error('Bundle has no detail512 checkpoint')
    rows=inputs(a.images,a.metadata,5855 if a.stage=='predict' else None)
    total_count=len(rows);metadata_sha=sha(a.metadata)
    if a.stage=='benchmark':rows=select(rows,a.limit)
    image_stats={name:(path.stat().st_size,path.stat().st_mtime_ns) for name,path in rows}
    if a.mode=='detail-flip512' and any(path.suffix.casefold() not in ('.jpg','.jpeg') for _,path in rows):p.error('Validated detail512 decoder requires JPEG inputs')
    refs={}
    if a.reference:
        view='original' if a.mode=='captain-original' else 'flip'
        for line in a.reference.read_text().splitlines():
            record=json.loads(line)
            if record.get('view',view)!=view:continue
            if record['image_id'] in refs:raise ValueError('Duplicate reference ID/view')
            refs[record['image_id']]=record
        if not set(n for n,_ in rows)<=set(refs):raise ValueError('Reference does not cover selected inputs')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
    a.output.mkdir(parents=True,exist_ok=False);pool=None
    try:
        import torch,torchvision,numpy,PIL
        if sys.version_info[:2]!=(3,12) or torch.__version__!='2.10.0+cu128' or torchvision.__version__!='0.25.0+cu128':raise ValueError('Use verified Python3.12/torch2.10.0+cu128/torchvision0.25.0+cu128')
        if numpy.__version__!='2.5.4' or PIL.__version__!='12.3.0':raise ValueError('Use verified NumPy2.5.4/Pillow12.3.0')
        if not torch.cuda.is_available():raise ValueError('CUDA required')
        gpu=torch.cuda.get_device_name(0)
        if a.stage=='benchmark' and not summary([0],gpu,False)['hardware_is_t4'] and not a.diagnostic:raise ValueError('T4 benchmark required; use --diagnostic explicitly for other GPUs')
        sys.path[:0]=[str(root/'adapter'),str(root/'inference')]
        from predict import Inference
        started=time.perf_counter();engine=Inference(root/'inference','cuda')
        if a.mode=='captain-original':
            # Explicit historical policy; known near-tie/rounding cross-GPU sensitivity remains.
            torch.backends.cudnn.allow_tf32=True
            predict=engine.predict
        else:
            from stable_precision import install_locator_policy
            install_locator_policy(engine,torch)
            if a.mode=='stable-flip384':
                from adapter import predict_view
                predict=lambda path:predict_view(engine,path,True)
            else:
                from contracts import DETAIL_SHA
                from parallel_detail_adapter import predict_parallel
                from concurrent.futures import ThreadPoolExecutor
                ck=torch.load(root/'model/detail512.pt',map_location='cpu',weights_only=True)
                if ck['epoch']!=3 or ck['partial_epoch'] or ck['branch']!='detail512_candidate':raise ValueError('Wrong detail candidate')
                engine.local_model.load_state_dict(ck['state_dict'],strict=True)
                pool=ThreadPoolExecutor(max_workers=1)
                predict=lambda path:predict_parallel(engine,path,pool,True)
        sync=lambda:torch.cuda.synchronize(engine.device)
        sync();load_ms=1000*(time.perf_counter()-started)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        first,first_ms=timed_call(predict,rows[0][1],sync)
        def budget():
            if time.perf_counter()-start>a.max_seconds:raise TimeoutError('Inference budget exhausted; preserve partial output')
        for _ in range(a.warmup):budget();timed_call(predict,rows[0][1],sync)
        results=[];elapsed=[];delta=0.
        with (a.output/'predictions.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
            for i,(name,path) in enumerate(rows):
                if (path.stat().st_size,path.stat().st_mtime_ns)!=image_stats[name]:raise ValueError('Input image changed: '+name)
                budget();result,ms=timed_call(predict,path,sync);result.update(image_id=name,e2e_ms=ms)
                if refs:
                    ref=refs[name];delta=max(delta,abs(ref['probability']-result['probability']))
                    if abs(ref['probability']-result['probability'])>1e-7:raise ValueError('Reference probability changed: '+name)
                    for key in ('roi_box','crop_rgb_sha256'):
                        if key in ref and ref[key]!=result[key]:raise ValueError('Reference geometry/pixels changed: '+name)
                stream.write(json.dumps(result,allow_nan=False)+'\n');results.append(result);elapsed.append(ms)
                if (i+1)%128==0 or i+1==len(rows):print(a.stage,i+1,'/',len(rows),flush=True)
        if a.stage=='predict':
            with (a.output/'submission.csv').open('x',encoding='utf-8',newline='') as stream:
                writer=csv.writer(stream,lineterminator='\n');writer.writerow(('Image_name','Probability','Pred_Label'))
                writer.writerows((r['image_id'],format(r['probability'],'.17g'),int(r['probability']>=.5)) for r in results)
        env=dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,numpy=numpy.__version__,pillow=PIL.__version__,cuda=torch.version.cuda,gpu=gpu,cpu_threads=torch.get_num_threads(),platform=platform.platform(),driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
        if sha(a.metadata)!=metadata_sha:raise ValueError('Input metadata changed')
        for name,path in rows:
            if (path.stat().st_size,path.stat().st_mtime_ns)!=image_stats[name]:raise ValueError('Input image changed: '+name)
        report=dict(mode=a.mode,count=len(rows),metadata_sha256=metadata_sha,bundle_sha256=sha(root/'bundle.json'),model_bytes=manifest['weight_bytes'],environment=env,load_ms=load_ms,first_prediction_ms=first_ms,warmup=a.warmup,statistics=summary(elapsed,gpu,len(rows)==total_count),reference=dict(enabled=bool(refs),max_probability_delta=delta,sha256=sha(a.reference) if a.reference else None),official_acceptance=False,timing_scope='B1 sync -> open/decode/EXIF/RGB/view/global/locator/local/fusion/crop hash/worker cleanup -> sync. Excludes model load, source/weight hashes, metadata scan, CSV/trace I/O, final executor shutdown. OS caches uncontrolled; first call is not cold-disk acceptance.')
        import resource
        report['memory']=dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        report['precision']=dict(locator_tf32=a.mode=='captain-original',classifier='captain FP16 autocast',cpu_threads=8,cudnn_benchmark=False)
        verify(root)
        (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        files={p.name:sha(p) for p in a.output.iterdir() if p.is_file()}
        (a.output/'complete.json').write_text(json.dumps(files,indent=2)+'\n')
        print(json.dumps(report,allow_nan=False),flush=True)
    except BaseException:
        (a.output/'failed.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:
        if pool is not None:pool.shutdown(wait=True)

if __name__=='__main__':main()
