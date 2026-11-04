"""Pinned detail512 test export, gated by archived full validation; no training."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from core import digest, read_csv, resolve_images, verify_materials, write_json, write_submission, probability
from validate_detail_live import BEST_SHA

VALIDATION_SHA='1916aef70c538e96e2e46cd5499a98a9d87721b8c6550186f2e3063052303beb'
POLICY='parallel-nocopy-v1'
PREDICTOR_FILES=('core.py','run_stable.py','parallel_decode.py','parallel_detail_adapter.py','detail_crop.py','imaging.py','stable_precision.py')


def validate_tests(rows):
    if len(rows)!=5855:raise ValueError('Full official test required')
    seen=set()
    for row in rows:
        name=row['image_id'];relative=Path(row['relative_path'])
        if (row['label'] or row['role']!='test' or row['official_split']!='test'
            or relative.is_absolute() or '..' in relative.parts or len(relative.parts)!=2
            or relative.parts[0]!='test' or relative.name.casefold()!=name.casefold()
            or name.casefold() in seen):raise ValueError('Invalid unlabeled test identity/order')
        seen.add(name.casefold())


def identity():
    root=Path(__file__).resolve().parent
    return {p.name:digest(p) for p in sorted(root.glob('*.py'))}


def evidence(train,validation,materials):
    verify_materials(materials)
    for stage in ('check','evaluate'):
        d=validation/stage
        for name,sha in json.loads((d/'complete.json').read_text()).items():
            if digest(d/name)!=sha:raise ValueError('Validation output changed')
    report=json.loads((validation/'evaluate/report.json').read_text())
    if digest(validation/'evaluate/report.json')!=VALIDATION_SHA:raise ValueError('Wrong fixed full-validation report')
    if (not report['passed'] or report['count']!=5849 or report['identity']['best_sha256']!=BEST_SHA
        or report['identity']['execution_policy']!=POLICY):raise ValueError('Unvalidated candidate/policy')
    root=Path(__file__).resolve().parent
    for name in PREDICTOR_FILES:
        if digest(root/name)!=report['identity']['source'][name]:raise ValueError('Validated predictor changed: '+name)
    if digest(train/'train/best.pt')!=BEST_SHA:raise ValueError('Wrong detail512 weights')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','export'))
    p.add_argument('--project',required=True,type=Path)
    p.add_argument('--training-project',type=Path)
    p.add_argument('--validation-project',type=Path)
    p.add_argument('--materials-root',type=Path)
    p.add_argument('--data-root',type=Path)
    a=p.parse_args()
    if a.stage=='export' and sys.platform!='linux':p.error('Linux CUDA server required; no local model execution')
    if a.stage=='prepare':
        if any(v is None for v in (a.training_project,a.validation_project,a.materials_root,a.data_root)):p.error('prepare requires all four input roots')
        train=a.training_project.resolve();validation=a.validation_project.resolve();materials=a.materials_root.resolve()
        evidence(train,validation,materials)
        rows=read_csv(materials/'metadata/test_manifest.csv');validate_tests(rows)
        paths=resolve_images(a.data_root,rows)
        stats=[dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns) for path in paths]
        a.project.mkdir(parents=True,exist_ok=False)
        write_json(a.project/'project.json',dict(training=str(train),validation=str(validation),materials=str(materials),data=str(a.data_root.resolve()),source=identity(),images=stats,test_manifest_sha256=digest(materials/'metadata/test_manifest.csv'),policy=POLICY))
        print('Prepared file identities only; no torch/model loaded:',a.project,flush=True)
        return
    if any(v is not None for v in (a.training_project,a.validation_project,a.materials_root,a.data_root)):p.error('export uses prepare-fixed paths only')
    import fcntl,subprocess
    with (a.project/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads((a.project/'project.json').read_text())
        if identity()!=config['source'] or config['policy']!=POLICY:raise ValueError('Prepared source/policy changed')
        train=Path(config['training']);validation=Path(config['validation']);materials=Path(config['materials'])
        valid_report=evidence(train,validation,materials)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
        rows=read_csv(materials/'metadata/test_manifest.csv');validate_tests(rows)
        if digest(materials/'metadata/test_manifest.csv')!=config['test_manifest_sha256']:raise ValueError('Test order changed')
        paths=resolve_images(config['data'],rows)
        def check_images():
            for path,saved in zip(paths,config['images']):
                stat=path.stat()
                if str(path)!=saved['path'] or stat.st_size!=saved['bytes'] or stat.st_mtime_ns!=saved['mtime_ns']:raise ValueError('Test JPEG changed')
        if len(paths)!=len(config['images']):raise ValueError('Test count changed')
        check_images()
        out=a.project/'export';out.mkdir(exist_ok=False)
        pool=None
        try:
            from concurrent.futures import ThreadPoolExecutor
            from run_stable import make_engine
            from stable_precision import install_locator_policy
            import torch
            engine,env=make_engine(materials)
            from parallel_detail_adapter import predict_parallel
            ck=torch.load(train/'train/best.pt',map_location='cpu',weights_only=True)
            if ck['epoch']!=3 or ck['partial_epoch'] or ck['branch']!='detail512_candidate':raise ValueError('Ineligible candidate')
            engine.local_model.load_state_dict(ck['state_dict'],strict=True);install_locator_policy(engine,torch)
            pool=ThreadPoolExecutor(max_workers=1)
            def predict(path,mirrored):
                start=time.perf_counter();r=predict_parallel(engine,path,pool,mirrored)
                r['elapsed_ms']=1000*(time.perf_counter()-start)
                probability(r['probability'])
                return r
            # Replay all 74 fixed check records before touching test predictions.
            assets={(r['image_id'],r['mirrored']):r for r in (json.loads(s) for s in (train/'assets/valid.jsonl').read_text().splitlines())}
            refs=[json.loads(s) for s in (validation/'check/trace.jsonl').read_text().splitlines()]
            if len(refs)!=74:raise ValueError('Full 74-check required')
            for ref in refs:
                mirrored=ref['view']=='flip';row=assets[ref['image_id'],mirrored]
                result=predict(row['path'],mirrored)
                for key in ('roi_box','source_box','crop_rgb_sha256','global_logit','local_logit','probability'):
                    if result[key]!=ref[key]:raise ValueError('Fixed valid replay changed: '+ref['image_id']+'/'+key)
            write_json(out/'replay.json',dict(passed=True,count=74,scope='Exact prediction/ROI/crop replay; not T4 latency acceptance'))
            for _ in range(2):predict(paths[0],True)
            predictions=[];start=time.perf_counter()
            with (out/'trace.jsonl').open('x',encoding='utf-8',buffering=1) as stream:
                for i,(row,path) in enumerate(zip(rows,paths)):
                    result=predict(path,True);result.update(image_id=row['image_id'],view='flip')
                    stream.write(json.dumps(result,allow_nan=False)+'\n');predictions.append(result)
                    if (i+1)%128==0 or i+1==len(rows):print('export',i+1,'/',len(rows),flush=True)
            write_submission(out/'submission_detail512.csv',rows,predictions)
            check_images();evidence(train,validation,materials)
            if identity()!=config['source']:raise ValueError('Source changed during export')
            report=dict(stage='export',count=len(rows),best_sha256=BEST_SHA,policy=POLICY,source=config['source'],environment=env,validation_report_sha256=VALIDATION_SHA,validation_metrics=valid_report['metrics'],submission_sha256=digest(out/'submission_detail512.csv'),elapsed_seconds=time.perf_counter()-start,test_labels_available=False,latency_accepted=False,note='Score candidate only. Existing full-valid mean116.48ms/p95176.01ms on4090D; T4/100ms NOT accepted.')
            write_json(out/'report.json',report)
            write_json(out/'complete.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
            print(json.dumps(report,allow_nan=False),flush=True)
        except BaseException:
            (out/'failed.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
        finally:
            if pool is not None:pool.shutdown(wait=True)


if __name__=='__main__':main()
