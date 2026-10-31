"""One-shot fresh official train/valid baseline; no old trajectory resumes."""
import gc,hashlib,json,math,os,random,shutil,subprocess,sys,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch,torchvision,yaml
from torch.utils.data import DataLoader
from glaucoma.models import build_model
from glaucoma.data import FundusDataset
from glaucoma.r08 import configure_runtime,selection_key,bounded_train_epoch,predict_bounded,fuse_logits
from glaucoma.r08_export import R08Pipeline
from glaucoma.r10 import load_22k_initializer
from glaucoma.r12_v3 import SharedGlobalDataset
from glaucoma.r13 import LocalizationInputs,localization_collate,roi_from_cam,align_anchors
from glaucoma.r13_runtime import require_official_dev_environment
from glaucoma.r16_protocol import validate_roles
from glaucoma.evaluate import evaluate_predictions
from glaucoma.train import _checkpoint,_seed_worker

OUT=ROOT/'reports/R16-20261026-official';SF=ROOT/'configs/r16_registration_20261026.json'
SET=json.loads(SF.read_text());DEVICE=torch.device('cuda')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def state(stage,**kw):
    v=dict(stage=stage,time_beijing=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
    dump(OUT/'status.json',v);print('R16_STATUS',json.dumps(v),flush=True)
def seed():
    value=SET['seed'];random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
def free():return shutil.disk_usage(ROOT).free
def optimizer(model):
    return torch.optim.AdamW([dict(params=model.backbone.parameters(),lr=SET['backbone_lr']),
        dict(params=model.head.parameters(),lr=SET['head_lr'])],weight_decay=SET['weight_decay'])
def fused(pred,assets):
    m=pred.merge(assets[['image_id','global_logit','fallback']],on='image_id',validate='one_to_one')
    if len(m)!=len(assets):raise ValueError('Fused valid coverage mismatch')
    return pd.DataFrame(dict(image_id=m.image_id,probability=[fuse_logits(g,l,bool(f)) for g,l,f in zip(m.global_logit,m.logit,m.fallback)]))

def prepare():
    require_official_dev_environment(torch.__version__,torchvision.__version__)
    if SET['t4_limit_ms']!=100 or SET['test_used'] or SET['old_competition_initializers_allowed']:
        raise ValueError('Competition constraints changed')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    if free()<2_000_000_000:raise RuntimeError('Need 2GB startup reserve; no deletion allowed')
    expected=json.loads((OUT/'upload_manifest.json').read_text())
    for relative,digest in expected.items():
        if sha(ROOT/relative)!=digest:raise ValueError('Source/registration mismatch: '+relative)
    for key in ['global','local']:
        if sha(ROOT/SET[key+'_initializer'])!=SET[key+'_initializer_sha256']:raise ValueError('Generic initializer changed')
    source=ROOT/'data/manifest_versions/initial-v1'
    fit=pd.read_csv(source/'train.csv');val=pd.read_csv(source/'val.csv')
    fit['role']='r16_official_train';val['role']='r16_official_valid'
    counts=validate_roles(fit.to_dict('records'),val.to_dict('records'))
    # Compare IDs/labels/paths with the official CSVs, without scanning images.
    official_hashes={}
    for frame,folder in [(fit,'train'),(val,'valid')]:
        csv=Path(SET['data_root'])/folder/(folder+'.csv')
        if not csv.is_file():csv=Path(SET['data_root'])/(folder+'.csv')
        raw=pd.read_csv(csv);official_hashes[folder]=sha(csv)
        def normalized(x):return str(x).rsplit('.',1)[0].lower()
        a={normalized(r.image_id):int(r.label) for r in frame.itertuples()}
        b={normalized(r.Image_name):int(r.True_Label) for r in raw.itertuples()}
        if len(raw)!=len(frame) or a!=b:raise ValueError('Original official CSV mismatch: '+folder)
        if any(normalized(Path(r.relative_path).name)!=normalized(r.image_id) for r in frame.itertuples()):
            raise ValueError('Manifest filename/ID mismatch')
        for index in [0,len(frame)//2,len(frame)-1]:
            if not (Path(SET['data_root'])/frame.iloc[index].relative_path).is_file():raise FileNotFoundError('Required sample path missing')
    fit.to_csv(OUT/'official_train.csv',index=False);val.to_csv(OUT/'official_valid.csv',index=False)
    registration=dict(**counts,train_sha256=sha(OUT/'official_train.csv'),valid_sha256=sha(OUT/'official_valid.csv'),
        source_train_sha256=sha(source/'train.csv'),source_valid_sha256=sha(source/'val.csv'),official_csv_sha256=official_hashes,
        data_root=SET['data_root'],valid_gradient_or_statistics_fit=False,test_used=False,old_roles_unchanged=True,
        caveat='Official valid is model selection, historically researched; not a new blind test')
    dump(OUT/'split_registration.json',registration)
    return fit,val

def train(kind,fit,val,assets=None,teacher=None):
    if free()<1_000_000_000:raise RuntimeError('Need 1GB phase reserve; no deletion allowed')
    local=kind=='local';configure_runtime(True);seed()
    run=ROOT/'runs'/(('E33' if local else 'E32')+'-r16-official-'+kind+'-'+time.strftime('%Y%m%d-%H%M%S'))
    run.mkdir(exist_ok=False);epochs=SET[kind+'_epochs'];budget=SET[kind+'_seconds']
    config=dict(model='convnext_tiny' if local else 'efficientnet_b2',image_size=384 if local else 512,
        use_age=False,augmentation=True,fundus_crop=True,cache_dir=None,data_root=SET['data_root'],batch_size=32,
        seed=SET['seed'],epochs=epochs,round='R16',run_name=run.name,loss='weighted_bce',
        effective_positive_weight=(len(fit)-int(fit.label.sum()))/int(fit.label.sum()),
        backbone_lr=SET['backbone_lr'],head_lr=SET['head_lr'],weight_decay=SET['weight_decay'],wallclock_seconds=budget,
        selection=SET['selection'],jpeg_geometry='shared768',ema=False,initializer_sha256=SET[kind+'_initializer_sha256'])
    if local:config['local_region']=dict(boxes_csv=str(OUT/'roi_boxes.csv'),decode_size=768,teacher_sha256=sha(teacher),version='r16-official-train-only')
    (run/'config.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    fit.to_csv(run/'fit_manifest_snapshot.csv',index=False);val.to_csv(run/'val_manifest_snapshot.csv',index=False)
    source=run/'source';source.mkdir();shutil.copytree(ROOT/'src/glaucoma',source/'glaucoma',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,source/Path(__file__).name)
    model=build_model(config,False)
    if local:
        initialization=load_22k_initializer(model,ROOT/SET['local_initializer'],SET['local_initializer_sha256'])
        pretrained=json.loads((ROOT/'pretrained/convnext22k_r10/provenance.json').read_text())
    else:
        initial=torch.load(ROOT/SET['global_initializer'],map_location='cpu',weights_only=True)
        if tuple(initial['classifier.1.weight'].shape)!=(1000,1408):raise ValueError('Expected generic 1000-class B2 initializer')
        model.backbone.load_state_dict({k:v for k,v in initial.items() if not k.startswith('classifier.')},strict=True)
        del initial
        initialization=dict(source_sha256=SET['global_initializer_sha256'],discarded_upstream_classifier=True)
        pretrained=dict(name='torchvision EfficientNet-B2 ImageNet1K V1',domain='generic ImageNet1K',code_license='BSD-3-Clause',
            url='https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth',sha256=SET['global_initializer_sha256'])
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    dump(run/'provenance.json',dict(torch=torch.__version__,torchvision=torchvision.__version__,gpu=torch.cuda.get_device_name(),
        seed=SET['seed'],settings_sha256=sha(SF),split_registration_sha256=sha(OUT/'split_registration.json'),
        fit_sha256=sha(run/'fit_manifest_snapshot.csv'),val_sha256=sha(run/'val_manifest_snapshot.csv'),
        source_sha256={str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},initializer=initialization,pretrained=pretrained,
        initializer_is_competition_checkpoint=False,medical_teacher=False,test_used=False,valid_fit=False,
        fit_n=len(fit),fit_positive=int(fit.label.sum()),val_n=len(val),val_positive=int(val.label.sum()),
        teacher_sha256=sha(teacher) if teacher else None,roi_sha256=sha(OUT/'roi_boxes.csv') if local else None,
        init_stem_sha256=hashlib.sha256(initial['backbone.features.0.0.weight' if not local else 'backbone.features.0.0.weight'].numpy().tobytes()).hexdigest()))
    model=model.to(DEVICE)
    train_data=FundusDataset(fit,SET['data_root'],config,True) if local else SharedGlobalDataset(fit,SET['data_root'],True)
    valid_data=FundusDataset(val,SET['data_root'],config,False) if local else SharedGlobalDataset(val,SET['data_root'],False)
    indices=np.flatnonzero(fit.label.to_numpy()==1)[:2].tolist()+np.flatnonzero(fit.label.to_numpy()==0)[:30].tolist()
    batch=torch.utils.data.default_collate([train_data[i] for i in indices]);before=model.head[-1].weight.detach().clone()
    opt=optimizer(model);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
    probe=bounded_train_epoch(model,[batch]*4,opt,DEVICE,config['effective_positive_weight'],lambda:False,scaler)
    probe.update(discarded=True,weight_changed=not torch.equal(before,model.head[-1].weight));dump(run/'engineering_probe.json',probe)
    if not probe['updates'] or not probe['weight_changed']:raise RuntimeError('Real GPU update probe failed')
    model.load_state_dict(initial)
    if any(not torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items()):raise RuntimeError('Probe reset mismatch')
    dump(run/'probe_reset.json',dict(all_parameters_and_buffers_restored=True,optimizer_scaler_rng_recreated=True))
    del initial,batch,before,opt,scaler;seed();opt=optimizer(model)
    args=dict(num_workers=8,pin_memory=True,persistent_workers=True,prefetch_factor=2,worker_init_fn=_seed_worker)
    a=DataLoader(train_data,batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(SET['seed']),**args)
    b=DataLoader(valid_data,batch_size=32 if local else 64,shuffle=False,generator=torch.Generator().manual_seed(SET['seed']+1),**args)
    steps=len(a);total=epochs*steps
    factor=lambda s:(s+1)/steps if s<steps else .5*(1+math.cos(math.pi*min(1.,(s-steps)/(total-steps))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,factor);scaler=torch.amp.GradScaler('cuda',init_scale=8192.)
    history=[];best=(-1.,-1.);updates=0;stale=0;stop='epoch_limit';start=time.monotonic()
    expired=lambda:time.monotonic()-start>=budget
    for epoch in range(1,epochs+1):
        state('training',kind=kind,run=str(run),epoch=epoch,updates=updates)
        stats=bounded_train_epoch(model,a,opt,DEVICE,config['effective_positive_weight'],expired,scaler,scheduler);updates+=stats['updates']
        dump(run/f'train_epoch_{epoch:02d}.json',stats)
        payload=dict(state_dict=model.state_dict(),config=config,epoch=epoch,updates=updates,
            optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict())
        if not stats['complete'] or expired():
            payload['partial_epoch']=True;_checkpoint(run/'partial.pt',payload);stop='training_budget';break
        if stats['class_count']!=[49739,1657]:raise ValueError('Official train coverage changed')
        pred=predict_bounded(model,b,DEVICE,expired)
        if pred is None:_checkpoint(run/'last.pt',payload);stop='validation_budget';break
        combined=fused(pred,assets) if local else pred
        metrics=evaluate_predictions(val,combined);pred.to_csv(run/f'valid_raw_{epoch:02d}.csv',index=False)
        if local:combined.to_csv(run/f'valid_fused_{epoch:02d}.csv',index=False)
        history.append(dict(epoch=epoch,updates=updates,seconds=time.monotonic()-start,**metrics))
        key=selection_key(metrics)
        if key>best:
            best=key;stale=0;_checkpoint(run/'best.pt',dict(state_dict=model.state_dict(),config=config,epoch=epoch,updates=updates,branch='raw'))
            dump(run/'metrics.json',dict(epoch=epoch,metrics=metrics));combined.to_csv(run/'valid_predictions.csv',index=False)
        else:stale+=1
        _checkpoint(run/'last.pt',payload);pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        state('epoch_completed',kind=kind,run=str(run),epoch=epoch,updates=updates,auc=metrics['auc_probability'],
            ap=metrics['ap_probability'],seconds=time.monotonic()-start,free_bytes=free())
        if stale>=SET['patience']:stop='auc_patience';break
        if free()<800_000_000:stop='disk_reserve';break
    if not (run/'best.pt').exists():raise RuntimeError('No complete epoch with full validation; no automatic restart')
    result=dict(kind=kind,run=str(run),completed_epochs=len(history),updates=updates,seconds=time.monotonic()-start,stop=stop,
        best=json.loads((run/'metrics.json').read_text()),best_sha256=sha(run/'best.pt'))
    dump(run/'completion.json',result);dump(OUT/(kind+'_result.json'),result)
    del model,opt,scaler,scheduler,a,b,payload,train_data,valid_data;gc.collect();torch.cuda.empty_cache()
    return result

@torch.inference_mode()
def make_assets(result,fit,val):
    configure_runtime(False);run=Path(result['run']);ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
    model=build_model(ck['config'],False);model.load_state_dict(ck['state_dict']);model=model.to(DEVICE).eval().requires_grad_(False)
    start=time.monotonic();frames=[];summary={}
    for role,frame,batch_size in [('train',fit,32),('valid',val,1)]:
        state('roi_assets',role=role,count=len(frame),teacher=str(run),batch_size=batch_size)
        data=LocalizationInputs(frame,SET['data_root'],for_training=role=='train')
        loader=DataLoader(data,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True,prefetch_factor=2,collate_fn=localization_collate)
        rows=[];next_report=2048
        for batch in loader:
            if time.monotonic()-start>SET['assets_seconds']:raise RuntimeError('ROI assets budget exceeded')
            with torch.autocast('cuda',dtype=torch.float16):
                features=model.backbone.features(batch['image'].to(DEVICE,non_blocking=True))
                logits=model.head(torch.flatten(model.backbone.avgpool(features),1)).squeeze(1)
                cam=(features.float()*model.head[-1].weight[0].float()[None,:,None,None]).sum(1)
            if not bool(torch.isfinite(logits).all()):raise ValueError('Nonfinite fresh global output')
            for i,name in enumerate(batch['image_id']):
                roi=roi_from_cam(cam[i].cpu().numpy(),batch['valid_mask'][i],batch['geometry'][i]);x0,y0,x1,y1=roi['source_box']
                rows.append(dict(image_id=name,global_logit=float(logits[i]),x0=x0,y0=y0,x1=x1,y1=y1,
                    fallback=bool(roi['fallback']),concentration=roi['concentration']))
            if len(rows)>=next_report:print('R16_ROI',role,len(rows),len(frame),flush=True);next_report+=2048
        asset=pd.DataFrame(rows);align_anchors(frame,asset);asset.to_csv(OUT/(role+'_assets.csv'),index=False)
        frames.append(asset);summary[role]=dict(n=len(asset),fallback=int(asset.fallback.sum()))
    pd.concat(frames,ignore_index=True).to_csv(OUT/'roi_boxes.csv',index=False)
    gm=evaluate_predictions(val,pd.DataFrame(dict(image_id=frames[1].image_id,probability=torch.sigmoid(torch.tensor(frames[1].global_logit.to_numpy())).numpy())))
    dump(OUT/'assets_registration.json',dict(teacher_run=str(run),teacher_sha256=sha(run/'best.pt'),teacher_train_role='r16_official_train',
        valid_used_for_teacher_gradient_or_stats=False,roi_sha256=sha(OUT/'roi_boxes.csv'),counts=summary,global_valid_b1_metrics=gm,
        seconds=time.monotonic()-start,test_used=False))
    del model,loader;gc.collect();torch.cuda.empty_cache()
    return frames[1]

@torch.inference_mode()
def final_b1(global_result,local_result,val):
    configure_runtime(False);models=[]
    for result in [global_result,local_result]:
        ck=torch.load(Path(result['run'])/'best.pt',map_location='cpu',weights_only=True)
        model=build_model(ck['config'],False);model.load_state_dict(ck['state_dict']);models.append(model.to(DEVICE).eval())
    pipe=R08Pipeline(*models,DEVICE)
    for row in val.head(5).itertuples():pipe.predict_details(Path(SET['data_root'])/row.relative_path)
    rows=[];times=[];start=time.monotonic()
    for i,row in enumerate(val.itertuples()):
        if time.monotonic()-start>SET['final_b1_seconds']:raise RuntimeError('Full original valid B1 budget exceeded')
        torch.cuda.synchronize();t=time.perf_counter();details=pipe.predict_details(Path(SET['data_root'])/row.relative_path)
        torch.cuda.synchronize();times.append(1000*(time.perf_counter()-t));rows.append(dict(image_id=row.image_id,**details))
        if (i+1)%1000==0:print('R16_FINAL_B1',i+1,len(val),flush=True)
    pred=pd.DataFrame(rows);pred.to_csv(OUT/'official_valid_original_b1.csv',index=False)
    metrics=evaluate_predictions(val,pred)
    weights=sum((Path(r['run'])/'best.pt').stat().st_size for r in [global_result,local_result])
    if weights+SF.stat().st_size>SET['inference_limit_bytes']:raise ValueError('Inference size exceeds500MB')
    result=dict(metrics=metrics,mean_ms=float(np.mean(times)),p95_ms=float(np.quantile(times,.95)),max_ms=max(times),
        timing_scope='4090 full official valid diagnosis, NOT T4 acceptance',t4_accepted=False,
        inference_weight_bytes=weights,threshold=.5,test_used=False,test_exported=False,
        global_checkpoint_sha256=global_result['best_sha256'],local_checkpoint_sha256=local_result['best_sha256'],
        valid_gradient_or_statistics_fit=False,selection_caveat='Official valid selected epochs; historically researched, not blind test')
    dump(OUT/'official_valid_b1_metrics.json',result);del pipe,models,model;gc.collect();torch.cuda.empty_cache()
    return result

def archive(results):
    target=ROOT/'artifacts/experiments/R16-20261026-official';target.mkdir(parents=True,exist_ok=False)
    files=[]
    for folder in [OUT]+[Path(r['run']) for r in results]:
        files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.yaml','.py','.md','.log') and '__pycache__' not in p.parts)
    files.extend([SF,ROOT/'tools/run_round_sixteen.py',ROOT/'tools/run_r16_workflow.py',ROOT/'tests/test_r16_protocol.py'])
    files=sorted(set(files));manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dump(target/'manifest.json',manifest)
    with zipfile.ZipFile(target/'R16-evidence.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
        z.write(target/'manifest.json','manifest.json')
    result=dict(zip_sha256=sha(target/'R16-evidence.zip'),zip_bytes=(target/'R16-evidence.zip').stat().st_size,members=len(files)+1,
        checkpoints=[dict(run=r['run'],sha256=r['best_sha256'],bytes=(Path(r['run'])/'best.pt').stat().st_size) for r in results],
        local_independent_verification_pending=True)
    dump(target/'archive_summary.json',result);dump(OUT/'archive_summary.json',result)

def main():
    with (OUT/'registration.json').open('x') as f:json.dump(dict(pid=os.getpid(),source_sha256=sha(__file__),settings_sha256=sha(SF),max_trajectories=2),f)
    state('preflight');fit,val=prepare();global_result=train('global',fit,val)
    assets=make_assets(global_result,fit,val);local_result=train('local',fit,val,assets,Path(global_result['run'])/'best.pt')
    state('final_official_valid_original_b1');evaluation=final_b1(global_result,local_result,val)
    dump(OUT/'results.json',dict(global_result=global_result,local_result=local_result,evaluation=evaluation))
    state('archiving');archive([global_result,local_result]);state('completed_pending_local_verification')

if __name__=='__main__':
    try:main()
    except Exception:
        dump(OUT/'failure.json',dict(traceback=traceback.format_exc()));state('failed');raise
