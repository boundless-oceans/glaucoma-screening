import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader,WeightedRandomSampler

from .data import FundusDataset,fit_age_stats
from .evaluate import evaluate_predictions,select_threshold
from .models import build_model
from .resume import restore_training_state
from .losses import positive_weight_from_labels


def train_epoch(model,loader,optimizer,criterion,device,scaler=None,scheduler=None,accumulation=1):
    if accumulation<1: raise ValueError('Positive gradient accumulation required')
    model.train();optimizer.zero_grad(set_to_none=True);total=0.;count=0
    amp=bool(scaler is not None and scaler.is_enabled())
    for step,batch in enumerate(loader):
        x=batch['image'].to(device,non_blocking=True);age=batch['age'].to(device,non_blocking=True)
        y=batch['label'].to(device,non_blocking=True)
        group_start=(step//accumulation)*accumulation
        group_size=min(accumulation,len(loader)-group_start)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            loss=criterion(model(x,age),y)
        if not bool(torch.isfinite(loss)): raise RuntimeError(f'Nonfinite loss at step {step}')
        total+=float(loss.detach())*len(y);count+=len(y)
        if scaler is not None: scaler.scale(loss/group_size).backward()
        else: (loss/group_size).backward()
        if (step+1)%accumulation==0 or step+1==len(loader):
            if scaler is not None:
                old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                stepped=scaler.get_scale()>=old_scale
            else: optimizer.step();stepped=True
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and stepped: scheduler.step()
        if (step+1)%100==0:
            print('TRAIN_STEP',step+1,len(loader),'loss',round(total/count,5),flush=True)
    return total/count


def _seed_worker(worker_id):
    seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)


@torch.inference_mode()
def predict_loader(model,loader,device,amp=True):
    model.eval();ids=[];probs=[]
    for batch in loader:
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp and device.type=='cuda'):
            logits=model(batch['image'].to(device,non_blocking=True),batch['age'].to(device,non_blocking=True))
        probs.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist());ids.extend(batch['image_id'])
    return pd.DataFrame({'image_id':ids,'probability':probs})


def _checkpoint(path,payload):
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def run_training(config):
    config=dict(config);seed=int(config.get('seed',20261015))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(config.get('cpu_threads',8)))
    torch.backends.cudnn.benchmark=True
    device=torch.device(config.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    fit=pd.read_csv(config['fit_manifest']);dev=pd.read_csv(config['dev_manifest'])
    if set(fit.official_split)!= {'train'} or set(dev.official_split)!= {'train'}:
        raise ValueError('Internal fit/dev must both originate from official train')
    if set(fit.image_id)&set(dev.image_id): raise ValueError('Fit/dev leakage')
    if config.get('smoke'):
        def small(df):
            return pd.concat([df[df.label==v].sample(n=min(n,len(df[df.label==v])),random_state=seed)
                              for v,n in [(0,112),(1,16)]]).reset_index(drop=True)
        fit,dev=small(fit),small(dev);config['epochs']=2;config['workers']=2
    config['age_stats']=fit_age_stats(fit)
    name=config.get('run_name','run')+'-'+time.strftime('%Y%m%d-%H%M%S')
    run=Path(config.get('run_root','runs'))/name;run.mkdir(parents=True,exist_ok=False)
    (run/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=True))
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    provenance=dict(torch=torch.__version__,torchvision=__import__('torchvision').__version__,
                    cuda=torch.version.cuda,device=str(device),seed=seed,source_sha256=sources,
                    split_sha256=hashlib.sha256(Path(config['fit_manifest']).read_bytes()+Path(config['dev_manifest']).read_bytes()).hexdigest(),
                    fit_n=len(fit),dev_n=len(dev),pretrained='torchvision ImageNet1K V1',
                    gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                    inference_precision=config.get('inference_precision','fp16'))
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    source_copy=run/'source/glaucoma';source_copy.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,source_copy/path.name)
    fit.to_csv(run/'fit_manifest_snapshot.csv',index=False)
    dev.to_csv(run/'dev_manifest_snapshot.csv',index=False)
    loader_args=dict(num_workers=int(config.get('workers',8)),pin_memory=device.type=='cuda',worker_init_fn=_seed_worker)
    if loader_args['num_workers']>0: loader_args.update(persistent_workers=True,prefetch_factor=2)
    batch_size=int(config.get('batch_size',32));generator=torch.Generator().manual_seed(seed)
    sampler=None
    if config.get('sampling')=='moderate':
        p=float(fit.label.mean());q=max(p,min(.25,4*p))
        weights=np.where(fit.label.to_numpy()==1,q/int(fit.label.sum()),(1-q)/int((fit.label==0).sum()))
        sampler=WeightedRandomSampler(weights,len(fit),replacement=True,generator=generator)
    train_loader=DataLoader(FundusDataset(fit,config['data_root'],config,True),batch_size=batch_size,
                            shuffle=sampler is None,sampler=sampler,generator=generator,**loader_args)
    dev_loader=DataLoader(FundusDataset(dev,config['data_root'],config,False),batch_size=batch_size*2,
                          shuffle=False,**loader_args)
    model=build_model(config,pretrained=not bool(config.get('resume_from'))).to(device)
    head_parameters=[p for n,p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':config.get('backbone_lr',1e-4)},
                                 {'params':head_parameters,'lr':config.get('head_lr',1e-3)}],weight_decay=config.get('weight_decay',1e-4))
    pos_weight=None
    if config.get('loss')=='weighted_bce':
        if sampler is not None: raise ValueError('Do not stack sampling and loss compensation')
        value=positive_weight_from_labels(fit.label.to_numpy(),config.get('positive_weight_policy','moderate'))
        pos_weight=torch.tensor(value,device=device)
        config['effective_positive_weight']=value;provenance['effective_positive_weight']=value
        (run/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=True))
        (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    accumulation=int(config.get('accumulation',1));epochs=int(config.get('epochs',20))
    steps_per_epoch=math.ceil(len(train_loader)/accumulation);total_steps=epochs*steps_per_epoch
    warmup=steps_per_epoch
    def lr_factor(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total_steps-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=(-1.,-1.);best_auc=-1.;stale=0;start=time.time();history=[];start_epoch=0
    if config.get('resume_from'):
        resume_path=Path(config['resume_from'])
        state=restore_training_state(model,optimizer,scheduler,scaler,resume_path,provenance['split_sha256'])
        for key in ['model','use_age','image_size','loss','positive_weight_policy','sampling','batch_size','accumulation',
                    'backbone_lr','head_lr','weight_decay','epochs','seed','selection_policy']:
            if state['config'].get(key)!=config.get(key):raise ValueError(f'Resume configuration changed: {key}')
        start_epoch=int(state['epoch'])
        if start_epoch>=epochs:raise ValueError('Checkpoint already reached requested epochs')
        origin=resume_path.parent
        for name in ['best.pt','best_probability.pt','metrics.json','dev_predictions.csv','threshold_candidate.json']:
            if (origin/name).exists():shutil.copy2(origin/name,run/name)
        if (origin/'history.csv').exists():
            old=pd.read_csv(origin/'history.csv');history=old[old.epoch<=start_epoch].to_dict('records')
        previous=json.loads((origin/'metrics.json').read_text())['fixed']
        best=(previous['balanced_accuracy'],previous['auc_probability'])
        if config.get('selection_policy')=='probability':best=best[::-1]
        best_auc=max(x['auc_probability'] for x in history) if history else previous['auc_probability']
        provenance.update(resume_from=str(resume_path),resume_epoch=start_epoch,
                          resume_stochastic_stream='DataLoader worker augmentation RNG restarts; not bitwise continuation')
        (run/'provenance.json').write_text(json.dumps(provenance,indent=2))
    for epoch in range(start_epoch,epochs):
        epoch_start=time.time();loss=train_epoch(model,train_loader,optimizer,criterion,device,scaler,scheduler,accumulation)
        pred=predict_loader(model,dev_loader,device);metrics=evaluate_predictions(dev,pred,.5)
        aligned=dev[['image_id','label']].merge(pred,on='image_id',validate='one_to_one')
        threshold=select_threshold(aligned.label.to_numpy(),aligned.probability.to_numpy())
        selected=evaluate_predictions(dev,pred,threshold['threshold'])
        score=(metrics['balanced_accuracy'],metrics['auc_probability'])
        if config.get('selection_policy')=='probability':score=(metrics['auc_probability'],metrics['balanced_accuracy'])
        entry=dict(epoch=epoch+1,loss=loss,auc_probability=metrics['auc_probability'],
                    ba_0_5=metrics['balanced_accuracy'],ba_selected=selected['balanced_accuracy'],
                    threshold_selected=threshold['threshold'],ap=metrics['ap_probability'],
                    seconds=time.time()-epoch_start,lr=optimizer.param_groups[0]['lr'])
        history.append(entry);pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        payload=dict(state_dict=model.state_dict(),config=config,epoch=epoch+1)
        if score>best:
            best=score;stale=0
            _checkpoint(run/'best.pt',payload)
            aligned.to_csv(run/'dev_predictions.csv',index=False,float_format='%.17g')
            (run/'metrics.json').write_text(json.dumps({'fixed':metrics,'selected':selected},indent=2,allow_nan=False))
            (run/'threshold_candidate.json').write_text(json.dumps(threshold,indent=2))
        else:stale+=1
        if metrics['auc_probability']>best_auc:
            best_auc=metrics['auc_probability'];_checkpoint(run/'best_probability.pt',payload)
        _checkpoint(run/'last.pt',{**payload,'optimizer':optimizer.state_dict(),
                                   'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                                   'split_sha256':provenance['split_sha256']})
        print('EPOCH',json.dumps(entry),flush=True)
        if stale>=int(config.get('patience',5)):
            print('EARLY_STOP',flush=True);break
        if time.time()-start>config.get('wallclock_seconds',10800):
            print('WALLCLOCK_LIMIT_CHECKPOINT_SAVED',flush=True);break
    (run/'threshold.json').write_text(json.dumps({'threshold':.5,'comparison':'>=','rule_status':'custom_threshold_unconfirmed'}))
    (run/'complete.json').write_text(json.dumps({'epochs':len(history),'seconds':time.time()-start,'best_score':best,
                                                'smoke':bool(config.get('smoke'))},indent=2))
    print('RUN_COMPLETE',str(run),flush=True)
    return run


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume',help='Own last.pt with optimizer and scheduler state')
    args=ap.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    if args.smoke:config.update(smoke=True,run_name='E00-smoke')
    if args.resume:config['resume_from']=str(Path(args.resume).resolve())
    run_training(config)


if __name__=='__main__':main()
