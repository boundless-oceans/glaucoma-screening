"""Isolated probability-policy experiment; historical R05 numerics stay frozen."""
import math
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def configure_runtime(training=False,hub_dir=None):
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=bool(training)
    if hub_dir is not None:torch.hub.set_dir(str(hub_dir))


def merge_training_pool(fit, valid, dev):
    for frame,split,role in [(fit,'train','fit'),(valid,'val','val'),(dev,'train','dev')]:
        if set(frame.official_split)!={split} or set(frame.role)!={role}:
            raise ValueError('Unexpected official split or role')
        if frame.image_id.isna().any() or frame.image_id.duplicated().any() or not frame.label.isin([0,1]).all():
            raise ValueError('Invalid IDs or binary labels')
    pool=pd.concat([fit,valid],ignore_index=True)
    if pool.image_id.duplicated().any():raise ValueError('Duplicate training IDs')
    for column in ['image_id','group_id','patient_id','sha256','pixel_sha256']:
        if column in pool and column in dev:
            a=set(pool[column].dropna().astype(str));b=set(dev[column].dropna().astype(str))
            if a&b:raise ValueError(f'Training/dev {column} overlap')
    pool['role']='r08_fit'
    return pool


def selection_key(metrics):
    values=tuple(float(metrics[k]) for k in ['auc_probability','ap_probability'])
    if not all(math.isfinite(v) and 0<=v<=1 for v in values):raise ValueError('Invalid selection metric')
    return values


def sigmoid(value):
    if not math.isfinite(value):raise ValueError('Invalid logit')
    return 1/(1+math.exp(-value)) if value>=0 else math.exp(value)/(1+math.exp(value))


def fuse_logits(global_logit,local_logit,fallback):
    if fallback:return sigmoid(global_logit)
    if local_logit is None or not math.isfinite(local_logit):raise ValueError('Invalid local logit')
    sigmoid(global_logit)
    return sigmoid(.5*(global_logit+local_logit))


def bounded_train_epoch(model,loader,optimizer,device,pos_weight,expired,scaler=None,scheduler=None,on_update=None):
    model.train();optimizer.zero_grad(set_to_none=True)
    stats=dict(complete=True,steps=0,updates=0,skipped_updates=0,class_count=[0,0],
               loss_sum=[0.,0.],logit_sum=[0.,0.],gradient_norm_samples=[],gradient_clip='none')
    amp=scaler is not None and scaler.is_enabled()
    weight=torch.tensor(pos_weight,device=device)
    for batch in loader:
        if expired():stats['complete']=False;break
        x=batch['image'].to(device,non_blocking=True);age=batch['age'].to(device,non_blocking=True)
        y=batch['label'].to(device,non_blocking=True)
        with torch.autocast(device.type,dtype=torch.float16,enabled=amp):
            logits=model(x,age)
            losses=F.binary_cross_entropy_with_logits(logits,y,pos_weight=weight,reduction='none')
            loss=losses.mean()
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite R08 training loss')
        if scaler is not None:
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
        else:loss.backward()
        if stats['steps']%100==0:
            norms=[p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None]
            norm=float(torch.stack(norms).norm())
            stats['gradient_norm_samples'].append(norm if math.isfinite(norm) else None)
        if scaler is not None:
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
            updated=scaler.get_scale()>=previous
        else:optimizer.step();updated=True
        optimizer.zero_grad(set_to_none=True)
        stats['steps']+=1;stats['updates']+=int(updated);stats['skipped_updates']+=int(not updated)
        if scheduler is not None and updated:scheduler.step()
        if on_update is not None and updated:on_update()
        for c in [0,1]:
            mask=y==c;stats['class_count'][c]+=int(mask.sum())
            stats['loss_sum'][c]+=float(losses.detach()[mask].float().sum())
            stats['logit_sum'][c]+=float(logits.detach()[mask].float().sum())
        if stats['steps']%100==0:print('TRAIN_STEP',stats['steps'],len(loader),'updates',stats['updates'],flush=True)
    stats['class_mean_weighted_loss']=[s/n if n else None for s,n in zip(stats['loss_sum'],stats['class_count'])]
    stats['class_mean_logit']=[s/n if n else None for s,n in zip(stats['logit_sum'],stats['class_count'])]
    return stats


@torch.inference_mode()
def predict_bounded(model,loader,device,expired):
    model.eval();rows=[]
    for batch in loader:
        if expired():return None
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            logits=model(batch['image'].to(device),batch['age'].to(device))
        values=logits.float();prob=torch.sigmoid(values)
        rows.extend(dict(image_id=i,probability=float(p),logit=float(z)) for i,p,z in zip(batch['image_id'],prob.cpu(),values.cpu()))
    return pd.DataFrame(rows)
