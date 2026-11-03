"""Independent high-detail pilot; server only. No submission generation."""
import argparse
import contextlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from core import HERE, digest, read_csv, verify_materials
from detail_crop import prepare_detail

TRAIN_SHA = '8a532cedf9e7944b8e6ee2c8581877a82770e0235a1e0642af2c8b9a7d385515'
ARMS = ('low384', 'low512', 'detail512')


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def identity():
    return {p.name: digest(p) for p in HERE.glob('*.py')}


def require(project, stage):
    marker = json.loads((project/stage/'complete.json').read_text())
    for name, sha in marker.items():
        if digest(project/stage/name) != sha:
            raise ValueError('Prior stage changed: '+name)


def complete(output):
    atomic_json(output/'complete.json', {p.name: digest(p) for p in output.iterdir() if p.is_file()})


def loader(dataset, batch, workers, shuffle=False, seed=20261103):
    import torch
    return torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed), num_workers=workers,
        pin_memory=True, persistent_workers=workers>0)


class Crops:
    def __init__(self, records, arm, training=False):
        self.records, self.arm, self.training, self.epoch = records, arm, training, 0
    def __len__(self):
        return len(self.records)
    def __getitem__(self, index):
        from PIL import Image, ImageOps, ImageEnhance
        from glaucoma.local_region import prepare_shared, image_tensor
        import torch
        row = self.records[index]
        if self.arm == 'detail512':
            image, _ = prepare_detail(row['path'], row, mirrored=row['mirrored'])
        else:
            _, _, decoded = prepare_shared(row['path'])
            if row['mirrored']:
                decoded = ImageOps.mirror(decoded)
            if list(decoded.size) != row['decoded_size']:
                raise ValueError('Asset JPEG size changed')
            image = ImageOps.pad(decoded.crop(row['roi_box']), (384,384) if self.arm=='low384' else (512,512),
                                 method=Image.Resampling.BICUBIC, color=(0,0,0))
        if self.training:
            rng = random.Random(20261103+self.epoch*1000003+index)
            if rng.random()<.5:
                image = ImageOps.mirror(image)
            from PIL import Image
            image = image.rotate(rng.uniform(-15,15), resample=Image.Resampling.BICUBIC, fillcolor=(0,0,0))
            for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                image = enhancer(image).enhance(rng.uniform(.9,1.1))
        return image_tensor(image), torch.tensor(float(row['label'])), index


def train_step(model, batch, optimizer, scaler, weight):
    import torch
    images, labels, _ = batch
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        logits = model(images.cuda(non_blocking=True))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.cuda(non_blocking=True), pos_weight=weight)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite training loss')
    old = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=False)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach()), scaler.get_scale()>=old


def save_model(path, model, config, epoch, optimizer=None, scaler=None, partial=False):
    import torch
    payload = dict(state_dict=model.state_dict(), config=config, epoch=epoch,
                   partial_epoch=partial, branch='detail512_candidate', optimizer=optimizer.state_dict() if optimizer else None,
                   scaler=scaler.state_dict() if scaler else None)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def evaluate(model, records, arm, output, prefix, batch, workers, deadline):
    import torch
    from glaucoma.models import build_model
    from run_stable import load_metrics
    predictions = []
    model.eval()
    with torch.inference_mode():
        for images, _, ids in loader(Crops(records, arm), batch, workers):
            if time.monotonic()>deadline:
                raise TimeoutError('Budget exhausted during validation; incomplete validation is ineligible')
            with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False):
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(images.cuda(non_blocking=True)).float().cpu().tolist()
            for index, local in zip(ids.tolist(), logits):
                r = records[index]
                z = r['global_logit'] if r['fallback'] else .5*(r['global_logit']+local)
                p = 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
                predictions.append(dict(image_id=r['image_id'], view='flip' if r['mirrored'] else 'original', probability=p, global_logit=r['global_logit'], local_logit=local))
    metric = load_metrics()
    result = {}
    for mirrored in (False,True):
        rows = [r for r in records if r['mirrored']==mirrored]
        pp = [p for p in predictions if p['view']==('flip' if mirrored else 'original')]
        if [r['image_id'] for r in rows] != [r['image_id'] for r in pp]:
            raise ValueError('Validation alignment changed')
        result['flip' if mirrored else 'original'] = metric([r['label'] for r in rows], [p['probability'] for p in pp])
    atomic_json(output/(prefix+'.json'), dict(metrics=result, predictions=predictions,
        timing_scope='Cached frozen global/ROI; not live end-to-end inference acceptance'))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare','assets','benchmark','smoke','train','evaluate'))
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--materials-root', type=Path)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--max-hours', type=float, default=3.)
    p.add_argument('--order', nargs=3, choices=ARMS, default=list(ARMS))
    args = p.parse_args()
    if args.max_hours<=0 or not math.isfinite(args.max_hours):
        p.error('Positive finite budget required')
    project = args.project.resolve()
    if args.stage=='prepare':
        if args.materials_root is None or args.data_root is None:
            p.error('prepare requires materials and data roots')
        mat = args.materials_root.resolve()
        verify_materials(mat)
        if digest(mat/'metadata/official_train.csv')!=TRAIN_SHA:
            raise ValueError('Official train manifest changed')
        project.mkdir(parents=True, exist_ok=False)
        atomic_json(project/'project.json', dict(materials=str(mat), data_root=str(args.data_root.resolve()),
            source=identity(), budget_hours=args.max_hours, batch_size=16, workers=4, epochs=3, backbone_lr=3e-6, head_lr=3e-5,
            pos_weight=49739/1657, source_checkpoint='R27 raw epoch12', recipe='detail512-pilot-v1'))
        print('Prepared file identities only:', project)
        return 0
    if sys.platform!='linux':
        p.error('Model stages require Linux GPU server; local training disabled')
    import fcntl
    with (project/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=json.loads((project/'project.json').read_text())
        if cfg['source']!=identity():
            raise ValueError('Prepared source changed')
        mat=Path(cfg['materials']);verify_materials(mat)
        if digest(mat/'metadata/official_train.csv')!=TRAIN_SHA:
            raise ValueError('Train manifest changed')
        out=project/args.stage;out.mkdir(exist_ok=False)
        budget_file=project/'budget.json'
        if not budget_file.exists():
            if args.stage!='assets':raise ValueError('Start with assets to initialize the total wall-clock budget')
            atomic_json(budget_file,dict(started_at=time.time(),hours=cfg['budget_hours']))
        budget=json.loads(budget_file.read_text())
        started=time.monotonic();deadline=started+max(0.,budget['started_at']+budget['hours']*3600-time.time())
        try:
            if time.monotonic()>=deadline:
                raise TimeoutError('Total project budget exhausted before model loading')
            occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if occupied:
                raise RuntimeError('GPU occupied; no overlapping job will be started')
            from run_stable import make_engine
            from stable_precision import install_locator_policy
            import torch
            import torchvision
            import numpy as np
            random.seed(20261103);np.random.seed(20261103);torch.manual_seed(20261103);torch.cuda.manual_seed_all(20261103)
            engine,env=make_engine(mat);install_locator_policy(engine,torch)
            atomic_json(out/'environment.json',dict(**env,locator_tf32=False))
            if args.stage=='assets':
                from imaging import prepare_view
                from glaucoma.local_region import image_tensor
                from glaucoma.r24_protocol import locator_input,locator_array
                from glaucoma.r23_model import center_from_logits
                from glaucoma.r23_geometry import unmap_center
                from glaucoma.r27_geometry import uniform_roi
                for split,count,positive in (('train',51396,1657),('valid',5849,191)):
                    rows=read_csv(mat/('metadata/official_train.csv' if split=='train' else 'metadata/official_valid.csv'))
                    if len(rows)!=count or sum(int(r['label']) for r in rows)!=positive:
                        raise ValueError('Official counts changed')
                    index={}
                    for path in (Path(cfg['data_root'])/split).iterdir():
                        if path.is_file():
                            if path.name.casefold() in index:raise ValueError('Ambiguous JPEG name')
                            index[path.name.casefold()]=path.resolve()
                    assets=[]
                    with (out/(split+'.jsonl')).open('x') as stream, torch.inference_mode():
                        for i,row in enumerate(rows):
                            path=index[row['image_id'].casefold()]
                            if not path.is_relative_to(Path(cfg['data_root']).resolve()/split):raise ValueError('Unsafe path')
                            for mirrored in ((False,) if split=='train' else (False,True)):
                                if time.monotonic()>deadline:raise TimeoutError('Asset budget exhausted')
                                full,g,d=prepare_view(path,mirrored)
                                from PIL import Image
                                with Image.open(path) as original:
                                    size=original.size[::-1] if original.getexif().get(274,1) in (5,6,7,8) else original.size
                                pil,lg=locator_input(d)
                                with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=True):
                                    heat=engine.locator(torch.from_numpy(locator_array(pil))[None].cuda());point,_=center_from_logits(heat)
                                field=tuple(round(v*s) for v,s in zip(g['field'],(d.width,d.height,d.width,d.height)))
                                roi=uniform_roi(size,d.size,field,unmap_center(point[0].cpu().tolist(),lg))
                                gl=None
                                if split=='valid':
                                    with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False),torch.autocast('cuda',dtype=torch.float16):
                                        gl=float(engine.global_model(image_tensor(full)[None].cuda()).float().item())
                                record=dict(image_id=row['image_id'],label=int(row['label']),path=str(path),mirrored=mirrored,
                                    decoded_size=list(d.size),original_size=list(size),roi_box=[roi[k] for k in ('px0','py0','px1','py1')],
                                    global_logit=gl,fallback=roi['fallback'],jpeg_bytes=path.stat().st_size,jpeg_mtime_ns=path.stat().st_mtime_ns)
                                stream.write(json.dumps(record)+'\n')
                            if (i+1)%1024==0:print('assets',split,i+1,'/',count,flush=True)
                complete(out);return 0
            require(project,'assets')
            train=[json.loads(s) for s in (project/'assets/train.jsonl').read_text().splitlines()]
            valid=[json.loads(s) for s in (project/'assets/valid.jsonl').read_text().splitlines()]
            if len(train)!=51396 or len(valid)!=11698:raise ValueError('Incomplete assets')
            for row in train+valid:
                stat=Path(row['path']).stat()
                if stat.st_size!=row['jpeg_bytes'] or stat.st_mtime_ns!=row['jpeg_mtime_ns']:raise ValueError('JPEG changed since assets')
            from glaucoma.models import build_model
            checkpoint=torch.load(mat/'inference/weights/local.pt',map_location='cpu',weights_only=True)
            def fresh():
                model=build_model(checkpoint['config'],False);model.load_state_dict(checkpoint['state_dict'],strict=True)
                return model.cuda()
            del engine
            torch.cuda.empty_cache()
            weight=torch.tensor(cfg['pos_weight'],device='cuda')
            if args.stage=='benchmark':
                if set(args.order)!=set(ARMS):raise ValueError('Exactly three unique arms required')
                subset=random.Random(20261103).sample(train,16*55)
                results={}
                for arm in args.order:
                    model=fresh().train();opt=torch.optim.AdamW(model.parameters(),lr=3e-6);scaler=torch.amp.GradScaler('cuda',init_scale=8192.)
                    batches=loader(Crops(subset,arm,True),16,cfg['workers']);updates=0
                    for step,batch in enumerate(batches):
                        if time.monotonic()>deadline:raise TimeoutError('Benchmark budget exhausted')
                        if step==5:
                            torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats()
                        _,updated=train_step(model,batch,opt,scaler,weight)
                        if step>=5:updates+=int(updated)
                    torch.cuda.synchronize();elapsed=time.monotonic()-start
                    results[arm]=dict(seconds_per_batch=elapsed/50,successful_updates=updates,attempted_updates=50,
                                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
                    print(arm,results[arm],flush=True)
                    del model,opt,scaler,batches;torch.cuda.empty_cache()
                atomic_json(out/'report.json',dict(results=results,order=args.order,
                    scope='DataLoader + H2D + forward/backward + AdamW; excludes assets/startup/checkpoints/validation; OS cache uncontrolled'))
                complete(out);return 0
            if args.stage=='smoke':
                subset=[r for label in (0,1) for r in [x for x in train if x['label']==label][:16]]
                model=fresh().train();opt=torch.optim.AdamW(model.parameters(),lr=3e-6);scaler=torch.amp.GradScaler('cuda')
                updates=sum(int(train_step(model,b,opt,scaler,weight)[1]) for b in loader(Crops(subset,'detail512',True),16,0))
                if not updates:raise ValueError('No successful smoke optimizer step')
                save_model(out/'smoke.pt',model,cfg,-1)
                model.load_state_dict(torch.load(out/'smoke.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
                # Include both classes, both views; this is integration, not a score estimate.
                ids=set(next(r['image_id'] for r in valid if r['label']==label) for label in (0,1))
                evaluate(model,[r for r in valid if r['image_id'] in ids],'detail512',out,'smoke',1,0,deadline)
                atomic_json(out/'report.json',dict(successful_updates=updates,no_real_model_selection=True));complete(out);return 0
            if args.stage=='train':
                require(project,'benchmark');require(project,'smoke')
                bench=json.loads((project/'benchmark/report.json').read_text())['results']['detail512']
                if bench['successful_updates']<45:raise ValueError('Excess AMP skips in benchmark')
                if 3*math.ceil(len(train)/16)*bench['seconds_per_batch']>max(0.,deadline-time.monotonic())*.7:
                    raise ValueError('Three-epoch estimate exceeds 70% of budget; do not start automatically')
                model=fresh();baseline=evaluate(model,valid,'low384',out,'source384',16,cfg['workers'],deadline)
                initial=evaluate(model,valid,'detail512',out,'epoch00',16,cfg['workers'],deadline)
                best=(initial['flip']['auroc'],initial['flip']['average_precision']);best_epoch=0
                save_model(out/'best.pt',model,cfg,0)
                opt=torch.optim.AdamW([dict(params=model.backbone.parameters(),lr=cfg['backbone_lr']),dict(params=model.head.parameters(),lr=cfg['head_lr'])],weight_decay=1e-4)
                scaler=torch.amp.GradScaler('cuda',init_scale=8192.);history=[];updates=0
                for epoch in range(1,4):
                    epoch_updates=updates
                    dataset=Crops(train,'detail512',True);dataset.epoch=epoch;model.train();losses=[];class_count=[0,0]
                    for step,batch in enumerate(loader(dataset,16,cfg['workers'],True,seed=20261103+epoch)):
                        if time.monotonic()>deadline:
                            save_model(out/'last.pt',model,cfg,epoch,opt,scaler,True)
                            raise TimeoutError('Training paused; partial epoch not eligible, no automatic resume')
                        loss,updated=train_step(model,batch,opt,scaler,weight);updates+=int(updated);losses.append(loss)
                        class_count[0]+=int((batch[1]==0).sum());class_count[1]+=int((batch[1]==1).sum())
                        if (step+1)%100==0:print('epoch',epoch,'batch',step+1,'/',math.ceil(len(train)/16),'loss',sum(losses)/len(losses),'updates',updates,flush=True)
                    if class_count!=[49739,1657]:raise ValueError('Incomplete training class coverage')
                    if updates==epoch_updates:raise ValueError('No successful optimizer updates in epoch')
                    save_model(out/'last.pt',model,cfg,epoch,opt,scaler)
                    metrics=evaluate(model,valid,'detail512',out,f'epoch{epoch:02}',16,cfg['workers'],deadline)
                    key=(metrics['flip']['auroc'],metrics['flip']['average_precision'])
                    if key>best:best=key;best_epoch=epoch;save_model(out/'best.pt',model,cfg,epoch)
                    history.append(dict(epoch=epoch,loss=sum(losses)/len(losses),successful_updates=updates,metrics=metrics))
                    atomic_json(out/'history.json',history)
                atomic_json(out/'report.json',dict(best_epoch=best_epoch,best_flip_key=best,source384=baseline,
                    initial_detail512=initial,seconds=time.monotonic()-started,selection='flip AUROC then AP; epoch0 retained',
                    remaining_b1_final_validation_required=True,no_submission=True))
                complete(out);return 0
            require(project,'train')
            model=fresh();best=torch.load(project/'train/best.pt',map_location='cpu',weights_only=True);model.load_state_dict(best['state_dict'],strict=True)
            result=evaluate(model,valid,'detail512',out,'best_b1',1,0,deadline)
            atomic_json(out/'report.json',dict(metrics=result,epoch=best['epoch'],no_submission=True,
                note='B1 local with frozen global/ROI assets. Live ROI/end-to-end latency and export packaging remain separate acceptance steps.'))
            complete(out);return 0
        except BaseException as error:
            atomic_json(out/'status.json',dict(status='paused' if isinstance(error,TimeoutError) else 'failed',traceback=traceback.format_exc()))
            if isinstance(error,TimeoutError):return 3
            raise


if __name__=='__main__':
    raise SystemExit(main())
