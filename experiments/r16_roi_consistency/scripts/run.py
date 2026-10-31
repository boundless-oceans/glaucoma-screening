"""SERVER ONLY, isolated R16 stages; --help is stdlib-only. No automatic training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','assets','train','evaluate'])
    parser.add_argument('--local-initializer',type=Path,help='Train only; hash must match registered generic initializer')
    args=parser.parse_args()
    if args.local_initializer is not None and args.stage!='train':
        parser.error('--local-initializer is for the train stage only')
    root=Path(__file__).resolve().parent
    registration=json.loads((root/'PREPARATION.json').read_bytes())
    if sha(root/'run.py')!=registration['runner_sha256']: raise ValueError('Runner changed')
    for name,expected in registration['source_sha256'].items():
        if sha(root/name)!=expected: raise ValueError('Prepared source/config changed: '+name)
    checkpoint=Path(registration['global_checkpoint'])
    if sha(checkpoint)!=registration['global_checkpoint_sha256']: raise ValueError('Reference checkpoint changed')
    data=Path(registration['data_root'])
    for split,expected in registration['official_csv_sha256'].items():
        if sha(data/(split+'.csv'))!=expected: raise ValueError('Official CSV changed')
    report=root/'reports/R16-roi-consistency'
    lock=report/'.stage.lock'
    # OS-held lock releases after crashes; stale file contents do not block recovery.
    descriptor=lock.open('a+b')
    if os.name=='nt': raise RuntimeError('Run model stages on Linux server; local CPU checks live in the recipe tests')
    import fcntl
    fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(root);sys.path.insert(0,str(root/'src'))
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    from PIL import __version__ as pillow_version
    from glaucoma.r13_runtime import require_official_dev_environment
    from glaucoma.r16_protocol import validate_roles
    require_official_dev_environment(torch.__version__,torchvision.__version__)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA server required')
    spec=importlib.util.spec_from_file_location('r16_controller',root/'tools/run_round_sixteen.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    fit=pd.read_csv(root/'manifests/train.csv');valid=pd.read_csv(root/'manifests/valid.csv')
    validate_roles(fit.to_dict('records'),valid.to_dict('records'))
    for frame,split in ((fit,'train'),(valid,'valid')):
        official=pd.read_csv(data/(split+'.csv'))
        if len(official)!=len(frame) or official.Image_name.str.casefold().duplicated().any():
            raise ValueError('Official row count or duplicate IDs differ')
        normalize=lambda n:str(n).rsplit('.',1)[0].casefold()
        if {normalize(r.image_id):int(r.label) for r in frame.itertuples()}!={normalize(r.Image_name):int(r.True_Label) for r in official.itertuples()}:
            raise ValueError('Official labels/IDs differ')
    contract=dict(variant=registration['variant'],reference_sha256=registration['global_checkpoint_sha256'],
                  source_sha256=registration['source_sha256'],official_csv_sha256=registration['official_csv_sha256'],
                  runtime=dict(torch=str(torch.__version__),torchvision=str(torchvision.__version__),
                               pandas=pd.__version__,numpy=np.__version__,pillow=pillow_version))
    fingerprint=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    reference=dict(run=str(checkpoint.parent),best_sha256=registration['global_checkpoint_sha256'])
    if checkpoint.name!='best.pt': raise ValueError('Reference must retain best.pt filename for archived R16 loader')
    if args.stage=='check':
        (report/'check.json').unlink(missing_ok=True)
        from glaucoma.local_region import prepare_shared,crop_decoded,prepare_local,image_tensor
        from glaucoma.r13 import roi_from_cam
        from glaucoma.models import build_model
        from glaucoma.data import FundusDataset
        from glaucoma.r08 import bounded_train_epoch
        controller.configure_runtime(False);controller.seed()
        ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if ck['epoch']!=7 or ck['config']['model']!='efficientnet_b2' or ck['config'].get('use_age'):
            raise ValueError('Wrong global checkpoint architecture/epoch')
        teacher=build_model(ck['config'],False);teacher.load_state_dict(ck['state_dict'],strict=True)
        teacher=teacher.cuda().eval().requires_grad_(False)
        rows=[];pixel_matches=0
        sample=pd.concat([fit[fit.label==label].sample(n=16,random_state=20261029) for label in (0,1)],ignore_index=True)
        for row in sample.itertuples():
            full,geometry,decoded=prepare_shared(data/row.relative_path)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                features=teacher.backbone.features(image_tensor(full)[None].cuda())
                logit=teacher.head(torch.flatten(teacher.backbone.avgpool(features),1)).squeeze(1)
                cam=(features.float()*teacher.head[-1].weight[0].float()[None,:,None,None]).sum(1)
            mask=np.asarray(full.resize((16,16))).max(2)>25
            roi=roi_from_cam(cam[0].cpu().numpy(),mask,geometry)
            rows.append(dict(image_id=row.image_id,global_logit=float(logit.item()),
                             **dict(zip(['x0','y0','x1','y1'],roi['source_box'])),fallback=bool(roi['fallback'])))
        assets=pd.DataFrame(rows);path=report/'check_boxes.csv';assets.to_csv(path,index=False,float_format='%.17g')
        restored=pd.read_csv(path,float_precision='round_trip')
        for row,box in zip(sample.itertuples(),restored.itertuples()):
            coords=(box.x0,box.y0,box.x1,box.y1)
            expected=assets.loc[assets.image_id==row.image_id,['x0','y0','x1','y1']].to_numpy()[0]
            if not np.array_equal(np.array(coords),expected): raise ValueError('Coordinate round trip failed')
            _,_,decoded=prepare_shared(data/row.relative_path)
            a=np.asarray(prepare_local(data/row.relative_path,coords));b=np.asarray(crop_decoded(decoded,expected))
            if not np.array_equal(a,b): raise ValueError('Live/cache crop pixels differ')
            pixel_matches+=1
        del teacher,ck;torch.cuda.empty_cache()
        config=dict(model='convnext_tiny',image_size=384,use_age=False,augmentation=True,
                    local_region=dict(boxes_csv=str(path),decode_size=768,teacher_sha256=reference['best_sha256'],version='r16-consistency-check'))
        local=build_model(config,False).cuda()
        dataset=FundusDataset(sample,data,config,True)
        batch=torch.utils.data.default_collate([dataset[i] for i in (0,1,16,17)])
        before=local.head[-1].weight.detach().clone()
        stats=bounded_train_epoch(local,[batch]*2,controller.optimizer(local),torch.device('cuda'),
                                 (len(fit)-int(fit.label.sum()))/int(fit.label.sum()),lambda:False,
                                 torch.amp.GradScaler('cuda',init_scale=16.))
        if stats['updates']<1 or torch.equal(before,local.head[-1].weight):
            raise RuntimeError('Local forward/backward update check failed')
        controller.dump(report/'check.json',dict(status='completed',fingerprint=fingerprint,contract=contract,
                         training_images=32,exact_pixel_matches=pixel_matches,updates=stats['updates'],
                         random_initialized_local_weights_discarded=True,generic_initializer_checked=False))
        print('CHECK_PASSED; local random model discarded, no formal training')
        return
    check=json.loads((report/'check.json').read_bytes())
    if check['status']!='completed' or check['fingerprint']!=fingerprint: raise ValueError('Run check in this same environment first')
    if args.stage=='assets':
        if (report/'assets.complete.json').exists(): raise FileExistsError('Assets already completed; reuse or prepare a new project')
        teacher_roi=controller.make_assets(reference,fit,valid)
        if len(teacher_roi)!=5849: raise ValueError('Incomplete validation anchors')
        files={name:sha(report/name) for name in ('roi_boxes.csv','train_assets.csv','valid_assets.csv','assets_registration.json')}
        controller.dump(report/'assets.complete.json',dict(status='completed',fingerprint=fingerprint,files=files))
        print('ASSETS_COMPLETE; train/valid B1, round_trip read, no local training')
        return
    assets=json.loads((report/'assets.complete.json').read_bytes())
    if assets['fingerprint']!=fingerprint: raise ValueError('Assets use another runtime/source')
    for name,expected in assets['files'].items():
        if sha(report/name)!=expected: raise ValueError('Assets changed: '+name)
    if args.stage=='train':
        initializer=(args.local_initializer.resolve() if args.local_initializer else Path(controller.SET['local_initializer']))
        controller.SET['local_initializer']=str(initializer)
        if not initializer.is_file() or sha(initializer)!=registration['generic_local_initializer_sha256']:
            raise ValueError('Prepare a new project with the generic 22K->1K initializer before training')
        if (report/'training_started.json').exists(): raise FileExistsError('One-shot trajectory already started; no implicit restart/resume')
        controller.dump(report/'training_started.json',dict(fingerprint=fingerprint,source_sha256=registration['source_sha256']))
        result=controller.train('local',fit,valid,pd.read_csv(report/'valid_assets.csv',float_precision='round_trip'),checkpoint)
        controller.dump(report/'local_result.json',result)
        print('LOCAL_TRAINING_COMPLETED; run evaluate explicitly')
    else:
        result=json.loads((report/'local_result.json').read_bytes())
        evaluation=controller.final_b1(reference,result,valid)
        controller.dump(report/'results.json',dict(frozen_global=reference,local=result,evaluation=evaluation,variant=registration['variant']))
        print('VALIDATION_EVALUATION_COMPLETED; no test predictions or platform submission')


if __name__=='__main__': main()
