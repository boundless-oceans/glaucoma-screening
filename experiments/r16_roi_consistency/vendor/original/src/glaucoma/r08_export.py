"""Frozen E16 + R05 global/CAM inference. No training, adaptation or downloads."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .local_region import prepare_shared,image_tensor,region_from_cam,crop_decoded
from .r08 import configure_runtime,fuse_logits

POLICY=dict(global_size=512,local_size=384,shared_jpeg_draft=768,roi_fraction=.60,
            roi_min_ratio=1.5,fusion='equal_raw_logit_mean',fallback='global',
            precision='fp16',tta='none',batch_size=1,threshold=.5,use_age=False,
            cudnn_benchmark=False)


def validate_bundle(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'r08_logit.json').read_text())
    if manifest.get('schema')!='r08-logit-v1' or manifest.get('policy')!=POLICY:
        raise ValueError('R08 bundle policy mismatch')
    if set(manifest['weights'])!={'global','local'}:raise ValueError('Two weights required')
    for name,item in manifest['weights'].items():
        path=folder/item['path']
        if item['path']!=name+'.pt' or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('Invalid weight path')
        if path.stat().st_size!=item['bytes']:raise ValueError('Weight size mismatch')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Weight digest mismatch')
    if sum(x['bytes'] for x in manifest['weights'].values())>450000000:raise ValueError('Weight budget exceeded')
    return manifest


class R08Pipeline:
    def __init__(self,global_model,local_model,device):
        if global_model.training or local_model.training:raise ValueError('Frozen eval models required')
        self.global_model=global_model;self.local_model=local_model
        self.device=torch.device(device);self.threshold=.5;self.config=dict(POLICY)

    @torch.inference_mode()
    def predict_details(self,path):
        full,geometry,decoded=prepare_shared(path)
        with torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
            features=self.global_model.backbone.features(image_tensor(full)[None].to(self.device))
            logits=self.global_model.head(torch.flatten(self.global_model.backbone.avgpool(features),1)).squeeze(1)
            cam=(features.float()*self.global_model.head[-1].weight[0].float()[None,:,None,None]).sum(1)
        roi=region_from_cam(cam[0].cpu().numpy(),full,geometry);local_logit=None
        if not roi['fallback']:
            with torch.autocast(self.device.type,dtype=torch.float16,enabled=self.device.type=='cuda'):
                value=self.local_model(image_tensor(crop_decoded(decoded,roi['source_box']))[None].to(self.device))
            local_logit=float(value.item())
        global_logit=float(logits.item())
        probability=fuse_logits(global_logit,local_logit,roi['fallback'])
        return dict(probability=probability,global_logit=global_logit,local_logit=local_logit,fallback=roi['fallback'])

    def predict_one(self,path,age=None):
        probability=self.predict_details(path)['probability']
        return probability,int(probability>=self.threshold)


def load_r08(folder,device='cuda'):
    from .models import build_model
    manifest=validate_bundle(folder);models=[]
    configure_runtime(training=False)
    for name,architecture,size in [('global','efficientnet_b2',512),('local','convnext_tiny',384)]:
        checkpoint=torch.load(Path(folder)/manifest['weights'][name]['path'],map_location='cpu',weights_only=True)
        config=checkpoint['config']
        if config.get('model')!=architecture or config.get('image_size')!=size or config.get('use_age',False):
            raise ValueError('Unexpected checkpoint architecture')
        model=build_model(config,pretrained=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        models.append(model.to(device).eval())
    pipeline=R08Pipeline(*models,device)
    pipeline.weights=[dict(name=name,**item) for name,item in manifest['weights'].items()]
    return pipeline


def main():
    from .standalone import run_inference
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','images','metadata','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--expected-count',type=int,default=5855)
    args=parser.parse_args()
    report=run_inference(args.bundle,args.images,args.metadata,args.output,args.device,args.expected_count,pipeline_loader=load_r08)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
