"""Server inference only. Original arm calls the unchanged archived R08Pipeline."""
from .imaging import prepare_view


def predict_view(pipe,path,mirrored=True):
    # No top-level torch import, so geometry and CLI tests remain CPU-only.
    import torch
    from glaucoma.local_region import image_tensor,region_from_cam,crop_decoded
    from glaucoma.r08 import fuse_logits
    full,geometry,decoded=prepare_view(path,mirrored=mirrored)
    with torch.inference_mode():
        with torch.autocast(pipe.device.type,dtype=torch.float16,enabled=pipe.device.type=='cuda'):
            features=pipe.global_model.backbone.features(image_tensor(full)[None].to(pipe.device))
            logits=pipe.global_model.head(torch.flatten(pipe.global_model.backbone.avgpool(features),1)).squeeze(1)
            cam=(features.float()*pipe.global_model.head[-1].weight[0].float()[None,:,None,None]).sum(1)
        roi=region_from_cam(cam[0].cpu().numpy(),full,geometry)
        local_logit=None
        if not roi['fallback']:
            with torch.autocast(pipe.device.type,dtype=torch.float16,enabled=pipe.device.type=='cuda'):
                local_logit=float(pipe.local_model(image_tensor(crop_decoded(decoded,roi['source_box']))[None].to(pipe.device)).item())
        global_logit=float(logits.item())
        return dict(probability=fuse_logits(global_logit,local_logit,roi['fallback']),
                    global_logit=global_logit,local_logit=local_logit,fallback=bool(roi['fallback']))
