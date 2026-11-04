"""Experimental locator-first schedule with CPU crop/global GPU overlap.

Derived from pinned captain inference/predict.py::Inference.predict. The mirror
is after EXIF/RGB decode and before BOTH global preprocessing and localization.
No original-view ROI or cached logits are used. False is an equivalence audit.
"""
import hashlib, time

from contextlib import contextmanager


def _crop_after_decode(original_future, mapped, mirrored):
    from parallel_decode import crop_decoded
    return crop_decoded(original_future.result(),mapped,mirrored=mirrored)


@contextmanager
def background_crop(executor,original_future,mapped,mirrored):
    # Same single worker: original decode was queued first and completes before
    # this crop starts. The outer decode context owns the original image.
    future=executor.submit(_crop_after_decode,original_future,mapped,mirrored)
    try:
        yield future
    finally:
        # Drain on global-model failure before outer context closes the image.
        if not future.cancel():
            crop,_=future.result()
            crop.close()


from predict import fuse_probability

def predict_scheduled(self,path,executor,mirrored=True):
    import torch
    from PIL import Image
    from glaucoma.local_region import image_tensor
    from imaging import prepare_view
    from glaucoma.r23_model import center_from_logits
    from glaucoma.r23_geometry import unmap_center
    from glaucoma.r24_protocol import locator_input,locator_array
    from glaucoma.r27_geometry import uniform_roi
    from parallel_decode import background_decode, crop_decoded
    cuda=self.device.type=='cuda'
    if cuda:torch.cuda.synchronize(self.device)
    begin=time.perf_counter()
    with background_decode(executor,path) as original_future, torch.inference_mode():
        with Image.open(path) as im:
            ow,oh=im.size
            if im.getexif().get(274,1) in (5,6,7,8):ow,oh=oh,ow
        full,g,decoded=prepare_view(path,mirrored=mirrored)
        field=tuple(round(v*s) for v,s in zip(g['field'],(decoded.width,decoded.height,decoded.width,decoded.height)))
        pil,lg=locator_input(decoded)
        with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=True):
            with torch.autocast(self.device.type,enabled=False):
                heat=self.locator(torch.from_numpy(locator_array(pil))[None].to(self.device));points,_=center_from_logits(heat)
        point=points[0].cpu().tolist();center=unmap_center(point,lg)
        roi=uniform_roi((ow,oh),decoded.size,field,center)
        mapped=dict(roi_box=[roi[k] for k in ('px0','py0','px1','py1')], decoded_size=list(decoded.size), original_size=[ow,oh])
        with background_crop(executor,original_future,mapped,mirrored) as crop_future:
            with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False):
                with torch.autocast(self.device.type,dtype=torch.float16,enabled=cuda):
                    gl=float(self.global_model(image_tensor(full)[None].to(self.device),torch.zeros((1,2),device=self.device)).float().item())
            crop,source_box=crop_future.result()
            with torch.backends.cudnn.flags(enabled=True,benchmark=False,deterministic=False):
                with torch.autocast(self.device.type,dtype=torch.float16,enabled=cuda):
                    ll=float(self.local_model(image_tensor(crop)[None].to(self.device),torch.zeros((1,2),device=self.device)).float().item())
            p=fuse_probability(gl,ll,roi['fallback'])
            if cuda:torch.cuda.synchronize(self.device)
            return dict(global_logit=gl,local_logit=ll,fused_logit=gl if roi['fallback'] else .5*(gl+ll),probability=p,label=int(p>=.5),
                        roi_box=[roi[k] for k in ('px0','py0','px1','py1')],decoded_size=list(decoded.size),original_size=[ow,oh],field=list(field),
                        locator_input_center=point,locator_center=list(center),padding_fraction=roi['padding_fraction'],fraction=.30,
                        fallback=roi['fallback'],fallback_reason=roi['fallback_reason'],crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                        source_box=list(source_box),elapsed_ms=1000*(time.perf_counter()-begin))
