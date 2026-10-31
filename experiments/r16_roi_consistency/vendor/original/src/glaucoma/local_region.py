"""Frozen classifier attention and original-JPEG ROI geometry.

This is weakly supervised attention, not a validated optic-disc detector.
No fitting or label lookup occurs in these functions.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .data import fundus_bbox


def prepare_global(path, size=512):
    """Exactly preserve the existing draft1 whole-fundus input and its mapping."""
    result,geometry,_=_prepare(path,size,size)
    return result,geometry


def prepare_shared(path, size=512):
    """One original JPEG decode shared by global512 and local384 (draft768)."""
    return _prepare(path,size,768)


def _prepare(path,size,decode_size):
    with Image.open(path) as im:
        width,height=im.size
        if im.getexif().get(274,1) in [5,6,7,8]:width,height=height,width
        im.draft('RGB',(decode_size,decode_size))
        # exif_transpose/convert copy even already-oriented RGB images. Avoid
        # redundant full-frame copies; pixels remain identical (tests cover EXIF).
        image=ImageOps.exif_transpose(im) if im.getexif().get(274,1)!=1 else im
        if image.mode!='RGB':image=image.convert('RGB')
        else:image.load()
    dw,dh=image.size
    x0,y0,x1,y1=fundus_bbox(image)
    actual_width=(x1-x0)*width/dw;actual_height=(y1-y0)*height/dh
    target=(size,max(1,round(actual_height*size/actual_width))) if actual_width>=actual_height else (max(1,round(actual_width*size/actual_height)),size)
    resized=image.crop((x0,y0,x1,y1)).resize(target,Image.Resampling.BICUBIC)
    result=ImageOps.pad(resized,(size,size),method=Image.Resampling.BICUBIC,color=(0,0,0))
    geometry=dict(size=size,field=(x0/dw,y0/dh,x1/dw,y1/dh),target=target,
                  padding=(round((size-target[0])/2),round((size-target[1])/2)))
    # The decoded pixels stay live after closing the source file.
    return result,geometry,image


def validate_box(box):
    values=tuple(float(x) for x in box)
    if len(values)!=4 or not all(math.isfinite(x) for x in values):raise ValueError('Invalid finite box')
    x0,y0,x1,y1=values
    if not (0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('Invalid normalized box')
    return values


def map_box_to_source(box, geometry):
    """Unpad and map a global-input rectangle to the oriented original image."""
    x0,y0,x1,y1=box;px,py=geometry['padding'];tw,th=geometry['target']
    x0=max(x0,px);y0=max(y0,py);x1=min(x1,px+tw);y1=min(y1,py+th)
    if x0>=x1 or y0>=y1:raise ValueError('ROI lies entirely in padding')
    fx0,fy0,fx1,fy1=geometry['field']
    return validate_box((fx0+(x0-px)/tw*(fx1-fx0),fy0+(y0-py)/th*(fy1-fy0),
                         fx0+(x1-px)/tw*(fx1-fx0),fy0+(y1-py)/th*(fy1-fy0)))


def attention_box(cam, valid_mask, size=512, fraction=.60, min_ratio=1.5):
    cam=np.asarray(cam,dtype=np.float64);valid=np.asarray(valid_mask,dtype=bool).copy()
    if cam.ndim!=2 or cam.shape!=valid.shape:raise ValueError('CAM and mask shape mismatch')
    if not np.isfinite(cam).all():raise ValueError('CAM must be finite')
    if not (0<fraction<=1 and min_ratio>1):raise ValueError('Invalid ROI policy')
    h,w=cam.shape;yy,xx=np.mgrid[:h,:w]
    valid &= ((xx+.5-w/2)/(w*.46))**2+((yy+.5-h/2)/(h*.46))**2<1
    heat=np.abs(cam);n=int(valid.sum())
    peak=float(heat[valid].max()) if n else 0.
    mean=float(heat[valid].mean()) if n else 0.
    ratio=peak/max(mean,1e-12)
    if n<max(4,round(h*w*.05)) or peak<=1e-10 or ratio<min_ratio:
        return dict(box=(0,0,size,size),fallback=True,concentration=ratio)
    cy,cx=np.unravel_index(np.argmax(np.where(valid,heat,-np.inf)),heat.shape)
    side=round(size*fraction)
    x0=min(max(round((cx+.5)*size/w-side/2),0),size-side)
    y0=min(max(round((cy+.5)*size/h-side/2),0),size-side)
    return dict(box=(x0,y0,x0+side,y0+side),fallback=False,concentration=ratio)


def prepare_local(path, box, size=384):
    """Use the same draft768 source as shared inference, never a global cache."""
    validate_box(box)
    if Path(path).suffix.casefold() not in ['.jpg','.jpeg']:raise ValueError('ROI source must be original JPEG')
    with Image.open(path) as im:
        im.draft('RGB',(768,768))
        image=ImageOps.exif_transpose(im).convert('RGB')
    return crop_decoded(image,box,size)


def crop_decoded(image,box,size=384):
    x0,y0,x1,y1=validate_box(box)
    w,h=image.size
    pixel_box=(math.floor(x0*w),math.floor(y0*h),min(w,math.ceil(x1*w)),min(h,math.ceil(y1*h)))
    return ImageOps.pad(image.crop(pixel_box),(size,size),method=Image.Resampling.BICUBIC,color=(0,0,0))


def image_tensor(image):
    import torch
    arr=np.asarray(image,dtype=np.float32).transpose(2,0,1)/255
    mean=np.array([.485,.456,.406],dtype=np.float32)[:,None,None]
    std=np.array([.229,.224,.225],dtype=np.float32)[:,None,None]
    return torch.from_numpy((arr-mean)/std)


def forward_global_cam(model, images):
    """Reuse the same B2 feature pass for probability and signed CAM."""
    import torch
    if model.training or model.use_age:raise ValueError('Frozen pure-vision eval model required')
    features=model.backbone.features(images)
    logits=model.head(torch.flatten(model.backbone.avgpool(features),1)).squeeze(1)
    cam=(features.float()*model.head[-1].weight[0].float()[None,:,None,None]).sum(1)
    return torch.sigmoid(logits.float()),cam


def region_from_cam(cam, global_image, geometry):
    small=np.asarray(global_image.resize((cam.shape[1],cam.shape[0])))
    result=attention_box(cam,small.max(2)>25)
    return dict(**result,source_box=map_box_to_source(result['box'],geometry))
