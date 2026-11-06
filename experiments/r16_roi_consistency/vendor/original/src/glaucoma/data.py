from pathlib import Path
import random
import math

import numpy as np
from PIL import Image,ImageOps,ImageEnhance


def fundus_bbox(image):
    """Conservative bright-field bounds with fixed thresholds and safe fallback."""
    width,height=image.size;probe=image.copy();probe.thumbnail((192,192))
    array=np.asarray(probe);h,w=array.shape[:2]
    mask=array.max(axis=2)>12
    ys=np.flatnonzero(mask.sum(axis=1)>=max(2,.02*w))
    xs=np.flatnonzero(mask.sum(axis=0)>=max(2,.02*h))
    full=(0,0,width,height)
    if not len(xs) or not len(ys):return full
    margin=max(1,round(max(h,w)*.02))
    x0,x1=max(0,int(xs[0])-margin),min(w,int(xs[-1])+1+margin)
    y0,y1=max(0,int(ys[0])-margin),min(h,int(ys[-1])+1+margin)
    if (x1-x0)*(y1-y0)<.35*h*w or not (x0<=w/2<=x1 and y0<=h/2<=y1):return full
    return (math.floor(x0*width/w),math.floor(y0*height/h),
            min(width,math.ceil(x1*width/w)),min(height,math.ceil(y1*height/h)))


def cache_namespace(config):
    if config.get('local_region'):
        region=config['local_region']
        version=region['version'];teacher=region['teacher_sha256']
        if not all(c.isalnum() or c in '-_' for c in version+teacher):raise ValueError('Invalid ROI cache identity')
        return f"{config.get('image_size',384)}_roi_{version}_{teacher[:16]}"
    draft=config.get('jpeg_draft_scale',0)
    return str(config.get('image_size',384))+('_crop' if config.get('fundus_crop',False) else '')+(f'_draft{draft}' if draft else '')


def prepare_image(image,size=384,jpeg_draft_scale=0,fundus_crop=False):
    if jpeg_draft_scale not in [0,1,2]:raise ValueError('jpeg_draft_scale must be 0, 1 or 2')
    if not isinstance(fundus_crop,bool):raise ValueError('fundus_crop must be boolean')
    if fundus_crop:
        width,height=image.size
        if image.getexif().get(274,1) in [5,6,7,8]:width,height=height,width
        if jpeg_draft_scale:image.draft('RGB',(size*jpeg_draft_scale,size*jpeg_draft_scale))
        image=ImageOps.exif_transpose(image).convert('RGB');dw,dh=image.size
        x0,y0,x1,y1=fundus_bbox(image)
        image=image.crop((x0,y0,x1,y1))
        actual_width=(x1-x0)*width/dw;actual_height=(y1-y0)*height/dh
        target=(size,max(1,round(actual_height*size/actual_width))) if actual_width>=actual_height else (max(1,round(actual_width*size/actual_height)),size)
        image=image.resize(target,resample=Image.Resampling.BICUBIC)
        return ImageOps.pad(image,(size,size),method=Image.Resampling.BICUBIC,color=(0,0,0))
    if not jpeg_draft_scale:
        image=ImageOps.exif_transpose(image).convert('RGB')
        return ImageOps.pad(image,(size,size),method=Image.Resampling.BICUBIC,color=(0,0,0))
    # JPEG draft may round decoder dimensions. Retain the original oriented
    # aspect ratio instead of deriving the field of view from rounded dimensions.
    width,height=image.size
    if image.getexif().get(274,1) in [5,6,7,8]:width,height=height,width
    image.draft('RGB',(size*jpeg_draft_scale,size*jpeg_draft_scale))
    image=ImageOps.exif_transpose(image).convert('RGB')
    target=(size,max(1,round(height*size/width))) if width>=height else (max(1,round(width*size/height)),size)
    image=image.resize(target,resample=Image.Resampling.BICUBIC)
    return ImageOps.pad(image,(size,size),method=Image.Resampling.BICUBIC,color=(0,0,0))


def fit_age_stats(manifest):
    import pandas as pd
    a=pd.to_numeric(manifest.age,errors='coerce').to_numpy(dtype=float)
    valid=np.isfinite(a)&(a>0)&(a<=120)
    median=float(np.median(a[valid])) if valid.any() else 0.
    filled=np.where(valid,a,median)
    return dict(median=median,mean=float(filled.mean()) if len(filled) else 0.,
                std=float(filled.std()) if len(filled) and filled.std()>1e-8 else 1.,
                valid_count=int(valid.sum()),invalid_count=int((~valid).sum()),upper_bound=120)


def transform_age(age,stats):
    try: a=float(age)
    except (TypeError,ValueError): a=float('nan')
    valid=np.isfinite(a) and 0<a<=stats['upper_bound']
    value=a if valid else stats['median']
    return np.array([(value-stats['mean'])/stats['std'],0. if valid else 1.],dtype=np.float32)


class FundusDataset:
    def __init__(self,manifest,data_root,config,training=False):
        self.manifest=manifest.reset_index(drop=True); self.data_root=Path(data_root)
        self.config=config; self.training=training
        self.region_boxes=None
        if config.get('local_region'):
            import pandas as pd
            from .local_region import validate_box
            boxes=pd.read_csv(config['local_region']['boxes_csv'])
            if boxes.image_id.duplicated().any():raise ValueError('Duplicate ROI image IDs')
            if not set(self.manifest.image_id)<=set(boxes.image_id):raise ValueError('ROI coverage mismatch')
            self.region_boxes={r.image_id:validate_box([r.x0,r.y0,r.x1,r.y1]) for r in boxes.itertuples()}

    def __len__(self): return len(self.manifest)

    def __getitem__(self,index):
        import torch
        row=self.manifest.iloc[index]; size=self.config.get('image_size',384)
        draft_scale=self.config.get('jpeg_draft_scale',0)
        cached=None
        if self.config.get('cache_dir'):
            namespace=cache_namespace(self.config)
            cached=Path(self.config['cache_dir'])/namespace/row.official_split/(row.image_id+'.png')
        path=cached if cached is not None and cached.is_file() else self.data_root/row.relative_path
        try:
            with Image.open(path) as im:
                if cached is not None and path==cached:
                    image=ImageOps.exif_transpose(im).convert('RGB')
                    if image.size!=(size,size):raise ValueError('Cached preprocessing size mismatch')
                elif self.region_boxes is not None:
                    from .local_region import prepare_local
                    image=prepare_local(self.data_root/row.relative_path,self.region_boxes[row.image_id],size)
                else:image=prepare_image(im,size,jpeg_draft_scale=draft_scale,fundus_crop=self.config.get('fundus_crop',False))
        except Exception as e: raise RuntimeError(f'Image decode failed: {row.image_id}: {e}') from e
        if self.training and self.config.get('augmentation',True):
            if random.random()<.5: image=ImageOps.mirror(image)
            image=image.rotate(random.uniform(-15,15),resample=Image.Resampling.BICUBIC,fillcolor=(0,0,0))
            for enhancer in [ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color]:
                image=enhancer(image).enhance(random.uniform(.9,1.1))
        arr=np.asarray(image,dtype=np.float32).transpose(2,0,1)/255.
        mean=np.array([.485,.456,.406],dtype=np.float32)[:,None,None]
        std=np.array([.229,.224,.225],dtype=np.float32)[:,None,None]
        age=transform_age(row.age,self.config['age_stats']) if self.config.get('age_stats') else np.zeros(2,dtype=np.float32)
        label=float(row.label)
        if not np.isfinite(label):label=-1.
        return dict(image=torch.from_numpy((arr-mean)/std),age=torch.from_numpy(age),
                    label=torch.tensor(label,dtype=torch.float32),image_id=row.image_id)
