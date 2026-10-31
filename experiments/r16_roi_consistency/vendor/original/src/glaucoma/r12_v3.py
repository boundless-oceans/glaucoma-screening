"""Whole-fundus training inputs with the frozen shared768 deployment geometry."""
import random
from pathlib import Path,PurePosixPath
import torch
from PIL import ImageOps,ImageEnhance,Image
from torch.utils.data import Dataset
from .local_region import prepare_shared,image_tensor


class SharedGlobalDataset(Dataset):
    def __init__(self,manifest,data_root,training=False):
        required={'image_id','relative_path','official_split','role','label'}
        if not required<=set(manifest.columns):raise ValueError('Missing manifest columns')
        if manifest.image_id.isna().any() or manifest.image_id.duplicated().any():raise ValueError('Invalid IDs')
        if not manifest.label.isin([0,1]).all():raise ValueError('Binary labelled samples required')
        if not manifest.official_split.isin(['train','val']).all():raise ValueError('Test split forbidden')
        if training and manifest.role.astype(str).str.lower().isin(['dev','test']).any():
            raise ValueError('Evaluation roles cannot be trained')
        for relative in manifest.relative_path:
            p=PurePosixPath(str(relative))
            if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in str(relative) or p.parts[0] not in ['train','val','valid']:
                raise ValueError('Non-training image path forbidden')
        self.manifest=manifest.reset_index(drop=True).copy()
        self.root=Path(data_root);self.training=bool(training)

    def __len__(self):return len(self.manifest)

    def __getitem__(self,index):
        row=self.manifest.iloc[index]
        full,_,_=prepare_shared(self.root/row.relative_path)
        if self.training:
            if random.random()<.5:full=ImageOps.mirror(full)
            full=full.rotate(random.uniform(-15,15),resample=Image.Resampling.BICUBIC,fillcolor=(0,0,0))
            for enhancer in [ImageEnhance.Brightness,ImageEnhance.Contrast,ImageEnhance.Color]:
                full=enhancer(full).enhance(random.uniform(.9,1.1))
        return dict(image=image_tensor(full),age=torch.zeros(2,dtype=torch.float32),
                    label=torch.tensor(float(row.label),dtype=torch.float32),image_id=row.image_id)
