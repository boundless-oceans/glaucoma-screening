"""Deterministic new-global localization assets and strict ID-aligned anchors."""
import numpy as np
import torch
from .r12_v3 import SharedGlobalDataset
from .local_region import prepare_shared,image_tensor,attention_box,map_box_to_source


def roi_from_cam(cam,valid_mask,geometry):
    cam=np.asarray(cam);valid=np.asarray(valid_mask,dtype=bool)
    if cam.ndim!=2 or cam.shape!=valid.shape:raise ValueError('CAM/mask shape mismatch')
    result=attention_box(cam,valid)
    return dict(**result,source_box=map_box_to_source(result['box'],geometry))


class LocalizationInputs(SharedGlobalDataset):
    def __init__(self,manifest,data_root,for_training=False):
        super().__init__(manifest,data_root,training=for_training)

    def __getitem__(self,index):
        row=self.manifest.iloc[index];full,geometry,_=prepare_shared(self.root/row.relative_path)
        valid=np.asarray(full.resize((16,16))).max(2)>25
        return dict(image=image_tensor(full),valid_mask=valid,geometry=geometry,image_id=row.image_id)


def localization_collate(rows):
    return dict(image=torch.stack([r['image'] for r in rows]),
                valid_mask=np.stack([r['valid_mask'] for r in rows]),
                geometry=[r['geometry'] for r in rows],image_id=[r['image_id'] for r in rows])


def align_anchors(manifest,anchors):
    for frame in [manifest,anchors]:
        if frame.image_id.isna().any() or frame.image_id.duplicated().any():raise ValueError('Invalid anchor IDs')
    if len(manifest)!=len(anchors) or set(manifest.image_id)!=set(anchors.image_id):raise ValueError('Anchor coverage mismatch')
    if not np.isfinite(anchors.global_logit.to_numpy()).all():raise ValueError('Nonfinite global anchors')
    if str(anchors.fallback.dtype)!='bool':raise ValueError('Boolean fallback required')
    return anchors.set_index('image_id').loc[manifest.image_id].reset_index()
