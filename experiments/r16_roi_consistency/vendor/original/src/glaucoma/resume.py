import json
from pathlib import Path
import torch


def restore_training_state(model,optimizer,scheduler,scaler,path,expected_split_sha256):
    path=Path(path);state=torch.load(path,map_location='cpu',weights_only=True)
    saved=state.get('split_sha256')
    if saved is None:
        metadata=path.parent/'provenance.json'
        if metadata.is_file():saved=json.loads(metadata.read_text()).get('split_sha256')
    if saved!=expected_split_sha256:
        raise ValueError('Resume checkpoint and current data split hashes differ')
    for key in ['state_dict','optimizer','epoch']:
        if key not in state:raise ValueError(f'Checkpoint missing {key}')
    model.load_state_dict(state['state_dict'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:scheduler.load_state_dict(state['scheduler'])
    if scaler is not None and state.get('scaler'):scaler.load_state_dict(state['scaler'])
    return state
