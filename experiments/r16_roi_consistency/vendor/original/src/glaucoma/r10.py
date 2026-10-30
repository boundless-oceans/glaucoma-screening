"""Strict official ConvNeXt 22K->1K initialization into the frozen torchvision backend."""
from collections.abc import Mapping
import hashlib
from pathlib import Path
import re
import torch


def convert_official_convnext_state(source_state, target_state):
    """Map every backbone tensor, discard only the verified 1000-class upstream head."""
    if not isinstance(source_state, Mapping) or not isinstance(target_state, Mapping):
        raise ValueError("State dictionaries required")
    norm=target_state.get("classifier.0.weight")
    if norm is None or norm.ndim!=1:
        raise ValueError("Missing target final normalization")
    dim=norm.numel()
    for key,shape in [("head.weight",(1000,dim)),("head.bias",(1000,))]:
        value=source_state.get(key)
        if not isinstance(value,torch.Tensor) or tuple(value.shape)!=shape:
            raise ValueError("Expected verified ImageNet-1K classifier")
    mapped={}
    for name,value in source_state.items():
        if name in {"head.weight","head.bias"}:
            continue
        if not isinstance(value,torch.Tensor) or not value.is_floating_point():
            raise ValueError("Invalid backbone tensor: "+name)
        found=re.fullmatch(r"downsample_layers.([0-3]).([01]).(weight|bias)",name)
        if found:
            stage,part,suffix=found.groups()
            target=f"features.{2*int(stage)}.{part}.{suffix}"
        else:
            found=re.fullmatch(r"stages.([0-3]).([0-9]+).(dwconv|norm|pwconv1|pwconv2|gamma)(?:.(weight|bias))?",name)
            if found:
                stage,block,operation,suffix=found.groups()
                base=f"features.{2*int(stage)+1}.{block}"
                if operation=="gamma":
                    if suffix is not None or value.ndim!=1:
                        raise ValueError("Invalid layer scale")
                    target=base+".layer_scale"
                    value=value.reshape(-1,1,1)
                else:
                    if suffix is None:
                        raise ValueError("Missing block tensor suffix")
                    index={"dwconv":0,"norm":2,"pwconv1":3,"pwconv2":5}[operation]
                    target=f"{base}.block.{index}.{suffix}"
            else:
                found=re.fullmatch(r"norm.(weight|bias)",name)
                if not found:
                    raise ValueError("Unknown source tensor: "+name)
                target="classifier.0."+found.group(1)
        if target in mapped or target not in target_state:
            raise ValueError("Unexpected/duplicate mapped tensor: "+target)
        if tuple(value.shape)!=tuple(target_state[target].shape):
            raise ValueError("Tensor shape mismatch: "+name+" -> "+target)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite source tensor: "+name)
        mapped[target]=value
    if set(mapped)!=set(target_state):
        raise ValueError("Missing target tensors: "+str(sorted(set(target_state)-set(mapped))))
    return mapped


def load_22k_initializer(model, path, expected_sha256):
    """No network or unsafe globals; verify bytes before safe CPU deserialization."""
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}",expected_sha256) or digest!=expected_sha256:
        raise ValueError("Initializer SHA256 mismatch")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(payload,Mapping) or not isinstance(payload.get("model"),Mapping):
        raise ValueError("Official model state dictionary missing")
    mapped=convert_official_convnext_state(payload["model"],model.backbone.state_dict())
    model.backbone.load_state_dict(mapped,strict=True)
    return dict(source_sha256=digest,loaded_backbone_tensors=len(mapped),
                discarded_upstream_classifier=["head.weight","head.bias"],
                backend="unchanged_torchvision_convnext_tiny")
