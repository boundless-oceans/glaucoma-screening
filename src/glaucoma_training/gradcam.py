"""Positive-logit Grad-CAM on a frozen FusionModel. Server model diagnostic only."""
import torch
from torch.nn import functional as F


class GradCam:
    def __init__(self, model):
        self.model = model
        self.activation = None
        if model.model_name == "efficientnet_b2":
            self.target = model.backbone.features[-1]
        elif model.model_name == "resnet18":
            self.target = model.backbone.layer4
        else:
            raise ValueError("Unsupported Grad-CAM architecture")
        self.handle = None

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("GradCam context is already active")
        self.flags = [(p, p.requires_grad) for p in self.model.parameters()]
        self.modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        for parameter, _ in self.flags:
            parameter.requires_grad_(False)

        def capture(module, inputs, output):
            # Only the score-to-feature graph is needed; no backbone backward or
            # parameter gradients. The returned leaf keeps the following head differentiable.
            self.activation = output.detach().requires_grad_(True)
            return self.activation

        self.handle = self.target.register_forward_hook(capture)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove(); self.handle = None
        self.activation = None
        for parameter, flag in self.flags:
            parameter.requires_grad_(flag)
        for module, mode in self.modes:
            module.training = mode

    def __call__(self, image, age):
        if self.handle is None:
            raise RuntimeError("Use GradCam as a context manager")
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("Grad-CAM preview uses one image at a time")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Grad-CAM requires autograd, not inference_mode")
        try:
            # Always explain the positive-class logit, irrespective of true/predicted label.
            # FP32 is a diagnostic choice, not the official AMP prediction pipeline.
            with torch.enable_grad(), torch.autocast(device_type=image.device.type, enabled=False):
                logits = self.model(image.float(), age.float())
                if self.activation is None:
                    raise RuntimeError("Target activation was not captured")
                gradient, = torch.autograd.grad(logits.sum(), self.activation)
                weights = gradient.float().mean(dim=(2, 3), keepdim=True)
                heat = torch.relu((weights * self.activation.float()).sum(dim=1, keepdim=True))
                heat = F.interpolate(heat, size=image.shape[-2:], mode="bilinear", align_corners=False)
                probability = float(logits.detach().float().sigmoid().cpu()[0])
                result = heat[0, 0].detach().cpu().numpy().copy()
            return result, probability
        finally:
            self.activation = None
