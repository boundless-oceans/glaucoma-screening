"""SERVER-only Grad-CAM integration check. Synthetic inputs, no pretrained download."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import numpy as np
    import torch
    from torch import nn
    from glaucoma_training.common import FusionModel
    from glaucoma_training.gradcam import GradCam
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_name = "efficientnet_b2"
            self.backbone = nn.Module()
            self.backbone.features = nn.Sequential(nn.Conv2d(3, 2, 1, bias=False))
            self.head = nn.Linear(4, 1, bias=False)
            with torch.no_grad():
                self.backbone.features[0].weight.zero_()
                self.backbone.features[0].weight[0, 0, 0, 0] = 1
                self.backbone.features[0].weight[1, 1, 0, 0] = 1
                self.head.weight.copy_(torch.tensor([[2., -1., 0., 0.]]))

        def forward(self, image, age):
            features = self.backbone.features(image).mean((2, 3))
            return self.head(torch.cat([features, age], 1)).squeeze(1)

    toy = Toy().to(device)
    image = torch.tensor([[[[1., 2.], [3., 4.]], [[2., 1.], [4., 2.]], [[0., 0.], [0., 0.]]]], device=device)
    age = torch.zeros((1, 2), device=device)
    with GradCam(toy) as cam:
        heat, _ = cam(image, age)
        np.testing.assert_allclose(heat, [[0., .75], [.5, 1.5]], atol=1e-6)
    assert toy.training and all(p.requires_grad for p in toy.parameters())
    print("Known-gradient mathematical check passed.", flush=True)
    for name in ("resnet18", "efficientnet_b2"):
        torch.manual_seed(42)
        model = FusionModel(model_name=name).to(device)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
        target = model.backbone.layer4 if name == "resnet18" else model.backbone.features[-1]
        hook_count = len(target._forward_hooks)
        tensor = torch.rand((1, 3, 64, 64), device=device)
        with GradCam(model) as cam:
            first, probability = cam(tensor, age)
            second, _ = cam(tensor, age)
            assert first.shape == (64, 64) and np.isfinite(first).all() and 0 <= probability <= 1
            np.testing.assert_allclose(first, second, atol=1e-6)
            assert not model.training and all(p.grad is None for p in model.parameters())
        assert model.training and len(target._forward_hooks) == hook_count
        assert all(p.requires_grad for p in model.parameters())
        assert all(torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items())
        print(f"{name}: heatmap, repeatability, parameters/BN buffers and hook cleanup passed.", flush=True)
        del model
    print("Local ROI model integration passed. No training or pretrained downloads.")


if __name__ == "__main__":
    main()
