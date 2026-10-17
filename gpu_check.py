"""Check actual GPU forward/backward without downloading pretrained weights."""
import sys
import torch
import torchvision

print("Python:", sys.version, flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("torchvision:", torchvision.__version__, flush=True)
print("PyTorch CUDA:", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("GPU unavailable. This is expected in no-GPU mode; retry after normal GPU startup.")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), flush=True)
model = torchvision.models.resnet18(weights=None).cuda()
model(torch.randn(2, 3, 224, 224, device="cuda")).sum().backward()
torch.cuda.synchronize()
print("GPU forward/backward OK", flush=True)
