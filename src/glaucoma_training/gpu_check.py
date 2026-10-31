"""Check actual GPU forward/backward without downloading pretrained weights."""
import sys
import argparse
import torch
import torchvision
from .common import FusionModel
from .model_config import load_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config")
args = parser.parse_args()
config = load_config(args.config)

print("Python:", sys.version, flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("torchvision:", torchvision.__version__, flush=True)
print("PyTorch CUDA:", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("GPU unavailable. This is expected in no-GPU mode; retry after normal GPU startup.")
print("GPU:", torch.cuda.get_device_name(0), flush=True)
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), flush=True)
print("Model:", config["model_name"], flush=True)
model = FusionModel(pretrained=False, model_name=config["model_name"], use_age=config["use_age"]).cuda()
model(torch.randn(2, 3, config["image_size"], config["image_size"], device="cuda"),
      torch.zeros(2, 2, device="cuda")).sum().backward()
torch.cuda.synchronize()
print("GPU forward/backward OK", flush=True)
