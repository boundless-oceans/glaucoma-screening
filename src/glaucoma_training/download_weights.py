"""Download official initialization weights on the server, without GPU execution."""
import argparse

from .model_config import SUPPORTED_MODELS, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    model = args.model or load_config()["model_name"]
    from torchvision.models import EfficientNet_B2_Weights, EfficientNet_B0_Weights, ResNet18_Weights
    weights = {"efficientnet_b2":EfficientNet_B2_Weights.DEFAULT, "efficientnet_b0":EfficientNet_B0_Weights.DEFAULT,
               "resnet18":ResNet18_Weights.DEFAULT}[model]
    print(f"Downloading official {model} weights (no GPU execution)...", flush=True)
    weights.get_state_dict(progress=True, check_hash=True)
    print("Weights cached successfully.", flush=True)


if __name__ == "__main__":
    main()
