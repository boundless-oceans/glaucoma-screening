"""Model/config metadata helpers; no PyTorch import or device allocation."""
import json
from pathlib import Path

from . import CONFIG_DIR

SUPPORTED_MODELS = ("resnet18", "efficientnet_b2")


def model_name_from_config(config):
    # Existing version-1 checkpoints predate model_name and contain ResNet18.
    name = config.get("model_name", "resnet18")
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name {name!r}; choose one of {SUPPORTED_MODELS}")
    return name


def checkpoint_model_name(state):
    config_name = model_name_from_config(state["config"])
    declared = state.get("model_name", config_name)
    if declared != config_name:
        raise ValueError("Checkpoint model_name disagrees with its saved config")
    return config_name


def canonical_config(config):
    return {**config, "model_name": model_name_from_config(config)}


def default_run_name(model_name, smoke=False):
    model_name_from_config({"model_name": model_name})
    if model_name == "resnet18":
        return "smoke" if smoke else "baseline"
    return model_name + ("_smoke" if smoke else "")


def load_config(path=None):
    defaults = json.loads((CONFIG_DIR / "train.json").read_text(encoding="utf-8-sig"))
    overrides = json.loads(Path(path or CONFIG_DIR / "train.json").read_text(encoding="utf-8-sig"))
    if not isinstance(overrides, dict):
        raise ValueError("Config must be a JSON object")
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return canonical_config({**defaults, **overrides})
