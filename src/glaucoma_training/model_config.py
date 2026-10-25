"""Model/config metadata helpers; no PyTorch import or device allocation."""
import json
from pathlib import Path

from . import CONFIG_DIR
from .preprocess_config import preprocessing_options

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
    return {**config, "model_name": model_name_from_config(config), **preprocessing_options(config),
            "finetune": config.get("finetune", False)}


def default_run_name(model_name, smoke=False, remove_black_border=False, finetune=False, jpeg_draft_size=0):
    preprocessing_options({"jpeg_draft_size": jpeg_draft_size})
    if jpeg_draft_size:
        return default_run_name(model_name, smoke, remove_black_border, finetune) + f"_draft{jpeg_draft_size}"
    if finetune:
        return default_run_name(model_name, smoke, remove_black_border) + "_finetune"
    model_name_from_config({"model_name": model_name})
    if remove_black_border:
        return model_name + "_blackborder" + ("_smoke" if smoke else "")
    if model_name == "resnet18":
        return "smoke" if smoke else "baseline"
    return model_name + ("_smoke" if smoke else "")


def load_config(path=None):
    defaults = json.loads((CONFIG_DIR / "train.json").read_text(encoding="utf-8-sig"))
    overrides = json.loads(Path(path or CONFIG_DIR / "train.json").read_text(encoding="utf-8-sig"))
    if not isinstance(overrides, dict):
        raise ValueError("Config must be a JSON object")
    defaults.setdefault("finetune", False)
    defaults.setdefault("jpeg_draft_size", 0)
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return canonical_config({**defaults, **overrides})
