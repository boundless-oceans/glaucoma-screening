"""Versioned preprocessing options, shared by configs and old checkpoints."""
import math

PREPROCESS_DEFAULTS = {
    "remove_black_border": False,
    "black_border_threshold": 8,
    "black_border_margin": 0.02,
    "black_border_version": 1,
}


def preprocessing_options(config):
    options = {key: config.get(key, default) for key, default in PREPROCESS_DEFAULTS.items()}
    if type(options["remove_black_border"]) is not bool:
        raise ValueError("remove_black_border must be true or false")
    threshold = options["black_border_threshold"]
    if type(threshold) is not int or not 0 <= threshold <= 32:
        raise ValueError("black_border_threshold must be an integer within [0, 32]")
    margin = options["black_border_margin"]
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin <= 0.1:
        raise ValueError("black_border_margin must be within [0, 0.1]")
    if type(options["black_border_version"]) is not int or options["black_border_version"] != 1:
        raise ValueError("Unsupported black_border_version")
    return options
