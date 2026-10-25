"""Small runtime guards. Importing this module never imports a training framework."""
import atexit
import math
import os
from pathlib import Path
from .model_config import model_name_from_config
from .preprocess_config import preprocessing_options


def validate_config(config):
    if type(config.get("finetune", False)) is not bool:
        raise ValueError("finetune must be true or false")
    model_name_from_config(config)
    preprocessing = preprocessing_options(config)
    for key in ("image_size", "batch_size", "epochs", "save_every_steps", "cpu_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("num_workers", "train_limit", "valid_limit", "test_limit", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("pretrained", "amp"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in ("learning_rate", "weight_decay", "pos_weight"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        if value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(f"Invalid {key}")
    if config["image_size"] < 64:
        raise ValueError("image_size must be >= 64")
    if preprocessing["jpeg_draft_size"] and preprocessing["jpeg_draft_size"] < config["image_size"]:
        raise ValueError("jpeg_draft_size must be at least image_size")


def protect_output(root, output, directory=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or (directory and output in root.parents):
        raise ValueError("Output must not replace/contain the dataset root")
    for split in ("train", "valid", "test"):
        folder = root / split
        if output == folder or folder in output.parents or output == root / f"{split}.csv":
            raise ValueError("Output must not overwrite dataset files or image directories")
    return output


class RunLock:
    """OS-level lock, automatically released even after an abrupt process exit."""
    def __init__(self, run):
        self.stream = (Path(run) / ".run.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                if not self.stream.read(1):
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise RuntimeError("Another training process is using this run directory") from error
        atexit.register(self.close)

    def close(self):
        if not self.stream.closed:
            self.stream.close()
