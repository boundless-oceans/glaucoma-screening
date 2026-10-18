"""Check uploaded CSVs and paths. No PyTorch import, GPU use or model training."""
import argparse
import json
from pathlib import Path

from .data_utils import age_value, fit_age, read_rows
from .run_utils import validate_config
from . import CONFIG_DIR, DEFAULT_DATA_ROOT
from .model_config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "train.json")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    print(f"Model: {config['model_name']}; image_size={config['image_size']}; batch_size={config['batch_size']}")
    root = args.data_root.resolve()
    expected = {"train": 51396, "valid": 5849, "test": 5855}
    for split in ("train", "valid", "test"):
        rows = read_rows(root, split)
        missing_age = sum(age_value(row["Age"]) is None for row in rows)
        print(f"{split}: {len(rows)} rows, every referenced image exists; invalid/missing ages={missing_age}")
        if len(rows) != expected[split]:
            print(f"WARNING: expected official {split} count {expected[split]}")
        if split != "test":
            positive = sum(row["True_Label"] == "1" for row in rows)
            if not 0 < positive < len(rows):
                raise ValueError(f"{split} must contain both classes")
            print(f"  positive={positive}, negative={len(rows) - positive}")
        if split == "train":
            print("  training-only age stats:", fit_age(rows))
    print("Preflight OK. This checks names/CSV/config only; it does not decode images or test CUDA.")


if __name__ == "__main__":
    main()
