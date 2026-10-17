"""Generate a full test submission from a completed validation checkpoint."""
import argparse
from pathlib import Path

from .common import predict_checkpoint
from . import DEFAULT_DATA_ROOT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    predict_checkpoint(args.checkpoint, args.data_root, args.output, args.device,
                       args.batch_size, args.workers, args.threshold)
