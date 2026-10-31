"""CPU-only traditional ROI preview entry point."""
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if __name__ == "__main__":
    runpy.run_module("glaucoma_training.preview_classic_roi", run_name="__main__")
