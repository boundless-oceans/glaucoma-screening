"""Glaucoma training application. Importing the package does not load PyTorch."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATA_ROOT = Path("/root/autodl-tmp/glaucoma_first")
