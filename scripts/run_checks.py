"""Run lightweight checks with a workspace-local writable temporary directory."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    temporary_root = ROOT / "outputs" / "test_tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary_root)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if "torch" in sys.modules:
        raise RuntimeError("Lightweight checks unexpectedly imported PyTorch")
    raise SystemExit(not result.wasSuccessful())
