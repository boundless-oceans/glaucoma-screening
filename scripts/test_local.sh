#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "glaucoma-cloud" ]]; then
  CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
  if [[ -z "$CONDA_BIN" ]]; then
    echo "Activate glaucoma-cloud before running this script" >&2
    exit 1
  fi
  CONDA_BASE="$("$CONDA_BIN" info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate glaucoma-cloud
fi
exec python -u -m glaucoma_training.evaluate_fusion "$@"
