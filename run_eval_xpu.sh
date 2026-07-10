#!/usr/bin/env bash
# GeneChat eval/generate entry — Intel Arc (XPU).
# Pass eval args through, e.g.:
#   ./run_eval_xpu.sh --checkpoint outputs/unsloth_checkpoints/<job>/checkpoint_1000.pth --out preds.json
set -euo pipefail
cd "$(dirname "$0")"

export GENECHAT_DEVICE=xpu
source "${VENV:-.venv}/bin/activate"

exec python scripts/eval_generate.py "$@"
