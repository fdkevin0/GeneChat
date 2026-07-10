#!/usr/bin/env bash
# GeneChat eval/generate entry — NVIDIA (CUDA).
# Pass eval args through, e.g.:
#   ./run_eval_cuda.sh --checkpoint outputs/unsloth_checkpoints/<job>/checkpoint_1000.pth --out preds.json
set -euo pipefail
cd "$(dirname "$0")"

export GENECHAT_DEVICE=cuda
source "${VENV:-.venv}/bin/activate"

exec python scripts/eval_generate.py "$@"
