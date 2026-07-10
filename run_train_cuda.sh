#!/usr/bin/env bash
# GeneChat training entry — NVIDIA (CUDA).
# Override the venv with:  VENV=/path/to/venv ./run_train_cuda.sh
set -euo pipefail
cd "$(dirname "$0")"

export GENECHAT_DEVICE=cuda
source "${VENV:-.venv}/bin/activate"

exec python scripts/train_unsloth.py \
    --cfg-path configs/genechat_unsloth_stage2.yaml "$@" 2>&1 | tee /tmp/train_output.log
