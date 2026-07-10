#!/usr/bin/env bash
# GeneChat training entry — Intel Arc (XPU).
# Override the venv on a different machine with:  VENV=/path/to/venv ./run_train_xpu.sh
set -euo pipefail
cd "$(dirname "$0")"

export GENECHAT_DEVICE=xpu
source "${VENV:-.venv}/bin/activate"

exec python scripts/train_unsloth.py \
    --cfg-path configs/genechat_unsloth_stage2.yaml "$@" 2>&1 | tee /tmp/train_output.log
