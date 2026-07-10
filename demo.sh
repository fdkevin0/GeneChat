#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --master_port 29501 scripts/inference_all.py
