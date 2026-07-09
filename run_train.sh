#!/usr/bin/env bash
# Launcher script for GeneChat Unsloth XPU training
cd /home/fdkevin/Workspaces/msc_project/GeneChat
source .venv/bin/activate
exec python train_unsloth.py --cfg-path configs/genechat_unsloth_stage2.yaml 2>&1 | tee /tmp/train_output.log
