#!/usr/bin/env python3
"""
Training entry-point for GeneChat + Unsloth + Llama 3.1 8B on Intel Arc A770 (XPU).

Applies all XPU patches documented in:
  [[Intel Arc A770 XPU 微调踩坑手册]]
  [[ChatLoRA（unsloth + Qwen3.5-4B）]]

Usage:
  cd /home/fdkevin/Workspaces/msc_project/GeneChat
  source .venv/bin/activate
  python train_unsloth.py --cfg-path configs/genechat_unsloth_stage2.yaml

Critical ordering:
  Phase 1 (pre-unsloth): disable triton → let unsloth detect XPU naturally
  Phase 2 (post-unsloth): redirect torch.cuda → torch.xpu for training infra
"""
from __future__ import annotations

import os
import sys

import torch

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: Pre-unsloth patches (consolidated — see genechat/common/xpu_patches.py)
# ═══════════════════════════════════════════════════════════════════════
from gcu_xpu import (
    apply_phase1_patches,
    patch_dnabert2_alibi,
    patch_dnabert2_flash_attn,
)
apply_phase1_patches()

# ═══════════════════════════════════════════════════════════════════════
# 1.5  Pre-load DNABERT-2 BEFORE unsloth (avoids meta-device crash)
# ═══════════════════════════════════════════════════════════════════════
from transformers import AutoConfig, AutoModel, AutoTokenizer

_gene_config = AutoConfig.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
if not hasattr(_gene_config, "pad_token_id"):
    _gene_config.pad_token_id = 0
_gene_config._attn_implementation = "eager"  # XPU: no flash attention

# Download + patch DNABERT-2 cached source files
try:
    _gene_encoder = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M", config=_gene_config, trust_remote_code=True)
except RuntimeError:
    pass  # Expected crash on first download — code is now cached

patch_dnabert2_alibi()
patch_dnabert2_flash_attn()

# Now load with patched code
_gene_encoder = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M", config=_gene_config, trust_remote_code=True)
_gene_tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
print(f"✅ DNABERT-2 loaded: {sum(p.numel() for p in _gene_encoder.parameters())/1e6:.1f}M params")

# Store pre-loaded models where genechat_unsloth can find them
import genechat.models.genechat_unsloth as _gcu
_gcu._PRELOADED_GENE_ENCODER = _gene_encoder
_gcu._PRELOADED_GENE_TOKENIZER = _gene_tokenizer

# ═══════════════════════════════════════════════════════════════════════
# 2.  Import genechat → triggers unsloth import (XPU detection works!)
# ═══════════════════════════════════════════════════════════════════════
import random
import argparse

import numpy as np
import torch.backends.cudnn as cudnn

import genechat.tasks as tasks
from genechat.common.config import Config
from genechat.common.dist_utils import get_rank, init_distributed_mode
from genechat.common.logger import setup_logger
from genechat.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from genechat.common.registry import registry
from genechat.common.utils import now

# Register sub-modules (imports GeneChatUnsloth which imports unsloth)
from genechat.datasets.builders import *   # noqa: F401, F403
from genechat.models import *               # noqa: F401, F403
from genechat.runners import *              # noqa: F401, F403
from genechat.tasks import *                # noqa: F401, F403

# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Post-unsloth patches (consolidated)
# ═══════════════════════════════════════════════════════════════════════
from gcu_xpu import apply_phase2_patches
apply_phase2_patches()


def parse_args():
    parser = argparse.ArgumentParser(description="GeneChat Unsloth XPU Training")
    parser.add_argument(
        "--cfg-path",
        default="configs/genechat_unsloth_stage2.yaml",
        help="path to configuration file",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override config values in xxx=yyy format",
    )
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    return registry.get_runner_class(cfg.run_cfg.get("runner", "runner_iter"))


def main():
    job_id = now()
    cfg = Config(parse_args())
    init_distributed_mode(cfg.run_cfg)
    print("Distributed mode initialized")
    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # wandb (optional)
    wandb_run = None
    try:
        import wandb
        wandb_run = wandb.init(
            project="GeneChat-Unsloth",
            name=f"llama31-8b-{job_id[:8]}",
            config=cfg.to_dict(),
        )
    except Exception:
        # base_task._train_inner_loop calls wandb.log() unconditionally
        class _MockWandb:
            def log(self, *a, **k): pass
            def finish(self, *a, **k): pass
        wandb_run = _MockWandb()
        print("wandb not available, training without logging")

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets,
        wandb=wandb_run,
    )
    runner.train()

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    print("=" * 60)
    print("XPU Training Environment:")
    print(f"  torch:     {torch.__version__}")
    print(f"  xpu avail: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"  xpu count: {torch.xpu.device_count()}")
        print(f"  xpu name:  {torch.xpu.get_device_name(0)}")
    print(f"  device:    {'xpu' if torch.xpu.is_available() else 'cpu'}")
    print("=" * 60)

    main()
