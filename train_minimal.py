#!/usr/bin/env python3
"""Minimal training loop — bypasses the complex runner infrastructure.
Validates that the full GeneChat+Unsloth pipeline works end-to-end on XPU.
"""
from __future__ import annotations
import os, sys, gc

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: XPU patches + DNABERT-2 pre-load (consolidated — see gcu_xpu.py)
# ═══════════════════════════════════════════════════════════════════════
import torch
from gcu_xpu import (
    apply_phase1_patches,
    apply_phase2_patches,
    patch_dnabert2_alibi,
    patch_dnabert2_flash_attn,
)
apply_phase1_patches()

# Auto-patch + pre-load DNABERT-2
from transformers import AutoConfig, AutoModel, AutoTokenizer

_gene_config = AutoConfig.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
if not hasattr(_gene_config, "pad_token_id"): _gene_config.pad_token_id = 0
# Disable flash attention — only works on CUDA, not XPU
_gene_config._attn_implementation = "eager"

try:
    _gene_encoder = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M", config=_gene_config, trust_remote_code=True)
except RuntimeError: pass

patch_dnabert2_alibi()
patch_dnabert2_flash_attn()

_gene_encoder = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M", config=_gene_config, trust_remote_code=True)
_gene_tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# Store pre-loaded
import genechat.models.genechat_unsloth as _gcu
_gcu._PRELOADED_GENE_ENCODER = _gene_encoder
_gcu._PRELOADED_GENE_TOKENIZER = _gene_tokenizer
_gene_encoder = _gene_tokenizer = None; gc.collect()

# ═══════════════════════════════════════════════════════════════════════
# Import genechat → unsloth
# ═══════════════════════════════════════════════════════════════════════
from genechat.common.config import Config
import genechat.tasks as tasks
from genechat.datasets.builders import *
from genechat.models import *
from genechat.runners import *
from genechat.tasks import *

# Phase 2 patches (consolidated — see gcu_xpu.py), plus extras this script
# needs that aren't part of the shared set (autocast/GradScaler mocks,
# memory-stat stubs — runner_iter/base_task don't need these, but the
# raw optimizer loop below does).
apply_phase2_patches()
if torch.xpu.is_available():
    torch.cuda.amp.autocast = lambda dtype=torch.bfloat16, enabled=True: torch.autocast(
        device_type="xpu", dtype=dtype, enabled=enabled)

    class _N:
        def scale(self, l): return l
        def step(self, o): o.step()
        def update(self): pass
        def get_scale(self): return 1.0
        def state_dict(self): return {}
        def load_state_dict(self, s): pass
    torch.cuda.amp.GradScaler = _N
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.memory_stats = lambda *a, **k: {}

# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print(f"torch: {torch.__version__}  xpu: {torch.xpu.is_available()}")
    print("=" * 60)

    # 1. Build model
    cfg = Config(type("Args", (), {"cfg_path": "configs/genechat_unsloth_stage2.yaml", "options": None})())
    model = tasks.setup_task(cfg).build_model(cfg)
    device = next(model.parameters()).device
    print(f"Model on device: {device}")
    model.train()

    # 2. Load one batch of mock data
    import json, random
    with open("data/train/qa_summary_rule.json") as f:
        rules = json.load(f)
    with open("data/train/seq.json") as f:
        seqs = json.load(f)

    # Pick 1 sample
    sample = random.choice(rules)
    gene_id = str(sample["Gene Id"])
    seq = seqs.get(gene_id, ["ATCG" * 128])

    batch = {
        "seq": [seq],  # list of seq strings
        "text_input": [sample["Summary"]],
        "prompt": [f"###Human: <gene>{gene_id}<geneHere></gene> Tell me about this gene. ###Assistant:"],
    }

    print(f"Gene: {gene_id}, seq len: {len(seq[0])}, answer len: {len(sample['Summary'])}")

    # 3. Forward + backward
    print("Starting training loop (3 steps)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for step in range(3):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        xpu_mem = torch.xpu.max_memory_allocated() / 1024**3 if torch.xpu.is_available() else 0
        print(f"  step {step+1}/3  loss={loss.item():.4f}  peak_mem={xpu_mem:.1f}GB")

    print("=" * 60)
    print("MINIMAL TRAINING LOOP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
