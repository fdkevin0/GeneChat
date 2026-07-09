#!/usr/bin/env python3
"""Minimal training loop — bypasses the complex runner infrastructure.
Validates that the full GeneChat+Unsloth pipeline works end-to-end on XPU.
"""
from __future__ import annotations
import os, sys, gc

# ═══════════════════════════════════════════════════════════════════════
# Phase 0-1: XPU patches + DNABERT-2 pre-load (same as train_unsloth.py)
# ═══════════════════════════════════════════════════════════════════════
os.environ["TORCH_COMPILE_DISABLE"] = "1"
import torch
import torch._dynamo; torch._dynamo.config.suppress_errors = True

if torch.xpu.is_available():
    import torch.utils._triton as _tr; _tr.is_device_compatible_with_triton = lambda: False
    if not hasattr(torch._C, "_cuda_getCurrentRawStream"):
        torch._C._cuda_getCurrentRawStream = lambda index=None: None

import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None

# Auto-patch + pre-load DNABERT-2
from transformers import AutoConfig, AutoModel, AutoTokenizer
import glob as _glob

_gene_config = AutoConfig.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True)
if not hasattr(_gene_config, "pad_token_id"): _gene_config.pad_token_id = 0
# Disable flash attention — only works on CUDA, not XPU
_gene_config._attn_implementation = "eager"

_dnabert2_cache = os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/"
    "zhihan1996/DNABERT_hyphen_2_hyphen_117M"
)
try:
    _gene_encoder = AutoModel.from_pretrained(
        "zhihan1996/DNABERT-2-117M", config=_gene_config, trust_remote_code=True)
except RuntimeError: pass

_dnabert2_files = _glob.glob(f"{_dnabert2_cache}/*/bert_layers.py")
if _dnabert2_files:
    with open(_dnabert2_files[0]) as f: _content = f.read()
    patched = False
    # Fix 1: ALiBi device=None → cpu
    if "if device is None:\n            device = torch.device" not in _content:
        _old_a = "):\n        # Alibi\n        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5"
        _new_a = "):\n        # Alibi\n        if device is None:\n            device = torch.device(\"cpu\")\n        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5"
        if _old_a in _content:
            _content = _content.replace(_old_a, _new_a); patched = True
    # Fix 2: disable flash_attn on XPU (checks q.is_cuda)
    _old_f = "    from .flash_attn_triton import flash_attn_qkvpacked_func\n"
    _new_f = "    flash_attn_qkvpacked_func = None  # disabled for XPU\n"
    if _old_f in _content:
        _content = _content.replace(_old_f, _new_f); patched = True
    if patched:
        with open(_dnabert2_files[0], "w") as f: f.write(_content)
        print("✅ DNABERT-2 patched (device=None + flash_attn disabled)")

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

# Phase 2 patches
if torch.xpu.is_available():
    class _P: major=8; minor=0; multi_processor_count=64; total_memory=16*1024**3
    torch.cuda.get_device_properties = lambda d=None: _P
    torch.cuda.get_device_capability = lambda d=None: (8, 0)
    import torch.cuda.memory as _cm; _cm.mem_get_info = lambda d=0: (0, 16*1024**3)
    torch.cuda.is_available = lambda: True
    torch.cuda.current_device = torch.xpu.current_device
    torch.cuda.device_count = torch.xpu.device_count
    for n in ("set_device","current_stream","synchronize"):
        if hasattr(torch.xpu,n): setattr(torch.cuda,n,getattr(torch.xpu,n))
    class _S:
        def wait_stream(self,*a,**k): pass
        def record_stream(self,*a,**k): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
    if not hasattr(torch.cuda,"Stream"): torch.cuda.Stream = _S
    torch.cuda.amp.autocast = lambda dtype=torch.bfloat16,enabled=True: torch.autocast(device_type="xpu",dtype=dtype,enabled=enabled)
    class _N:
        def scale(self,l): return l
        def step(self,o): o.step()
        def update(self): pass
        def get_scale(self): return 1.0
        def state_dict(self): return {}
        def load_state_dict(self,s): pass
    torch.cuda.amp.GradScaler = _N
    torch.cuda.max_memory_allocated = lambda *a,**k: 0
    torch.cuda.memory_stats = lambda *a,**k: {}
    torch.Tensor.record_stream = lambda self,s: None
    # XPU optimizer step — patch the captured reference in graphs module
    import torch.cuda.graphs as _graphs
    _graphs._cuda_isCurrentStreamCapturing = lambda: False

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
