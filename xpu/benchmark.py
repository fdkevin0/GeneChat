#!/usr/bin/env python3
"""XPU theoretical + empirical performance benchmarks for LoRA training & inference.

Covers:
  Group 1 — Hardware specs & theoretical peaks
  Group 2 — Micro-benchmarks (GEMM, attention, LoRA ops)
  Group 3 — Model-level forward/backward (Llama-3.1-8B QLoRA)
  Group 4 — Inference (prefill + token-by-token decode)
  Group 5 — Memory breakdown & throughput estimates

Usage:
  python xpu/benchmark.py                  # all benchmarks
  python xpu/benchmark.py --group 1        # hardware theory only
  python xpu/benchmark.py --group 2,3      # micro + model benchmarks
  python xpu/benchmark.py --quick          # skip long-running tests
"""

from __future__ import annotations

import gc
import sys
import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


# ═══════════════════════════════════════════════════════════════════════════
# Configuration — Llama-3.1-8B QLoRA (genechat_unsloth_stage2.yaml)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Llama-3.1-8B + DNABERT-2-117M architecture (genechat_unsloth_stage2.yaml)."""
    # ── LLM: Llama-3.1-8B ──
    n_layers: int       = 32
    hidden_size: int    = 4096
    intermediate_size: int = 14336
    n_heads: int        = 32
    n_kv_heads: int     = 8
    head_dim: int       = 128
    vocab_size: int     = 128256
    max_seq_len: int    = 2048
    max_txt_len: int    = 405       # answer text max length
    rope_theta: float   = 500_000.0

    # ── Gene Encoder: DNABERT-2-117M (frozen) ──
    gene_hidden: int    = 768
    gene_layers: int    = 12
    gene_heads: int     = 12
    gene_intermediate: int = 3072
    gene_max_len: int   = 512       # per-slice window
    gene_vocab: int     = 4096      # nucleotide k-mers

    # ── Adaptor: gene_hidden → llm_hidden (trainable) ──
    adaptor_hidden: int = 4096      # projection layer

    # ── QLoRA on LLM ──
    load_in_4bit: bool  = True
    lora_r: int         = 16
    lora_alpha: int     = 16
    lora_dropout: float = 0.0
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")

    # ── Training ──
    batch_size: int     = 1
    accum_steps: int    = 4
    lr: float           = 2e-4
    dtype: str          = "bfloat16"
    dataset_genes: int  = 47_275     # number of training genes


@dataclass
class HardwareSpec:
    """Intel Arc A770 theoretical peaks."""
    name: str           = "Intel Arc A770"
    vram_gb: float      = 16.0
    vram_bw_gbs: float  = 560.0      # memory bandwidth
    xmx_bf16_tflops: float = 138.0   # Xe Matrix Extensions peak BF16
    xmx_fp16_tflops: float = 138.0   # same hardware unit
    fp32_tflops: float  = 34.5       # ≈ XMX/4 for FP32 accumulate
    pcie_bw_gbs: float  = 15.75      # PCIe 4.0 x16 ≈ 15.75 GB/s per direction
    compute_units: int  = 32         # Xe-cores


CFG = ModelConfig()
HW  = HardwareSpec()

# Global results store — populated by each benchmark group, used by summary
_results: dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_xpu() -> torch.device:
    if not torch.xpu.is_available():
        sys.exit("❌ XPU not available. Source oneAPI setvars.sh and re-run.")
    return torch.device("xpu")


def _dtype() -> torch.dtype:
    return torch.bfloat16 if CFG.dtype == "bfloat16" else torch.float16


def _gb(bytes_val: int | float) -> str:
    return f"{bytes_val / 1024**3:.2f} GB"


def _tflops(ops: float, seconds: float) -> float:
    return ops / seconds / 1e12


def _bandwidth(bytes_moved: int | float, seconds: float) -> float:
    return bytes_moved / seconds / 1e9


def _warmup(fn, n: int = 5) -> None:
    for _ in range(n):
        fn()
    if torch.xpu.is_available():
        torch.xpu.synchronize()


def _bench(fn, n_warmup: int = 5, n_iter: int = 20) -> tuple[float, float]:
    """Run fn n_iter times, return (mean_sec, std_sec)."""
    _warmup(fn, n_warmup)
    if torch.xpu.is_available():
        torch.xpu.synchronize()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        if torch.xpu.is_available():
            torch.xpu.synchronize()
        times.append(time.perf_counter() - t0)
    arr = torch.tensor(times, dtype=torch.float64)
    return arr.mean().item(), arr.std().item()


def _separator(title: str = "", char: str = "═", width: int = 70) -> None:
    if title:
        print(f"\n{char * width}")
        print(f"  {title}")
        print(f"{char * width}")
    else:
        print(f"{'─' * width}")


def _label(name: str) -> str:
    return f"{name:<40s}"


# ═══════════════════════════════════════════════════════════════════════════
# Group 1 — Hardware specifications
# ═══════════════════════════════════════════════════════════════════════════

def bench_hardware() -> None:
    _separator("Group 1: Hardware Specifications")

    device = _ensure_xpu()

    # XPU device properties
    props = torch.xpu.get_device_properties(device)
    print(f"  Device       : {props.name}")
    print(f"  VRAM         : {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute units: {props.max_compute_units} (XMX: {hasattr(props, 'has_subgroup_matrix_multiply_accumulate') and props.has_subgroup_matrix_multiply_accumulate})")
    print(f"  L3 cache     : {props.last_level_cache_size / 1024**2:.0f} MB")
    print(f"  Driver ver   : {props.version}")

    # Memory info
    free, total = torch.xpu.mem_get_info()
    print(f"  VRAM free    : {free / 1024**3:.2f} GB")
    print(f"  VRAM total   : {total / 1024**3:.2f} GB")

    print(f"\n  ── Theoretical Peaks ({HW.name}) ──")
    print(f"  BF16/FP16 XMX : {HW.xmx_bf16_tflops:.0f} TFLOPS")
    print(f"  FP32          : {HW.fp32_tflops:.1f} TFLOPS")
    print(f"  Mem bandwidth : {HW.vram_bw_gbs:.0f} GB/s")
    print(f"  PCIe BW       : {HW.pcie_bw_gbs:.1f} GB/s")

    # Theoretical ops/byte ratio
    ops_per_byte = HW.xmx_bf16_tflops * 1e12 / (HW.vram_bw_gbs * 1e9)
    print(f"  Ops/byte      : {ops_per_byte:.0f} (compute-bound if >{ops_per_byte:.0f})")
    print(f"  Bytes/param   : ~0.5 (NF4 packed) → {HW.vram_bw_gbs/0.5:.0f} param-reads/s peak")

    # VRAM overhead
    reserved = total - free
    print(f"\n  ── VRAM Overhead ──")
    print(f"  Driver+PT reserve : {reserved / 1024**3:.2f} GB")
    print(f"  Available for model: {free / 1024**3:.2f} GB")


# ═══════════════════════════════════════════════════════════════════════════
# Group 2 — Micro-benchmarks (GEMM, Attention, LoRA ops)
# ═══════════════════════════════════════════════════════════════════════════

def _gemm_bench(m: int, n: int, k: int, label: str = "") -> dict:
    """Benchmark a single GEMM (M×K @ K×N → M×N) in BF16, return stats dict."""
    device = _ensure_xpu()
    dtype = _dtype()
    A = torch.randn(m, k, dtype=dtype, device=device)
    B = torch.randn(k, n, dtype=dtype, device=device)

    def run():
        # Use matmul (delegates to oneDNN/XMX on XPU)
        torch.matmul(A, B)

    mean_s, std_s = _bench(run, n_iter=30)
    flops = 2 * m * n * k  # multiply-add = 2 ops
    tflops = _tflops(flops, mean_s)
    bytes_moved = (A.numel() + B.numel() + m * n) * A.element_size()
    bw = _bandwidth(bytes_moved, mean_s)

    eff = tflops / HW.xmx_bf16_tflops * 100 if HW.xmx_bf16_tflops > 0 else 0

    print(f"  {_label(label)} {mean_s*1e3:7.2f} ms  "
          f"{tflops:6.2f} TFLOPS  ({eff:5.1f}% peak)  {bw:6.1f} GB/s")
    return {"name": label, "ms": mean_s * 1e3, "tflops": tflops, "eff_pct": eff, "bw_gbs": bw}


def bench_micro() -> None:
    _separator("Group 2: Micro-benchmarks")

    # ── GEMM at typical Llama sizes ──
    print()
    print("  GEMM (BF16 matmul, single op):")
    print(f"  {'Shape (M×K@K×N)':<40s} {'Time':>8s}  {'TFLOPS':>7s}  {'%Peak':>6s}  {'BW':>7s}")
    print(f"  {'─'*38}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*7}")

    results_gemm = []
    # Q/K/V projection: hidden × hidden  (batch=1, seq=1)
    results_gemm.append(_gemm_bench(1, CFG.hidden_size, CFG.hidden_size, "Q-proj 1×4096×4096"))
    # Attention: Q @ K^T  (single head)
    results_gemm.append(_gemm_bench(1, 1, CFG.head_dim, "Attn Q@K 1×128×128"))
    # Output projection per head
    results_gemm.append(_gemm_bench(1, CFG.head_dim, CFG.hidden_size, "O-proj 1×128×4096"))
    # Gate/Up projection (batch=1, seq=1)
    results_gemm.append(_gemm_bench(1, CFG.hidden_size, CFG.intermediate_size, "Gate 1×4096×14336"))
    # Down projection
    results_gemm.append(_gemm_bench(1, CFG.intermediate_size, CFG.hidden_size, "Down 1×14336×4096"))
    # LoRA B @ A (r=16)
    results_gemm.append(_gemm_bench(1, CFG.lora_r, CFG.hidden_size, "LoRA-B 1×16×4096"))
    # Full batch (2048 tokens) Q-proj
    results_gemm.append(_gemm_bench(CFG.max_seq_len, CFG.hidden_size, CFG.hidden_size,
                                    f"Q-proj {CFG.max_seq_len}×4096×4096"))
    # KV-cache sized attention: 2048 × 128
    results_gemm.append(_gemm_bench(1, CFG.max_seq_len, CFG.head_dim, "Attn 1×2048×128"))
    # Full batch gate/up
    results_gemm.append(_gemm_bench(CFG.max_seq_len, CFG.hidden_size, CFG.intermediate_size,
                                    f"Gate {CFG.max_seq_len}×4096×14336"))

    # ── Memory bandwidth test ──
    print(f"\n  ── Memory Bandwidth (device→device copy) ──")
    device = _ensure_xpu()
    for size_mb in [16, 64, 256, 1024]:
        n_bytes = size_mb * 1024 * 1024
        n_floats = n_bytes // 4
        src = torch.randn(n_floats, dtype=torch.float32, device=device)
        dst = torch.empty_like(src)

        def copy():
            dst.copy_(src)

        mean_s, _ = _bench(copy, n_iter=30)
        bw = _bandwidth(n_bytes * 2, mean_s)  # read + write
        print(f"  {_label(f'{size_mb} MB copy')} {mean_s*1e3:7.2f} ms  {bw:7.1f} GB/s  "
              f"({bw/HW.vram_bw_gbs*100:5.1f}% peak BW)")


# ═══════════════════════════════════════════════════════════════════════════
# Group 3 — Model-level Forward / Backward
# ═══════════════════════════════════════════════════════════════════════════

def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def bench_model_forward_backward() -> None:
    _separator("Group 3: Model Forward / Backward (Llama-3.1-8B QLoRA)")

    print("\n  Loading model (this may take a minute)...")

    # Apply XPU patches as in normal training flow
    from xpu.patches import apply_phase1_patches, apply_phase2_patches

    apply_phase1_patches()
    import unsloth
    apply_phase2_patches()

    from unsloth import FastLanguageModel

    device = _ensure_xpu()
    dtype = _dtype()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.1-8B-bnb-4bit",
        max_seq_length=CFG.max_seq_len,
        dtype=dtype,
        load_in_4bit=CFG.load_in_4bit,
        device_map={"": device},
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=list(CFG.lora_targets),
        use_gradient_checkpointing="unsloth",
    )

    total_p, trainable_p = _count_params(model)
    print(f"  Total params   : {total_p / 1e9:.2f}B")
    print(f"  Trainable (LoRA): {trainable_p / 1e6:.2f}M ({trainable_p / total_p * 100:.2f}%)")

    # Estimate memory
    print(f"\n  ── Memory Estimates ──")
    # NF4 weights: ~0.5 bytes/param for non-LoRA weights
    nf4_params = total_p - trainable_p
    nf4_mem_gb = nf4_params * 0.5 / 1024**3
    print(f"  NF4 base weights: ~{nf4_mem_gb:.2f} GB ({nf4_params/1e9:.1f}B params @ 4-bit)")

    # LoRA weights in BF16
    lora_mem_gb = trainable_p * 2 / 1024**3
    print(f"  LoRA weights (BF16): {lora_mem_gb:.3f} GB ({trainable_p/1e6:.1f}M params)")

    # Optimizer states (AdamW: param + exp_avg + exp_avg_sq = 3× in FP32 = 12 bytes/param)
    opt_mem_gb = trainable_p * 12 / 1024**3
    print(f"  Optimizer states  : {opt_mem_gb:.3f} GB (AdamW FP32, 12 bytes/param)")

    # Activations (rough: 34 * batch * seq * hidden * 2 bytes)
    act_mem_gb = 34 * CFG.batch_size * CFG.max_seq_len * CFG.hidden_size * 2 / 1024**3
    print(f"  Activations (est) : ~{act_mem_gb:.2f} GB (34× batch×seq×hidden, BF16)")

    total_est = nf4_mem_gb + lora_mem_gb + opt_mem_gb + act_mem_gb
    free_gb = torch.xpu.mem_get_info()[0] / 1024**3
    print(f"  Total estimated   : ~{total_est:.2f} GB")
    print(f"  Available VRAM    : ~{free_gb:.2f} GB")
    if total_est < free_gb:
        print(f"  ✓ Fits in VRAM (margin: {free_gb - total_est:.2f} GB)")
    else:
        print(f"  ⚠ May OOM (deficit: {total_est - free_gb:.2f} GB) — reduce seq_len or batch")

    # ── Forward pass benchmark ──
    print(f"\n  ── Forward Pass ──")
    input_ids = torch.randint(0, CFG.vocab_size, (CFG.batch_size, CFG.max_seq_len), device=device)
    attention_mask = torch.ones(CFG.batch_size, CFG.max_seq_len, device=device)

    def forward():
        with torch.no_grad():
            return model(input_ids=input_ids, attention_mask=attention_mask)

    mean_s, std_s = _bench(forward, n_warmup=3, n_iter=10)
    tokens_per_sec = CFG.max_seq_len / mean_s
    _results["fwd_ms"] = mean_s * 1e3
    _results["fwd_tok_per_s"] = tokens_per_sec
    print(f"  seq_len={CFG.max_seq_len}, bs={CFG.batch_size}")
    print(f"  Forward: {mean_s*1e3:.1f} ± {std_s*1e3:.1f} ms  ({tokens_per_sec:.1f} tok/s)")

    # ── Forward + Backward ──
    print(f"\n  ── Forward + Backward ──")
    model.train()

    def fwd_bwd():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                     labels=input_ids)
        out.loss.backward()

    # Only a few iterations — backward is expensive
    mean_s, std_s = _bench(fwd_bwd, n_warmup=2, n_iter=5)
    tokens_per_sec = CFG.max_seq_len / mean_s
    _results["fwd_bwd_ms"] = mean_s * 1e3
    _results["fwd_bwd_tok_per_s"] = tokens_per_sec
    print(f"  Fwd+Bwd: {mean_s*1e3:.1f} ± {std_s*1e3:.1f} ms  ({tokens_per_sec:.1f} tok/s)")

    # ── Optimizer step ──
    print(f"\n  ── Optimizer Step ──")
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    def opt_step():
        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                     labels=input_ids)
        out.loss.backward()
        optimizer.step()

    mean_s, std_s = _bench(opt_step, n_warmup=2, n_iter=5)
    _results["full_step_ms"] = mean_s * 1e3
    _results["full_step_tok_per_s"] = CFG.max_seq_len / mean_s
    _results["iters_per_s"] = 1.0 / mean_s
    print(f"  Full step: {mean_s*1e3:.1f} ± {std_s*1e3:.1f} ms  "
          f"({CFG.max_seq_len / mean_s:.1f} tok/s)")
    iters_per_sec = 1.0 / mean_s
    print(f"  Throughput: {iters_per_sec:.2f} it/s  "
          f"→ {iters_per_sec * CFG.accum_steps * CFG.max_seq_len:.0f} effective tok/s "
          f"(accum={CFG.accum_steps})")

    # Estimated training time
    print(f"\n  ── Training Time Estimates ──")
    steps_per_epoch = CFG.dataset_genes // (CFG.batch_size * CFG.accum_steps)
    _results["steps_per_epoch"] = steps_per_epoch
    print(f"  Dataset     : {CFG.dataset_genes:,} genes")
    print(f"  Steps/epoch : ~{steps_per_epoch:,} (bs={CFG.batch_size}, accum={CFG.accum_steps})")
    sec_per_step = mean_s
    min_per_epoch = steps_per_epoch * sec_per_step / 60
    _results["min_per_epoch"] = min_per_epoch
    _results["hours_per_epoch"] = min_per_epoch / 60
    print(f"  Time/epoch  : ~{min_per_epoch:.0f} min ({min_per_epoch/60:.1f} hr)")
    print(f"  Time/2 epochs: ~{min_per_epoch*2:.0f} min ({min_per_epoch*2/60:.1f} hr)")

    # Cleanup
    del model, input_ids, attention_mask, optimizer
    gc.collect()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# Group 4 — Inference
# ═══════════════════════════════════════════════════════════════════════════

def bench_inference(model=None, tokenizer=None):
    """Prefill + token-by-token decode benchmark."""
    _separator("Group 4: Inference")

    if model is None:
        print("\n  Loading model for inference benchmark...")
        from xpu.patches import apply_phase1_patches, apply_phase2_patches
        apply_phase1_patches()
        import unsloth
        apply_phase2_patches()
        from unsloth import FastLanguageModel

        device = _ensure_xpu()
        dtype = _dtype()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.1-8B-bnb-4bit",
            max_seq_length=CFG.max_seq_len,
            dtype=dtype,
            load_in_4bit=CFG.load_in_4bit,
            device_map={"": device},
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=CFG.lora_r,
            lora_alpha=CFG.lora_alpha,
            lora_dropout=CFG.lora_dropout,
            target_modules=list(CFG.lora_targets),
        )
        model.eval()
        _cleanup_needed = True
    else:
        model.eval()
        _cleanup_needed = False

    device = _ensure_xpu()

    # ── Prefill benchmark ──
    print(f"\n  ── Prefill (prompt encoding) ──")
    _prefill_results = {}
    for prompt_len in [128, 256, 512, 1024, 2048]:
        input_ids = torch.randint(0, CFG.vocab_size, (1, prompt_len), device=device)

        def prefill():
            with torch.no_grad():
                return model(input_ids=input_ids)

        mean_s, std_s = _bench(prefill, n_warmup=3, n_iter=10)
        tok_per_s = prompt_len / mean_s
        _prefill_results[prompt_len] = {"ms": mean_s * 1e3, "tok_per_s": tok_per_s}
        print(f"  {_label(f'prompt={prompt_len} tok')} {mean_s*1e3:7.1f} ms  "
              f"{tok_per_s:8.1f} tok/s  ({prompt_len/mean_s/1e3:.1f}k tok/s)")
    _results["prefill"] = _prefill_results

    # ── Token-by-token decode ──
    print(f"\n  ── Single-token Decode (seq_len=1, with KV cache) ──")
    # Build up a KV cache first
    prompt_len = 256
    past_key_values = None
    input_ids = torch.randint(0, CFG.vocab_size, (1, prompt_len), device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values

    # Now benchmark single-token decode
    new_token = torch.randint(0, CFG.vocab_size, (1, 1), device=device)

    def decode():
        with torch.no_grad():
            return model(input_ids=new_token, past_key_values=past_key_values, use_cache=True)

    # Warmup with actual decode to build any JIT caches
    _warmup(decode, n=5)

    # Run many iterations (token decode is fast)
    mean_s, std_s = _bench(decode, n_warmup=3, n_iter=50)
    tok_per_s = 1.0 / mean_s
    ms_per_token = mean_s * 1e3
    _results["decode_ms_per_tok"] = ms_per_token
    _results["decode_tok_per_s"] = tok_per_s
    print(f"  {_label(f'decode (KV={prompt_len})')} {ms_per_token:7.1f} ms/tok  "
          f"{tok_per_s:8.1f} tok/s  (interactive quality if {'<' if ms_per_token < 50 else '>'}50ms)")

    # ── Throughput at different batch sizes ──
    print(f"\n  ── Batch Decode Throughput ──")
    _batch_results = {}
    for bs in [1, 2, 4, 8]:
        batch_tokens = torch.randint(0, CFG.vocab_size, (bs, 1), device=device)

        def batch_decode():
            with torch.no_grad():
                return model(input_ids=batch_tokens)

        try:
            mean_s, _ = _bench(batch_decode, n_warmup=3, n_iter=20)
            tok_per_s = bs / mean_s
            _batch_results[bs] = {"ms": mean_s * 1e3, "tok_per_s": tok_per_s}
            print(f"  {_label(f'bs={bs}')} {mean_s*1e3:7.1f} ms  {tok_per_s:8.1f} tok/s")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {_label(f'bs={bs}')} {'OOM':>7s}")
                _batch_results[bs] = {"ms": float('inf'), "tok_per_s": 0}
                torch.xpu.empty_cache()
            else:
                raise
    _results["decode_batch"] = _batch_results

    # ── Inference Memory ──
    print(f"\n  ── Inference Memory ──")
    free, total = torch.xpu.mem_get_info()
    used = total - free
    _results["inf_vram_used_gb"] = used / 1024**3
    _results["inf_vram_free_gb"] = free / 1024**3
    print(f"  VRAM used (model loaded): {used / 1024**3:.2f} GB")
    # The LoRA adapters are tiny in memory
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA params: {n_lora / 1e6:.1f}M  → ~{n_lora * 2 / 1024**2:.1f} MB BF16")

    # Summary
    print(f"\n  ── Inference Summary (expected metrics) ──")
    pref_512 = _prefill_results.get(512, {"tok_per_s": 0})
    print(f"  Prefill (512 tok):  {pref_512['tok_per_s']:.0f} tok/s")
    print(f"  Decode (bs=1):      {ms_per_token:7.1f} ms/tok  →  {tok_per_s:.1f} tok/s")
    print(f"  Memory (model):     {used / 1024**3:.2f} GB")
    print(f"  Memory headroom:    {free / 1024**3:.2f} GB (for KV cache + activations)")

    if _cleanup_needed:
        del model
        gc.collect()
        torch.xpu.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# Group 5 — Summary & Estimated Metrics
# ═══════════════════════════════════════════════════════════════════════════

def bench_summary() -> None:
    _separator("Group 5: Expected Performance Summary")

    # Pull measured values (or use theoretical estimates if not run)
    fwd_ms      = _results.get("fwd_ms")
    fwd_bwd_ms  = _results.get("fwd_bwd_ms")
    step_ms     = _results.get("full_step_ms")
    decode_ms   = _results.get("decode_ms_per_tok")
    decode_tps  = _results.get("decode_tok_per_s")
    inf_vram    = _results.get("inf_vram_used_gb")
    prefill     = _results.get("prefill", {})
    batch_dec   = _results.get("decode_batch", {})
    steps_epoch = _results.get("steps_per_epoch", CFG.dataset_genes // (CFG.batch_size * CFG.accum_steps))
    hrs_epoch   = _results.get("hours_per_epoch")

    # Memory estimates (analytical)
    nf4_params   = 8.03e9   # Llama-3.1-8B
    lora_params  = sum([
        CFG.lora_r * CFG.hidden_size * 2,         # Q (A + B)
        CFG.lora_r * CFG.hidden_size * 2,         # K
        CFG.lora_r * CFG.hidden_size * 2,         # V
        CFG.lora_r * CFG.hidden_size * 2,         # O
        CFG.lora_r * CFG.intermediate_size * 2,   # gate
        CFG.lora_r * CFG.intermediate_size * 2,   # up
        CFG.lora_r * CFG.hidden_size * 2,         # down
    ]) * CFG.n_layers
    nf4_gb       = nf4_params * 0.5 / 1024**3
    lora_gb      = lora_params * 2 / 1024**3       # BF16
    opt_gb       = lora_params * 12 / 1024**3       # AdamW FP32 (param + m + v)
    act_gb       = 34 * CFG.batch_size * CFG.max_seq_len * CFG.hidden_size * 2 / 1024**3
    kv_gb        = (2 * CFG.n_kv_heads * CFG.head_dim * CFG.max_seq_len * 2
                    * CFG.n_layers / 1024**3)
    total_est_gb = nf4_gb + lora_gb + opt_gb + act_gb + kv_gb

    # Format helper
    def _v(val, fmt=".1f", default="  N/A"):
        if val is None:
            return default
        return f"{val:{fmt}}"

    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │              LoRA Training Expected Metrics (A770, BS=1)            │
  ├─────────────────────────────────────────────────────────────────────┤
  │  Model   : Llama-3.1-8B (NF4) + DNABERT-2-117M (frozen)           │
  │  LoRA    : r={CFG.lora_r}, α={CFG.lora_alpha}, 7 target modules              │
  │  Trainable: {lora_params/1e6:.1f}M params ({lora_params/nf4_params*100:.2f}% of {nf4_params/1e9:.2f}B)               │
  │  Optimizer: AdamW FP32 (β₁=0.9, β₂=0.999)                          │
  │                                                                     │
  │  ── Memory Breakdown (analytical) ──                                │
  │    NF4 base weights    : {nf4_gb:5.2f} GB                                    │
  │    LoRA weights (BF16) : {lora_gb:5.2f} GB                                    │
  │    Optimizer states    : {opt_gb:5.2f} GB (AdamW FP32)                      │
  │    Activations (seq={CFG.max_seq_len}) : {act_gb:5.2f} GB (34× layers, BF16)               │
  │    KV cache ({CFG.max_seq_len} tok)     : {kv_gb:5.2f} GB (GQA {CFG.n_heads}/{CFG.n_kv_heads})                       │
  │    ─────────────────────────────────────────                        │
  │    Total estimated     : {total_est_gb:5.2f} GB                                    │
  │    A770 available      : ~14.5 GB (after driver overhead)          │
  │    Headroom            : {14.5 - total_est_gb:5.2f} GB                                    │
  │                                                                     │
  │  ── Measured Performance ──                                         │
  │    Forward pass  (seq={CFG.max_seq_len}): {_v(fwd_ms, '7.0f')} ms  →  {_v(_results.get('fwd_tok_per_s'), '7.0f')} tok/s                   │
  │    Fwd+Bwd       (seq={CFG.max_seq_len}): {_v(fwd_bwd_ms, '7.0f')} ms  →  {_v(_results.get('fwd_bwd_tok_per_s'), '7.0f')} tok/s                   │
  │    Full opt step (seq={CFG.max_seq_len}): {_v(step_ms, '7.0f')} ms  →  {_v(_results.get('iters_per_s'), '5.2f')} it/s                    │
  │                                                                     │
  │  ── Training Time Estimate ({CFG.dataset_genes:,} genes) ──                         │
  │    Steps/epoch (bs=1, accum={CFG.accum_steps}): ~{steps_epoch:,}                          │
  │    Time/epoch       : {_v(hrs_epoch, '.1f')} hr                                    │
  │    Time/2 epochs    : {_v(hrs_epoch * 2 if hrs_epoch else None, '.1f')} hr                                    │
  ├─────────────────────────────────────────────────────────────────────┤
  │              Inference Expected Metrics (A770)                       │
  ├─────────────────────────────────────────────────────────────────────┤
  │  ── Prefill ──                                                      │
  │    128 tok prompt : {_v(prefill.get(128, {}).get('ms'), '5.0f')} ms    ({_v(prefill.get(128, {}).get('tok_per_s'), '6.0f')} tok/s)                     │
  │    512 tok prompt : {_v(prefill.get(512, {}).get('ms'), '5.0f')} ms    ({_v(prefill.get(512, {}).get('tok_per_s'), '6.0f')} tok/s)                     │
  │   2048 tok prompt : {_v(prefill.get(2048, {}).get('ms'), '5.0f')} ms    ({_v(prefill.get(2048, {}).get('tok_per_s'), '6.0f')} tok/s)                     │
  │                                                                     │
  │  ── Token-by-token Decode ──                                        │
  │    Latency (bs=1) : {_v(decode_ms, '5.1f')} ms/tok  →  {_v(decode_tps, '5.1f')} tok/s                         │
  │    Batch=4        : {_v(batch_dec.get(4, {}).get('ms'), '5.0f')} ms      ({_v(batch_dec.get(4, {}).get('tok_per_s'), '6.0f')} tok/s)                     │
  │    Batch=8        : {_v(batch_dec.get(8, {}).get('ms'), '5.0f')} ms      ({_v(batch_dec.get(8, {}).get('tok_per_s'), '6.0f')} tok/s)                     │
  │                                                                     │
  │  ── Memory ──                                                       │
  │    Model loaded    : {_v(inf_vram, '.1f')} GB                                      │
  │    Per-token KV    : ~0.13 MB/tok (GQA, 2 bytes/elem)              │
  │    Max context est.: ~{int((14.5 - (inf_vram or 5)) / 0.13 / 1024 * 1000) if inf_vram else 'N/A':>5s} tokens (before OOM)                   │
  └─────────────────────────────────────────────────────────────────────┘
""")

    # ── Bottleneck Analysis ──
    _separator("Bottleneck Analysis")
    nf4_bytes = nf4_params * 0.5
    time_mem_bound = nf4_bytes / (HW.vram_bw_gbs * 1e9)   # BW in bytes/s
    nf4_read_gb = nf4_bytes / 1024**3

    print(f"  {'─' * 60}")
    print(f"  NF4 model size  : {nf4_read_gb:.2f} GB")
    print(f"  A770 BW         : {HW.vram_bw_gbs:.0f} GB/s")
    print(f"  Mem-bound fwd   : {time_mem_bound*1e3:.0f} ms (theoretical min, read {nf4_read_gb:.1f} GB NF4 weights)")
    if fwd_ms:
        print(f"  Measured fwd    : {fwd_ms:.0f} ms")
        print(f"  BW efficiency   : {time_mem_bound*1e3/fwd_ms*100:.0f}% (relative to peak BW)")
    print(f"")
    # For single-token decode: ops ≈ 2*hidden_size, bytes ≈ 0.5*hidden_size
    # ops/byte ≈ 4 vs peak 246 → deeply memory-bound
    ops_per_byte_decode = 2 * CFG.hidden_size / (0.5 * CFG.hidden_size)  # ≈ 4
    print(f"")
    print(f"  Bottleneck      : {'MEMORY BANDWIDTH' if ops_per_byte_decode < HW.xmx_bf16_tflops*1e12/(HW.vram_bw_gbs*1e9) else 'COMPUTE'}")
    print(f"  Decode ops/byte : {ops_per_byte_decode:.0f} (peak balance: {HW.xmx_bf16_tflops*1e12/(HW.vram_bw_gbs*1e9):.0f})")
    print(f"  → Single-token decode requires >{HW.xmx_bf16_tflops*1e12/(HW.vram_bw_gbs*1e9):.0f}x more compute per byte to be compute-bound")
    print(f"  → NF4 4-bit inference is bandwidth-bound on A770 for bs≤~16")

    # ── Cross-device comparison ──
    print(f"\n  {'─' * 60}")
    print(f"  Cross-device comparison (theoretical BW-limited forward):")
    for name, bw in [("A100 80GB", 2000), ("RTX 4090 24GB", 1000), ("A770 16GB", HW.vram_bw_gbs)]:
        t = nf4_bytes / (bw * 1e9) * 1e3  # ms to read NF4 weights
        print(f"    {name:<18s} @ {bw:5.0f} GB/s  →  ~{t:6.0f} ms forward")

    # ── Key Takeaways ──
    print(f"\n  {'─' * 60}")
    print(f"  Key Takeaways:")
    print(f"    1. LoRA training on A770 is memory-bandwidth bound")
    print(f"    2. NF4 dequantize + GEMM is the critical path (fast_gemv)")
    print(f"    3. LoRA adds negligible compute (~{lora_params*2/1e9:.2f}B FLOPs vs {2*8.03:.0f}B dense)")
    print(f"    4. Gradient checkpointing trades ~20% compute for ~50% VRAM savings")
    print(f"    5. Inference decode latency is dominated by memory reads, not compute")
    print(f"    6. Batch inference scales sub-linearly (BW-bound, not compute-bound)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="XPU LoRA Training & Inference Benchmark")
    p.add_argument("--group", type=str, default="",
                   help="Comma-separated group numbers to run (e.g. '1,2'). Default: all.")
    p.add_argument("--quick", action="store_true",
                   help="Skip model-level benchmarks (groups 3,4)")
    args = p.parse_args()

    groups = set()
    if args.group:
        for g in args.group.split(","):
            groups.add(int(g.strip()))
    else:
        groups = {1, 2, 3, 4, 5}

    if args.quick:
        groups -= {3, 4}

    print("╔" + "═" * 68 + "╗")
    print("║  XPU LoRA Training & Inference Benchmark — Intel Arc A770" + " " * 11 + "║")
    print("║  Model: Llama-3.1-8B QLoRA (r=16)  |  Batch=1  |  BF16" + " " * 12 + "║")
    print("╚" + "═" * 68 + "╝")

    if 1 in groups:
        bench_hardware()

    if 2 in groups:
        bench_micro()

    if 3 in groups:
        bench_model_forward_backward()

    if 4 in groups:
        bench_inference()

    if 5 in groups:
        bench_summary()

    print(f"\n{'═' * 70}")
    print("  Benchmark complete. Use these metrics as expected baselines")
    print("  for experiment tracking in wandb / tensorboard.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
