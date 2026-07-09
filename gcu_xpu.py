"""
Consolidated XPU patches for Intel Arc GPUs (A770, B580, etc.).

All genuine hardware/library bugs and third-party compatibility workarounds
live here in ONE place. GeneChat entry scripts that need XPU support just:

    from gcu_xpu import apply_phase1_patches
    apply_phase1_patches()       # before import unsloth
    import unsloth
    from gcu_xpu import apply_phase2_patches
    apply_phase2_patches()       # after import unsloth

Bug-fix reference (17 bugs documented):
  #1-2: config / model attr issues (fixed inline in config + model)
  #3: DNABERT-2 ALiBi meta-device (patch_dnabert2_alibi)
  #4: DNABERT-2 flash_attn CUDA-only (patch_dnabert2_flash_attn)
  #5-7: dtype / embed / autocast (fixed inline in model)
  #8: _cuda_isCurrentStreamCapturing (phase 2, cached reference)
  #9-11: Stream mocking (phase 2, PrefetchLoader)
  #12: move_to_cuda (fixed in device.py, phase 2 fallback)
  #13: dist.barrier (phase 2)
  #14: wandb mock (fixed inline in train_unsloth)
  #15: gpt_oss mem_get_info (handled by bug #17)
  #16: caching_allocator_warmup (phase 1)
  #17: torch.xpu.mem_get_info not implemented on A770 (phase 1, dual-path)
  #18: device_map="auto" memory check (fixed in model with device_map dict)
"""
from __future__ import annotations

import glob as _glob
import os
import sys
from typing import Any

import torch


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Pre-Unsloth patches (genuine bugs + hardware limitations)
# ===================================================================
# These must run BEFORE `import unsloth`. They fix real PyTorch/
# library/hardware bugs that no device constant can solve.
# ═══════════════════════════════════════════════════════════════════════

_PHASE1_APPLIED = False


def apply_phase1_patches() -> None:
    """Apply pre-unsloth patches. Idempotent — safe to call multiple times."""
    global _PHASE1_APPLIED
    if _PHASE1_APPLIED:
        return
    _PHASE1_APPLIED = True

    # Bug #16 + #3: Disable torch.compile — breaks DNABERT-2 ALiBi init
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    torch._dynamo.config.suppress_errors = True
    try:
        torch._dynamo.disable()
    except Exception:
        pass

    if not torch.xpu.is_available():
        print("ℹ️  XPU not available, skipping Phase 1 patches")
        return

    # Bug: triton unavailable on XPU; importing it triggers _lazy_init → crash
    import torch.utils._triton as _triton
    _triton.is_device_compatible_with_triton = lambda: False

    # Bug: bitsandbytes may access CUDA symbols unconditionally during import
    if not hasattr(torch._C, "_cuda_getCurrentRawStream"):
        torch._C._cuda_getCurrentRawStream = lambda index=None: None

    # Bug #17: A770 doesn't implement _xpu_getMemoryInfo. unsloth_zoo uses
    # BOTH torch.xpu.memory.mem_get_info (gpt_oss at import) and
    # torch.xpu.mem_get_info (cross_entropy_loss at forward). Mock both.
    import torch.xpu.memory as _xpu_mem
    _mem_get_info_mock: Any = lambda device=0: (4 * 1024**3, 16 * 1024**3)
    _xpu_mem.mem_get_info = _mem_get_info_mock
    torch.xpu.mem_get_info = _mem_get_info_mock

    # Bug #16: >4GB alloc OOM on A770 — no-op caching_allocator_warmup
    import transformers.modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **k: None

    print("✅ Phase 1 patches applied (pre-unsloth)")


# ═══════════════════════════════════════════════════════════════════════
# DNABERT-2 cache file patching
# ═══════════════════════════════════════════════════════════════════════

_DNABERT2_CACHE = os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/"
    "zhihan1996/DNABERT_hyphen_2_hyphen_117M"
)


def patch_dnabert2_alibi() -> None:
    """Patch DNABERT-2 bert_layers.py: fix device=None → cpu in ALiBi code.

    Bug #3: PyTorch 2.12 torch.compile fake tensors crash when device=None
    is passed to tensor.expand().
    """
    files = _glob.glob(f"{_DNABERT2_CACHE}/*/bert_layers.py")
    if not files:
        print("⚠️  DNABERT-2 cache not found, skipping ALiBi patch")
        return

    with open(files[0]) as f:
        content = f.read()

    target = (
        "):\n        # Alibi\n"
        "        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5"
    )
    replacement = (
        "):\n        # Alibi\n        if device is None:\n            device = torch.device(\"cpu\")\n"
        "        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5"
    )

    if "if device is None:\n            device = torch.device" in content:
        print("✅ DNABERT-2 ALiBi patch already applied")
        return

    if target in content:
        content = content.replace(target, replacement)
        with open(files[0], "w") as f:
            f.write(content)
        print("✅ DNABERT-2 bert_layers.py patched (device=None → cpu)")
    else:
        print("⚠️  Could not find patch target in bert_layers.py")


def patch_dnabert2_flash_attn() -> None:
    """Patch DNABERT-2 bert_layers.py: disable triton flash_attn on XPU.

    Bug #4: flash_attn checks q.is_cuda before using triton, but triton
    is already disabled. On XPU the check fails silently → crash.
    """
    files = _glob.glob(f"{_DNABERT2_CACHE}/*/bert_layers.py")
    if not files:
        return

    with open(files[0]) as f:
        content = f.read()

    if "flash_attn_qkvpacked_func = None" in content:
        return  # Already patched

    old = "from flash_attn import flash_attn_qkvpacked_func"
    new = "flash_attn_qkvpacked_func = None  # patched for XPU"
    if old in content:
        content = content.replace(old, new)
        with open(files[0], "w") as f:
            f.write(content)
        print("✅ DNABERT-2 flash_attn disabled (XPU)")
    else:
        # Already patched or not present — fine
        pass


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Post-Unsloth patches (third-party library compatibility)
# ===================================================================
# These run AFTER `import unsloth`. They patch torch.cuda.* for
# transformers/accelerate/bitsandbytes compatibility (third-party code
# that calls torch.cuda.* directly, bypassing gcu_device.py). They do NOT
# duplicate gcu_device.py — genechat's own code should use that module
# instead.
# ═══════════════════════════════════════════════════════════════════════

_PHASE2_APPLIED = False


def _patch_unsloth_dequantize_xpu() -> None:
    """Bug #19: unsloth calls bitsandbytes C dequantize fns with raw pointers.

    bitsandbytes 0.49.2 has no XPU native lib — the ErrorHandlerMock raises
    RuntimeError on call.  Replace the three module-level symbols in
    unsloth.kernels.utils with PyTorch-tensor implementations that match
    the C functions' semantics.

    We cannot recover tensors from raw ctypes pointers, so we walk up one
    stack frame into fast_dequantize and use its local variables directly.
    This is a monkey-patch — remove once bitsandbytes ships a compiled XPU
    native library.
    """
    try:
        import unsloth.kernels.utils as _uku
    except ImportError:
        return

    if getattr(_uku, "DEVICE_TYPE", "") != "xpu":
        return

    import torch as _torch

    # NF4 asymmetric lookup table (matching bitsandbytes C library).
    # Indices 0-6: negative, 7: zero, 8-247: zeros (padding),
    # 248-255: positive.  The 4-bit index (0-15) is used directly.
    _NF4_TABLE = _torch.tensor([
        -1.0000000000, -0.6961928010, -0.5250730515, -0.3949174881,
        -0.2844413817, -0.1847734302, -0.0910500363,  0.0000000000,
         0.0795802996,  0.1609302014,  0.2461123019,  0.3379152417,
         0.4407098293,  0.5626170039,  0.7229568362,  1.0000000000,
    ])

    def _dequant_blockwise_fp32(
        code_ptr, A_ptr, absmax_ptr, out_ptr,
        blocksize, n_elements, stream,
    ) -> None:
        """Pure-PyTorch: dequantize 8-bit blockwise → float32 (vectorized)."""
        import sys as _sys
        frame = _sys._getframe(1)
        locs = frame.f_locals
        code2 = locs["code2"]
        absmax_q = locs["absmax"]
        absmax2_s = locs["absmax2"]
        out_absmax = locs["out_absmax"]
        blocksize2 = locs["blocksize2"]

        # Vectorized: reshape absmax_q to [n_blocks, blocksize], index, scale
        n_blk2 = absmax_q.numel() // blocksize2
        absmax_q_r = absmax_q.view(n_blk2, blocksize2)
        out_absmax_r = out_absmax.view(n_blk2, blocksize2)
        out_absmax_r.copy_(code2[absmax_q_r.long()] * absmax2_s.view(-1, 1))
        out_absmax.add_(locs["offset"])

    def _dequant_blockwise_nf4(
        code_ptr, A_ptr, absmax_ptr, out_ptr,
        blocksize, n_elements, stream,
    ) -> None:
        """Vectorized NF4 dequantize for fp16/bf16 output."""
        import sys as _sys
        frame = _sys._getframe(1)
        locs = frame.f_locals
        W = locs["W"]
        out = locs["out"]
        out_absmax = locs["out_absmax"]
        blocksize = locs["blocksize"]
        dtype = locs["dtype"]
        n_blk = out.numel() // blocksize
        half_bs = blocksize // 2

        nf4_tbl = _NF4_TABLE.to(device=W.device, dtype=_torch.float32)

        # Reshape packed weights to [n_blk, half_bs] for blockwise processing
        W_r = W.view(n_blk, half_bs)
        out_r = out.view(n_blk, blocksize)

        # Unpack 4-bit: interleave hi/lo nibbles
        hi = nf4_tbl[(W_r >> 4).long()]
        lo = nf4_tbl[(W_r & 0x0F).long()]

        # Interleave: [n_blk, half_bs*2]
        out_flat_inter = _torch.empty(n_blk, blocksize, dtype=_torch.float32, device=W.device)
        out_flat_inter[:, 0::2] = hi
        out_flat_inter[:, 1::2] = lo

        # Scale by per-block absmax
        scales = out_absmax.view(n_blk, -1).float()[:, :1]
        out_r.copy_(out_flat_inter * scales)

        if out.dtype != dtype:
            out.copy_(out.to(dtype))

    _dequant_blockwise_nf4_fp16 = _dequant_blockwise_nf4
    _dequant_blockwise_nf4_bf16 = _dequant_blockwise_nf4

    _uku.cdequantize_blockwise_fp32 = _dequant_blockwise_fp32
    _uku.cdequantize_blockwise_fp16_nf4 = _dequant_blockwise_nf4
    _uku.cdequantize_blockwise_bf16_nf4 = _dequant_blockwise_nf4
    print("  ✅ unsloth dequantize patched (PyTorch fallback for XPU)")


def apply_phase2_patches() -> None:
    """Apply post-unsloth patches. Idempotent."""
    global _PHASE2_APPLIED
    if _PHASE2_APPLIED:
        return
    _PHASE2_APPLIED = True

    if not torch.xpu.is_available():
        print("ℹ️  XPU not available, skipping Phase 2 patches")
        return

    import torch.xpu.memory as _xpu_mem

    # ── Third-party compatibility: transformers/accelerate check these ──
    torch.cuda.is_available = lambda: True

    class _MockDeviceProps:
        major = 8; minor = 0; multi_processor_count = 64
        total_memory = 16 * 1024 ** 3
    torch.cuda.get_device_properties = lambda dev=None: _MockDeviceProps()
    torch.cuda.get_device_capability = lambda dev=None: (8, 0)

    # Accelerate checks mem_get_info for device_map
    import torch.cuda.memory as _cuda_mem
    _cuda_mem.mem_get_info = lambda device=0: (4 * 1024**3, 16 * 1024**3)

    # Redirect core device APIs for third-party code paths
    torch.cuda.current_device = torch.xpu.current_device
    torch.cuda.device_count = torch.xpu.device_count
    for _name in ("set_device", "current_stream", "synchronize",
                  "reset_peak_memory_stats"):
        if hasattr(torch.xpu, _name):
            setattr(torch.cuda, _name, getattr(torch.xpu, _name))

    # ── Stream mocking (dataloader PrefetchLoader uses torch.cuda.Stream) ──
    class _MockStream:
        def wait_stream(self, *a, **k): pass
        def record_stream(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    torch.cuda.Stream = _MockStream
    torch.cuda.stream = lambda s=None: _MockStream()
    torch.cuda.current_stream = lambda d=None: _MockStream()

    # ── Tensor-level redirects ──
    torch.Tensor.record_stream = lambda self, s: None
    torch.Tensor.cuda = lambda self, *a, **k: self.to("xpu", *a, **k)

    # ── Bug #8: module-level cached reference ──
    import torch.cuda.graphs as _graphs
    _graphs._cuda_isCurrentStreamCapturing = lambda: False

    # ── move_to_cuda → xpu (data_utils still has legacy paths) ──
    try:
        import genechat.datasets.data_utils as _gdu
        _gdu.move_to_cuda = lambda sample: _gdu.apply_to_sample(
            lambda t: t.to("xpu"), sample)
    except ImportError:
        pass

    # ── dist.barrier guard (called unconditionally in runner_iter) ──
    import torch.distributed as _dist
    _orig_barrier = _dist.barrier
    _dist.barrier = lambda *a, **k: (
        _orig_barrier(*a, **k) if _dist.is_initialized() else None
    )

    # ── Bug #19: unsloth fast_dequantize calls bitsandbytes C lib ──
    # bitsandbytes 0.49.2 has no XPU native lib; the mock raises
    # RuntimeError. Patch to use PyTorch tensor ops instead of raw pointers.
    _patch_unsloth_dequantize_xpu()

    print("✅ Phase 2 patches applied (post-unsloth: third-party compat)")
