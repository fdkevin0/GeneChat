"""
XPU patches for Intel Arc GPUs — import before/after unsloth:

    from gcu_xpu import apply_phase1_patches
    apply_phase1_patches()       # before import unsloth
    import unsloth
    from gcu_xpu import apply_phase2_patches
    apply_phase2_patches()       # after import unsloth

Patches (verified by xpu_test.py):
  Phase 1:  #19 bnb .so deploy, #16 caching_allocator_warmup, triton disable
            #3 DNABERT-2 ALiBi, #4 DNABERT-2 flash_attn
  Phase 2:  #9-11 Stream mock, #12 tensor.cuda redirect,
            #8 _cuda_isCurrentStreamCapturing, #13 dist.barrier guard,
            #20 triton backend prune
"""
from __future__ import annotations

import glob as _glob
import os
import sys

import torch


# ═══════════════════════════════════════════════════════════════════════
# Bug #19: bnb .so links libsycl.so.8, env has libsycl.so.9 → ErrorHandlerMock.
# Fix: deploy oneAPI-2026 rebuild. Rebuild once:
#   cd bitsandbytes && source /opt/intel/oneapi/setvars.sh
#   cmake -B build . -DCOMPUTE_BACKEND=xpu && cmake --build build
# ═══════════════════════════════════════════════════════════════════════

_BNB_XPU_LIB_SOURCE = str(
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "bitsandbytes" / "bitsandbytes" / "libbitsandbytes_xpu.so"
)


def _find_bnb_package_dir() -> str | None:
    """Locate the installed bitsandbytes package directory."""
    import site
    for base in site.getsitepackages():
        pkg = os.path.join(base, "bitsandbytes")
        if os.path.isdir(pkg):
            return pkg
    # Fallback: search sys.path
    for p in sys.path:
        pkg = os.path.join(p, "bitsandbytes")
        if os.path.isdir(pkg):
            return pkg
    return None


def deploy_bnb_xpu_lib() -> bool:
    """Replace stock bitsandbytes XPU .so with the oneAPI-2026 rebuild.

    Call BEFORE ``import bitsandbytes`` (i.e. before ``import unsloth``).
    Idempotent — skips if already deployed or source unavailable.
    Returns True if a replacement was performed.
    """
    import shutil
    if not os.path.exists(_BNB_XPU_LIB_SOURCE):
        return False
    pkg_dir = _find_bnb_package_dir()
    if pkg_dir is None:
        return False
    dest = os.path.join(pkg_dir, "libbitsandbytes_xpu.so")
    # Idempotent: same size → already deployed
    if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(_BNB_XPU_LIB_SOURCE):
        return False
    shutil.copy2(_BNB_XPU_LIB_SOURCE, dest)
    print("✅ bnb XPU lib deployed (oneAPI 2026 rebuild)")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Pre-Unsloth patches
# ===================================================================
# These must run BEFORE `import unsloth`.
# ═══════════════════════════════════════════════════════════════════════

_PHASE1_APPLIED = False


def apply_phase1_patches() -> None:
    """Apply pre-unsloth patches.  Idempotent."""
    global _PHASE1_APPLIED
    if _PHASE1_APPLIED:
        return
    _PHASE1_APPLIED = True

    if not torch.xpu.is_available():
        print("ℹ️  XPU not available, skipping Phase 1 patches")
        return

    # Bug #19: deploy oneAPI-2026 bnb .so before any bnb import
    deploy_bnb_xpu_lib()

    # triton unavailable on XPU; importing it → _lazy_init crash
    import torch.utils._triton as _triton
    _triton.is_device_compatible_with_triton = lambda: False

    # Bug #16: >4GB alloc OOM on A770
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


# ── Bug #3: DNABERT-2 ALiBi passes device=None → tensor.expand() crash ──
def patch_dnabert2_alibi() -> None:
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
    elif target in content:
        with open(files[0], "w") as f:
            f.write(content.replace(target, replacement))
        print("✅ DNABERT-2 bert_layers.py patched (device=None → cpu)")
    else:
        print("⚠️  Could not find patch target in bert_layers.py")


# ── Bug #4: flash_attn is CUDA-only, XPU has no implementation ──
def patch_dnabert2_flash_attn() -> None:
    files = _glob.glob(f"{_DNABERT2_CACHE}/*/bert_layers.py")
    if not files:
        return
    with open(files[0]) as f:
        content = f.read()
    if "flash_attn_qkvpacked_func = None" in content:
        return
    old = "from flash_attn import flash_attn_qkvpacked_func"
    new = "flash_attn_qkvpacked_func = None  # patched for XPU"
    if old in content:
        with open(files[0], "w") as f:
            f.write(content.replace(old, new))
        print("✅ DNABERT-2 flash_attn disabled (XPU)")


# ═══════════════════════════════════════════════════════════════════════
# Phase 2 — mock torch.cuda.* APIs that third-party code calls at runtime
#
# Libraries (accelerate 1.14, transformers 5.13) now detect XPU natively —
# is_available / get_device_properties / device_count are NOT mocked.
# Only mock functional CUDA APIs that have no XPU equivalent:
#   Stream     — dataloader PrefetchLoader
#   tensor.cuda — dataloader batch transfer
#   dist.barrier — called unconditionally in runner_iter
#   _cuda_isCurrentStreamCapturing — optimizer
#   triton backends — amd+nvidia+intel → only intel
# ═══════════════════════════════════════════════════════════════════════

_PHASE2_APPLIED = False


def apply_phase2_patches() -> None:
    """Apply post-unsloth patches.  Idempotent."""
    global _PHASE2_APPLIED
    if _PHASE2_APPLIED:
        return
    _PHASE2_APPLIED = True

    if not torch.xpu.is_available():
        print("ℹ️  XPU not available, skipping Phase 2 patches")
        return

    # Bug #9-11: dataloader PrefetchLoader creates torch.cuda.Stream
    class _MockStream:
        def wait_stream(self, *a, **k): pass
        def record_stream(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    torch.cuda.Stream = _MockStream
    torch.cuda.stream = lambda s=None: _MockStream()
    torch.cuda.current_stream = lambda d=None: _MockStream()

    # Bug #12: dataloader calls tensor.cuda() / tensor.record_stream(stream)
    torch.Tensor.cuda = lambda self, *a, **k: self.to("xpu", *a, **k)
    torch.Tensor.record_stream = lambda self, s: None

    # Bug #8: optimizer checks _cuda_isCurrentStreamCapturing
    import torch.cuda.graphs as _graphs
    _graphs._cuda_isCurrentStreamCapturing = lambda: False

    # Bug #13: dist.barrier() called unconditionally in runner_iter
    import torch.distributed as _dist
    _orig_barrier = _dist.barrier
    _dist.barrier = lambda *a, **k: (
        _orig_barrier(*a, **k) if _dist.is_initialized() else None
    )

    # Bug #20: plain triton + triton-xpu → multiple backends → crash
    _patch_triton_single_backend_xpu()

    print("✅ Phase 2 patches applied (post-unsloth)")


def _patch_triton_single_backend_xpu() -> None:
    """Bug #20: plain triton (amd+nvidia backends) co-installed with triton-xpu
    (intel backend).  Multiple backends → RuntimeError at kernel launch.
    Prune to intel-only.  Must mutate in-place (triton.runtime.driver holds
    a reference to the same dict).
    """
    try:
        import triton.backends as _tb
    except ImportError:
        return
    if "intel" in _tb.backends and set(_tb.backends) != {"intel"}:
        be = _tb.backends["intel"]
        _tb.backends.clear()
        _tb.backends["intel"] = be
        print("  ✅ triton backend registry pruned to intel-only (XPU)")
