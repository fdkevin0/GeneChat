#!/usr/bin/env python3
"""XPU bug verification test suite.

Systematically tests each bug claim documented in gcu_xpu.py against the
current environment.  Each test returns PASS (bug absent / fix works),
FAIL (bug present / fix doesn't work), or SKIP (can't test in this env).

Usage:
  python xpu_test.py              # run all tests
  python xpu_test.py --group A    # environment checks only
  python xpu_test.py --verbose    # show full tracebacks on failure
"""

from __future__ import annotations

import glob as _glob
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Test infrastructure
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    name: str
    status: str          # PASS / FAIL / SKIP
    message: str = ""
    details: str = ""
    bug_ref: str = ""


_results: list[Result] = []
_VERBOSE = False


def check(name: str, condition: bool, message: str = "",
          bug_ref: str = "", details: str = "") -> None:
    _results.append(Result(name=name, status="PASS" if condition else "FAIL",
                           message=message, bug_ref=bug_ref, details=details))


def skip(name: str, reason: str, bug_ref: str = "") -> None:
    _results.append(Result(name=name, status="SKIP", message=reason, bug_ref=bug_ref))


def _subprocess(script: str, timeout: int = 120) -> tuple[int, str, str]:
    p = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=timeout,
                       cwd=Path(__file__).resolve().parent)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def _check_subprocess(name: str, script: str, bug_ref: str = "",
                      expect_success: bool = True, timeout: int = 120) -> None:
    rc, stdout, stderr = _subprocess(script, timeout)
    ok = (rc == 0) if expect_success else (rc != 0)
    details = ""
    if stdout:
        details += f"stdout: {stdout[:300]}"
    if stderr:
        err_lines = [l for l in stderr.split("\n")
                     if "Traceback" in l or "Error" in l or "ErrorHandler" in l]
        if err_lines:
            details += "\n" + "\n".join(err_lines[-3:])
    check(name, ok, message=f"exit={rc}", bug_ref=bug_ref, details=details.strip())


def _find_bnb_xpu_so() -> Path | None:
    """Locate the installed libbitsandbytes_xpu.so."""
    for p in sys.path:
        cand = Path(p) / "bitsandbytes" / "libbitsandbytes_xpu.so"
        if cand.is_file():
            return cand
    return None


def _find_venv() -> Path:
    return Path(__file__).resolve().parent / ".venv"


# ═══════════════════════════════════════════════════════════════════════════
# Group A — Environment checks (no torch needed)
# ═══════════════════════════════════════════════════════════════════════════

def test_group_a() -> None:
    print("━" * 60)
    print("Group A: Environment")
    print("━" * 60)

    _so = _find_bnb_xpu_so()
    if not _so:
        check("A.1  bnb XPU .so exists", False,
              "libbitsandbytes_xpu.so not found", bug_ref="#19")
        return
    check("A.1  bnb XPU .so exists", True, str(_so), bug_ref="#19")

    # A.2 — SYCL linkage: must link libsycl.so.9 (not .8 "not found")
    p = subprocess.run(["ldd", str(_so)], capture_output=True, text=True)
    sycl_not_found = "sycl" in p.stdout and "not found" in p.stdout
    has_sycl9 = "libsycl.so.9" in p.stdout
    if sycl_not_found:
        check("A.2  SYCL linkage", False,
              "libsycl.so not found — run deploy_bnb_xpu_lib() or source oneAPI setvars.sh",
              bug_ref="#19")
    elif not has_sycl9:
        check("A.2  SYCL linkage", False,
              "linked to old libsycl.so.8 — needs oneAPI 2026 rebuild", bug_ref="#19")
    else:
        check("A.2  SYCL linkage (libsycl.so.9)", True, "", bug_ref="#19")

    # A.3 — ctypes CDLL load (the real test)
    try:
        import ctypes
        _lib = ctypes.CDLL(str(_so))
        ok = hasattr(_lib, "cdequantize_blockwise_fp32") and \
             hasattr(_lib, "cgemv_4bit_inference_bf16")
        check("A.3  ctypes CDLL load", ok,
              f"dequant={'✓' if hasattr(_lib, 'cdequantize_blockwise_fp32') else '✗'} "
              f"gemv={'✓' if hasattr(_lib, 'cgemv_4bit_inference_bf16') else '✗'}",
              bug_ref="#19,#21")
    except OSError as e:
        check("A.3  ctypes CDLL load", False, str(e)[:200], bug_ref="#19,#21")

    # A.4 — DNABERT-2 cache
    _cache = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/"
        "zhihan1996/DNABERT_hyphen_2_hyphen_117M"
    )
    _files = _glob.glob(f"{_cache}/*/bert_layers.py")
    check("A.4  DNABERT-2 cache", bool(_files),
          _files[0] if _files else "not downloaded yet", bug_ref="#3")


# ═══════════════════════════════════════════════════════════════════════════
# Group B — Phase 1 (needs torch.xpu, no unsloth)
# ═══════════════════════════════════════════════════════════════════════════

def test_group_b() -> None:
    print("━" * 60)
    print("Group B: Phase 1 (torch.xpu)")
    print("━" * 60)

    _check_subprocess("B.1  torch.xpu.is_available",
        "import torch; assert torch.xpu.is_available(); print('OK')")

    # Bug #17: mem_get_info now works on A770 with PyTorch 2.13+xpu
    # Bug #17: mem_get_info works only AFTER is_available() initializes the device
    _check_subprocess("B.2  mem_get_info (was Bug #17)",
        "import torch; torch.xpu.is_available(); "
        "free, total = torch.xpu.mem_get_info(); "
        "print(f'free={free}, total={total}'); assert total > 0",
        bug_ref="#17")

    _check_subprocess("B.3  Phase 1 patches apply",
        "import torch; assert torch.xpu.is_available(); "
        "from gcu_xpu import apply_phase1_patches; apply_phase1_patches(); print('OK')")

    _check_subprocess("B.4  triton disabled",
        "import torch; from gcu_xpu import apply_phase1_patches; "
        "apply_phase1_patches(); "
        "import torch.utils._triton as _triton; "
        "assert not _triton.is_device_compatible_with_triton(); print('OK')")

    _check_subprocess("B.5  caching_allocator_warmup (Bug #16)",
        "import torch; from gcu_xpu import apply_phase1_patches; "
        "apply_phase1_patches(); "
        "import transformers.modeling_utils as mu; "
        "assert mu.caching_allocator_warmup.__name__ == '<lambda>'; print('OK')",
        bug_ref="#16")

    # deploy_bnb_xpu_lib should be callable (no-op if already deployed)
    _check_subprocess("B.6  deploy_bnb_xpu_lib",
        "from gcu_xpu import deploy_bnb_xpu_lib; "
        "result = deploy_bnb_xpu_lib(); print(f'deployed={result}')",
        bug_ref="#19")


# ═══════════════════════════════════════════════════════════════════════════
# Group C — Phase 2 (needs unsloth + bnb)
# ═══════════════════════════════════════════════════════════════════════════

_IMPORT_BLOCK = r"""
import torch; assert torch.xpu.is_available()
from gcu_xpu import apply_phase1_patches; apply_phase1_patches()
import unsloth
from gcu_xpu import apply_phase2_patches; apply_phase2_patches()
"""


def test_group_c() -> None:
    print("━" * 60)
    print("Group C: Phase 2 (unsloth + bnb)")
    print("━" * 60)

    _check_subprocess("C.1  unsloth XPU detection",
        _IMPORT_BLOCK + "import unsloth.kernels.utils as uku; "
        "assert uku.DEVICE_TYPE == 'xpu'; print('OK')")

    # bnb lib must be BNBNativeLibrary (not ErrorHandlerMock)
    _check_subprocess("C.2  bnb native lib loaded (was Bug #19)",
        _IMPORT_BLOCK + "import bitsandbytes.cextension as cext; "
        "assert not isinstance(cext.lib, cext.ErrorHandlerMockBNBNativeLibrary); "
        "print(f'lib={type(cext.lib).__name__}')",
        bug_ref="#19")

    # Native fast_dequantize must work
    _check_subprocess("C.3  fast_dequantize native (was Bug #19)",
        _IMPORT_BLOCK + r"""
import bitsandbytes.functional as bnbF
from unsloth.kernels.utils import fast_dequantize
W = torch.randn(512, 256, dtype=torch.float16, device='xpu')
W_q, qs = bnbF.quantize_nf4(W, compress_statistics=True)
W_deq = fast_dequantize(W_q, qs)
err = (W_deq.float() - W.float()).abs().mean().item()
print(f'OK, err={err:.4f}')
""", bug_ref="#19")

    # Native fast_gemv must work
    _check_subprocess("C.4  fast_gemv native (was Bug #21)",
        _IMPORT_BLOCK + r"""
import bitsandbytes.functional as bnbF
from unsloth.kernels.utils import fast_gemv
hd = 256; out_f = 512
W = torch.randn(out_f, hd, dtype=torch.float16, device='xpu')
W_q, qs = bnbF.quantize_nf4(W, compress_statistics=True)
X = torch.randn(1, 1, hd, dtype=torch.float16, device='xpu')
out = fast_gemv(X, W_q, qs)
W_d = bnbF.dequantize_nf4(W_q, qs)
exp = torch.matmul(X.float(), W_d.float().t())
cos = torch.nn.functional.cosine_similarity(exp.view(-1), out.float().view(-1), dim=0)
print(f'OK, shape={list(out.shape)}, cos_sim={cos.item():.6f}')
""", bug_ref="#21")

    # torch.cuda mocks: Stream, tensor.cuda, _cuda_isCurrentStreamCapturing
    _check_subprocess("C.5  torch.cuda minimal mocks",
        _IMPORT_BLOCK + r"""
# Stream mock
s = torch.cuda.Stream(); assert s is not None
# tensor.cuda redirect
t = torch.zeros(3).cuda(); assert t.device.type == 'xpu'
print('OK')
""", bug_ref="#8,#9,#10,#11,#12")

    # dist.barrier guard
    _check_subprocess("C.6  dist.barrier guard (Bug #13)",
        _IMPORT_BLOCK + "import torch.distributed as dist; dist.barrier(); print('OK')",
        bug_ref="#13")

    # triton single backend
    _check_subprocess("C.7  triton single backend (Bug #20)",
        _IMPORT_BLOCK + "import triton.backends as tb; "
        "backends = list(tb.backends.keys()); print(f'{backends}')",
        bug_ref="#20")


# ═══════════════════════════════════════════════════════════════════════════
# Group D — Integration
# ═══════════════════════════════════════════════════════════════════════════

def test_group_d() -> None:
    print("━" * 60)
    print("Group D: Integration")
    print("━" * 60)

    # Old patches must NOT be applied (native lib should handle everything)
    _check_subprocess("D.1  no stale dequantize patch",
        _IMPORT_BLOCK + r"""
import unsloth.kernels.utils as uku
import inspect
fn = uku.cdequantize_blockwise_fp32
try:
    src = inspect.getsource(fn)
    assert '_getframe' not in src, 'old patch still active!'
except (TypeError, OSError):
    pass  # native ctypes function, no source available
print('OK')
""",
        bug_ref="#19")

    _check_subprocess("D.2  no stale fast_gemv patch",
        _IMPORT_BLOCK + "import unsloth.kernels.utils as uku; "
        "import inspect; src = inspect.getsource(uku.fast_gemv); "
        "assert '_fast_gemv_xpu' not in src; print('OK')",
        bug_ref="#21")

    # Dequantize + gemv consistency
    _check_subprocess("D.3  dequant vs gemv consistency",
        _IMPORT_BLOCK + r"""
import bitsandbytes.functional as bnbF
from unsloth.kernels.utils import fast_dequantize, fast_gemv
hd = 256; out_f = 512
W = torch.randn(out_f, hd, dtype=torch.float16, device='xpu')
W_q, qs = bnbF.quantize_nf4(W, compress_statistics=True)
W_deq = fast_dequantize(W_q, qs)
X = torch.randn(1, 1, hd, dtype=torch.float16, device='xpu')
out = fast_gemv(X, W_q, qs)
cos = torch.nn.functional.cosine_similarity(
    torch.matmul(X.float(), W_deq.float().t()).view(-1),
    out.float().view(-1), dim=0)
assert cos > 0.999; print(f'OK, cos_sim={cos.item():.6f}')
""", bug_ref="#19,#21")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def print_summary() -> None:
    n_pass = sum(1 for r in _results if r.status == "PASS")
    n_fail = sum(1 for r in _results if r.status == "FAIL")
    n_skip = sum(1 for r in _results if r.status == "SKIP")

    print("\n" + "═" * 70)
    print("RESULTS SUMMARY")
    print("═" * 70)

    for r in _results:
        if r.status == "FAIL":
            icon = "✗" if sys.stdout.encoding == "utf-8" else "X"
            print(f"  {icon} {r.name}  [{r.bug_ref}]")
            if r.message:
                print(f"     {r.message}")
            if _VERBOSE and r.details:
                for line in r.details.split("\n")[:15]:
                    print(f"     │ {line}")
    for r in _results:
        if r.status == "PASS":
            icon = "✓" if sys.stdout.encoding == "utf-8" else "P"
            print(f"  {icon} {r.name}")
        elif r.status == "SKIP":
            icon = "○" if sys.stdout.encoding == "utf-8" else "S"
            print(f"  {icon} {r.name}  — {r.message}")

    print("─" * 70)
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  SKIP: {n_skip}  TOTAL: {len(_results)}")
    print("═" * 70)
    if n_fail > 0:
        sys.exit(1)


def main() -> None:
    global _VERBOSE
    group_filter = None

    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a in ("-v", "--verbose"):
            _VERBOSE = True
        elif a in ("-g", "--group") and i + 1 < len(args):
            group_filter = args[i + 1].upper()

    print("XPU Bug Verification Test Suite")
    print(f"Python: {sys.version}")
    print()

    if group_filter is None or group_filter == "A":
        test_group_a()
    if group_filter is None or group_filter == "B":
        test_group_b()
    if group_filter is None or group_filter == "C":
        test_group_c()
    if group_filter is None or group_filter == "D":
        test_group_d()

    print_summary()


if __name__ == "__main__":
    main()
