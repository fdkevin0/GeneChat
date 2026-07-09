"""
Centralized device module for GeneChat. Auto-detects XPU/CUDA/CPU at import
time and provides a single API surface so no other genechat code needs to
branch on device type.

After this module, all device operations flow through:
    from genechat.common import device as genechat_device
    genechat_device.to_device(tensor)
    genechat_device.autocast(dtype=torch.bfloat16)
    genechat_device.Stream()
    genechat_device.barrier()
etc.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any

import torch
import torch.distributed as dist


# ═══════════════════════════════════════════════════════════════════════
# Module-level auto-detection (runs once at first import)
# ═══════════════════════════════════════════════════════════════════════

def _resolve_device_type() -> str:
    """Priority: env var > XPU > CUDA > CPU."""
    env = os.environ.get("GENECHAT_DEVICE", "").lower()
    if env in ("xpu", "cuda", "cpu"):
        return env
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_DEVICE_TYPE: str = _resolve_device_type()
_DEVICE: torch.device = torch.device(_DEVICE_TYPE)


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def device_type() -> str:
    """Return 'xpu', 'cuda', or 'cpu'."""
    return _DEVICE_TYPE


def device() -> torch.device:
    """Return torch.device matching the resolved device type."""
    return _DEVICE


def is_available() -> bool:
    """True when an accelerator (XPU or CUDA) is available."""
    return _DEVICE_TYPE != "cpu"


def _backend():
    """Return the torch.xpu / torch.cuda module for the resolved device
    type, or None on CPU. xpu and cuda modules share method names for
    everything device_count/current_device/... below, so callers just
    dispatch to whichever module matches."""
    if _DEVICE_TYPE == "xpu":
        return torch.xpu
    if _DEVICE_TYPE == "cuda":
        return torch.cuda
    return None


def device_count() -> int:
    """Number of accelerator devices available."""
    backend = _backend()
    return backend.device_count() if backend else 0


def current_device() -> int:
    """Index of the current accelerator device."""
    backend = _backend()
    return backend.current_device() if backend else 0


def set_device(dev: int) -> None:
    """Set the current accelerator device."""
    backend = _backend()
    if backend:
        backend.set_device(dev)


def synchronize(dev: int | None = None) -> None:
    """Synchronize the current or specified accelerator device."""
    backend = _backend()
    if backend:
        backend.synchronize(dev)


def reset_peak_memory_stats(dev: int | None = None) -> None:
    """Reset peak memory statistics."""
    backend = _backend()
    if backend:
        backend.reset_peak_memory_stats(dev)


def max_memory_allocated(dev: int | None = None) -> int:
    """Peak memory allocated in bytes since last reset."""
    backend = _backend()
    return backend.max_memory_allocated(dev) if backend else 0


def memory_stats(dev: int | None = None) -> dict[str, Any]:
    """Return memory statistics dict."""
    backend = _backend()
    return backend.memory_stats(dev) if backend else {}


def dist_backend() -> str:
    """Return the appropriate distributed backend for the current device."""
    if _DEVICE_TYPE == "cuda":
        return "nccl"
    if _DEVICE_TYPE == "xpu":
        return "ccl"
    return "gloo"


def get_device_properties(dev: int | None = None):
    """Return device properties object (for compatibility with code that
    inspects compute capability, multiprocessor count, etc.)."""
    backend = _backend()
    if backend is None:
        raise RuntimeError("No accelerator available")
    return backend.get_device_properties(dev)


# ═══════════════════════════════════════════════════════════════════════
# Tensor movement
# ═══════════════════════════════════════════════════════════════════════

def to_device(tensor: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    """Move a tensor to the resolved accelerator device."""
    if _DEVICE_TYPE == "cpu":
        return tensor
    return tensor.to(_DEVICE, non_blocking=non_blocking)


# ═══════════════════════════════════════════════════════════════════════
# autocast — unified interface
# ═══════════════════════════════════════════════════════════════════════

def autocast(dtype: torch.dtype | None = None, enabled: bool = True):
    """Return an autocast context manager for the current device.

    On CUDA: uses torch.cuda.amp.autocast (legacy).
    On XPU:  uses torch.autocast(device_type='xpu', ...).
    On CPU:  returns a no-op context manager.
    """
    if not enabled or _DEVICE_TYPE == "cpu":
        return contextlib.nullcontext()

    if dtype is None:
        dtype = torch.bfloat16 if _DEVICE_TYPE == "xpu" else torch.float16

    if _DEVICE_TYPE == "xpu":
        return torch.autocast(device_type="xpu", dtype=dtype)
    if _DEVICE_TYPE == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    return contextlib.nullcontext()


# ═══════════════════════════════════════════════════════════════════════
# GradScaler — unified wrapper
# ═══════════════════════════════════════════════════════════════════════

class _NoOpScaler:
    """Pass-through scaler for bf16 (no loss scaling needed)."""

    def scale(self, loss): return loss
    def step(self, optimizer): optimizer.step()
    def update(self): pass
    def get_scale(self): return 1.0
    def state_dict(self): return {}
    def load_state_dict(self, s): pass


def GradScaler():
    """Return a GradScaler appropriate for the current device.

    On CUDA: returns torch.cuda.amp.GradScaler.
    On XPU:  returns a no-op (bf16 doesn't need loss scaling).
    On CPU:  returns a no-op.
    """
    if _DEVICE_TYPE == "cuda":
        return torch.cuda.amp.GradScaler()
    return _NoOpScaler()


# ═══════════════════════════════════════════════════════════════════════
# Stream — unified wrapper
# ═══════════════════════════════════════════════════════════════════════

class Stream:
    """A device stream wrapper for prefetch/compute overlap.

    On CUDA: wraps torch.cuda.Stream.
    On XPU:  Intel PyTorch extensions may not fully support XPU streams;
             falls back to a no-op mock (record_stream is not supported).
    """

    def __init__(self):
        if _DEVICE_TYPE == "cuda":
            self._stream = torch.cuda.Stream()
        else:
            self._stream = None  # XPU/CPU: no-op

    def wait_stream(self, other: Stream) -> None:
        if self._stream is not None and other._stream is not None:
            self._stream.wait_stream(other._stream)

    def record_stream(self, tensor: torch.Tensor) -> None:
        if self._stream is not None:
            tensor.record_stream(self._stream)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def current_stream(dev: int | None = None) -> Stream:
    """Return a Stream representing the current device stream."""
    s = Stream()
    if _DEVICE_TYPE == "cuda":
        s._stream = torch.cuda.current_stream(dev)
    return s


def using_stream(stream: Stream):
    """Context manager that makes `stream` the current stream.

    On CUDA: uses torch.cuda.stream(stream).
    On XPU/CPU: no-op.
    """
    if _DEVICE_TYPE == "cuda" and stream._stream is not None:
        return torch.cuda.stream(stream._stream)
    return contextlib.nullcontext()


def record_stream(tensor: torch.Tensor, stream: Stream | None = None) -> None:
    """Record a tensor for the given (or current) stream.

    On XPU: record_stream is not supported — this is a no-op.
    """
    if _DEVICE_TYPE == "cuda":
        s = stream._stream if stream is not None else torch.cuda.current_stream()
        tensor.record_stream(s)


# ═══════════════════════════════════════════════════════════════════════
# Distributed utilities
# ═══════════════════════════════════════════════════════════════════════

def barrier() -> None:
    """Call dist.barrier() with the appropriate device_ids for the backend.

    On CUDA: passes device_ids=[current_device()].
    On XPU/CPU: omits device_ids.
    Safe to call even when dist is not initialized (no-op).
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    if _DEVICE_TYPE == "cuda":
        dist.barrier(device_ids=[current_device()])
    else:
        dist.barrier()
