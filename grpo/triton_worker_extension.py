"""
TRL WeightSyncWorkerExtension wrapper that ensures Triton's allocator is set.

Triton 3.x may require an explicit allocator for kernels that do runtime
allocations (MoE routing / GDN / TMA scratch). If not set, you'll see:
  RuntimeError: Kernel requires a runtime memory allocation, but no allocator was set.
"""
from __future__ import annotations

import threading

import torch

_LOCK = threading.Lock()
_DONE = False


def ensure_triton_allocator():
    global _DONE
    if _DONE:
        return
    with _LOCK:
        if _DONE:
            return
        try:
            import triton
            # Prefer official allocator if available; else fallback to torch tensor allocator.
            try:
                import triton.runtime.driver as _triton_driver
                triton.set_allocator(_triton_driver.CudaAllocator())
            except Exception:
                def _alloc_fn(size: int, alignment: int, stream: int | None):
                    # size is in bytes; allocate a 1D int8 buffer of that size.
                    return torch.empty((size,), device="cuda", dtype=torch.int8)
                triton.set_allocator(_alloc_fn)
        except Exception:
            # If Triton isn't available in this process, just skip.
            pass
        _DONE = True


# Set at import time (good for workers that import this module).
ensure_triton_allocator()

from trl.scripts.vllm_serve import WeightSyncWorkerExtension


class TritonAllocatorWeightSyncWorkerExtension(WeightSyncWorkerExtension):
    # Also set at instantiation time (belt-and-suspenders).
    def __init__(self, *args, **kwargs):
        ensure_triton_allocator()
        super().__init__(*args, **kwargs)
