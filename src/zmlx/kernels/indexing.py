from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def _gather_add_kernel(d: int) -> Any:
    D = int(d)
    source = f"""
        constexpr uint D = {D};
        uint gid = thread_position_in_grid.x;
        uint i = gid / D;
        uint j = gid % D;
        
        uint idx = indices[i];
        out[gid] = (T)((float)src[idx * D + j] + (float)other[gid]);
    """
    return metal_kernel(
        name=f"kk_gather_add_D{D}",
        input_names=["src", "indices", "other"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def fused_gather_add(src: Any, indices: Any, other: Any, *, compute_dtype: Any | None = None) -> Any:
    """Gather rows from src, and add other.
    
    src: (N, D)
    indices: (M,)
    other: (M, D)
    returns: (M, D)
    """
    M, D = other.shape
    cd = compute_dtype or mx.float32
    k = _gather_add_kernel(D)
    return k(
        src, indices, other,
        template=[("T", cd)],
        grid=(M * D, 1, 1),
        output_shapes=[(M, D)],
        output_dtypes=[src.dtype],
    )[0]

@cache
def _scatter_add_kernel(d: int) -> Any:
    D = int(d)
    # Use atomic_fetch_add_explicit for thread-safe accumulation
    source = f"""
        constexpr uint D = {D};
        uint gid = thread_position_in_grid.x;
        uint i = gid / D;
        uint j = gid % D;
        
        uint idx = indices[i];
        // Metal atomics work on device atomic<float>* for atomic_outputs=True
        atomic_fetch_add_explicit(&out[idx * D + j], (float)updates[gid], memory_order_relaxed);
    """
    return metal_kernel(
        name=f"kk_scatter_add_D{D}",
        input_names=["indices", "updates"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        atomic_outputs=True,
        cache=True,
    )

def fused_scatter_add(indices: Any, updates: Any, out_shape: tuple[int, int], *, compute_dtype: Any | None = None) -> Any:
    """Scatter updates into a zero-initialized array of out_shape.
    
    indices: (M,)
    updates: (M, D)
    out_shape: (N, D)
    """
    M, D = updates.shape
    cd = compute_dtype or mx.float32
    k = _scatter_add_kernel(D)
    
    # We must provide init_value for atomic_outputs=True
    return k(
        indices, updates,
        template=[("T", cd)],
        grid=(M * D, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[cd],
        init_value=0.0,
    )[0]

__all__ = [
    "fused_gather_add",
    "fused_scatter_add",
]
