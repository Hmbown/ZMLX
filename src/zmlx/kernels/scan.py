from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .softmax import _validate_tg


@cache
def _cumsum_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    # Simple Hillis-Steele style scan if D <= TG, 
    # or a more complex multi-pass for larger D.
    # For now, let's implement a robust rowwise loop if D > TG.
    
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        // If TG is 1, just loop serially
        if (TG == 1) {{
            float acc = 0.0f;
            for (uint j = 0; j < D; ++j) {{
                acc += (float)inp[base + j];
                out[base + j] = (T)acc;
            }}
            return;
        }}

        // Parallel scan for D <= TG (simplified)
        // For arbitrary D, we use a serial-per-thread + parallel-between-threads approach
        threadgroup float buf[TG];
        
        float last_sum = 0.0f;
        for (uint chunk = 0; chunk < D; chunk += TG) {{
            uint j = chunk + tid;
            float val = (j < D) ? (float)inp[base + j] : 0.0f;
            
            // Inclusive scan in threadgroup
            buf[tid] = val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for (uint offset = 1; offset < TG; offset <<= 1) {{
                float t = (tid >= offset) ? buf[tid - offset] : 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf[tid] += t;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            
            if (j < D) {{
                out[base + j] = (T)(buf[tid] + last_sum);
            }}
            last_sum += buf[TG - 1];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    """
    return metal_kernel(
        name=f"kk_cumsum_D{D}_TG{TG}",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def cumsum_lastdim(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Inclusive sum (scan) over the last dimension."""
    if x.ndim < 1:
        return x
    D = int(x.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    k_fwd = _cumsum_kernel(D, TG)

    @mx.custom_function
    def op(x_in):
        return k_fwd(
            x_in,
            template=[("T", cd)],
            grid=(rows * TG, 1, 1),
            threadgroup=(TG, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]

    @op.vjp
    def op_vjp(primals, cotangents, outputs):
        cotan = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
        dx = cumsum_grad(cotan, threadgroup=TG, compute_dtype=cd)
        return (dx,)

    return op(x)


@cache
def _cumsum_bwd_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    # The gradient of inclusive scan is reverse exclusive scan
    # dL/dx_i = sum_{j=i}^{D-1} dL/dy_j
    
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        if (TG == 1) {{
            float acc = 0.0f;
            for (int j = (int)D - 1; j >= 0; --j) {{
                acc += (float)cotan[base + (uint)j];
                dout[base + (uint)j] = (T)acc;
            }}
            return;
        }}

        threadgroup float buf[TG];
        float last_sum = 0.0f;
        
        // Process in reverse chunks
        for (int chunk = ((int)D - 1) / (int)TG * (int)TG; chunk >= 0; chunk -= (int)TG) {{
            uint j = (uint)chunk + tid;
            float val = (j < D) ? (float)cotan[base + j] : 0.0f;
            
            // Inclusive scan in threadgroup
            buf[tid] = val;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Since we need REVERSE sum, we can either reverse indices or do prefix sum and then adjust.
            // Let's do a prefix sum of the reversed chunk.
            uint rev_tid = TG - 1 - tid;
            float rev_val = (chunk + (int)rev_tid < (int)D) ? (float)cotan[base + (uint)chunk + rev_tid] : 0.0f;
            buf[tid] = rev_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint offset = 1; offset < TG; offset <<= 1) {{
                float t = (tid >= offset) ? buf[tid - offset] : 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                buf[tid] += t;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            
            // Now buf[tid] contains sum of first tid elements of the REVERSED chunk.
            // Which means buf[rev_tid] contains suffix sum of the chunk.
            if (j < D) {{
                dout[base + j] = (T)(buf[rev_tid] + last_sum);
            }}
            last_sum += buf[TG - 1];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
    """
    return metal_kernel(
        name=f"kk_cumsum_bwd_D{D}_TG{TG}",
        input_names=["cotan"],
        output_names=["dout"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def cumsum_grad(cotan: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Gradient of cumsum (reverse inclusive scan)."""
    D = int(cotan.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = cotan.size // D
    cd = compute_dtype or mx.float32
    k = _cumsum_bwd_kernel(D, TG)
    return k(
        cotan,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[cotan.shape],
        output_dtypes=[cotan.dtype],
    )[0]

__all__ = [
    "cumsum_lastdim",
    "cumsum_grad",
]
