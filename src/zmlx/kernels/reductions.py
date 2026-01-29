from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..codegen import rowwise_parallel_reduction_source
from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .softmax import _validate_tg


@cache
def _reduction_kernel(d: int, tg: int, op: str, kahan: bool = False) -> Any:
    D = int(d)
    TG = _validate_tg(tg)

    if kahan and op in ("sum", "mean"):
        # Kahan summation for better precision in float16
        source = f"""
            constexpr uint D = {D};
            constexpr uint TG = {TG};

            uint gid = thread_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint row = gid / TG;
            uint base = row * D;

            threadgroup float buf[TG];

            float sum = 0.0f;
            float c = 0.0f;
            for (uint j = tid; j < D; j += TG) {{
                float x = (float)inp[base + j];
                float y = x - c;
                float t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }}
            buf[tid] = sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
                if (tid < stride) {{
                    buf[tid] += buf[tid + stride];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            
            if (tid == 0) {{
                float s = buf[0];
                out[row] = (T)({ "s" if op == "sum" else f"s / (float){D}" });
            }}
        """
        return metal_kernel(
            name=f"kk_{op}_kahan_D{D}_TG{TG}",
            input_names=["inp"],
            output_names=["out"],
            source=source,
            header=DEFAULT_HEADER,
            ensure_row_contiguous=True,
            cache=True,
        )

    if op == "sum":
        init_expr = "0.0f"
        update_expr = "acc + x"
        reduce_op = "a + b"
        finalize_expr = "s"
    elif op == "mean":
        init_expr = "0.0f"
        update_expr = "acc + x"
        reduce_op = "a + b"
        finalize_expr = f"s / (float){D}"
    elif op == "max":
        init_expr = "-INFINITY"
        update_expr = "metal::max(acc, x)"
        reduce_op = "metal::max(a, b)"
        finalize_expr = "s"
    elif op == "min":
        init_expr = "INFINITY"
        update_expr = "metal::min(acc, x)"
        reduce_op = "metal::min(a, b)"
        finalize_expr = "s"
    else:
        raise ValueError(f"Unknown reduction op: {op}")

    src = rowwise_parallel_reduction_source(
        d=D,
        tg=TG,
        init_expr=init_expr,
        update_expr=update_expr,
        reduce_op=reduce_op,
        finalize_expr=finalize_expr,
        inp="inp",
        out="out",
    )

    return metal_kernel(
        name=f"kk_{op}_D{D}_TG{TG}",
        input_names=["inp"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def sum_lastdim(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None, kahan: bool = False) -> Any:
    """Sum reduction over the last dimension."""
    if x.ndim < 1:
        return x.sum()
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _reduction_kernel(D, TG, "sum", kahan=kahan)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    return k(
        x,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape[:-1]],
        output_dtypes=[x.dtype],
    )[0]


def mean_lastdim(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None, kahan: bool = False) -> Any:
    """Mean reduction over the last dimension."""
    if x.ndim < 1:
        return x.mean()
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _reduction_kernel(D, TG, "mean", kahan=kahan)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    return k(
        x,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape[:-1]],
        output_dtypes=[x.dtype],
    )[0]


def max_lastdim(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Max reduction over the last dimension."""
    if x.ndim < 1:
        return x.max()
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _reduction_kernel(D, TG, "max")
    rows = x.size // D
    cd = compute_dtype or mx.float32
    return k(
        x,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape[:-1]],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _var_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    # Using 2-pass algorithm for better numerical stability than 1-pass naive
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // 1. Mean pass
        float s = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            s += (float)inp[base + j];
        }}
        buf[tid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG/2; st > 0; st >>= 1) {{
            if (tid < st) buf[tid] += buf[tid + st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float mean = buf[0] / (float)D;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Variance pass: mean((x - mean)^2)
        float v = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float diff = (float)inp[base + j] - mean;
            v += diff * diff;
        }}
        buf[tid] = v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG/2; st > 0; st >>= 1) {{
            if (tid < st) buf[tid] += buf[tid + st];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        
        if (tid == 0) {{
            out[row] = (T)(buf[0] / (float)D);
        }}
    """
    return metal_kernel(
        name=f"kk_var_D{D}_TG{TG}",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def var_lastdim(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Variance reduction over the last dimension."""
    if x.ndim < 1:
        return mx.array(0.0, dtype=x.dtype)
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _var_kernel(D, TG)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    return k(
        x,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape[:-1]],
        output_dtypes=[x.dtype],
    )[0]


def std_lastdim(x: Any, *, eps: float = 1e-6, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Standard deviation reduction over the last dimension."""
    # We can just reuse var_lastdim and sqrt
    v = var_lastdim(x, threadgroup=threadgroup, compute_dtype=compute_dtype)
    return mx.sqrt(v + eps)


@cache
def _argmax_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float val_buf[TG];
        threadgroup uint idx_buf[TG];

        float max_val = -INFINITY;
        uint max_idx = 0;
        
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            if (v > max_val) {{
                max_val = v;
                max_idx = j;
            }}
        }}
        val_buf[tid] = max_val;
        idx_buf[tid] = max_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint st = TG/2; st > 0; st >>= 1) {{
            if (tid < st) {{
                if (val_buf[tid + st] > val_buf[tid]) {{
                    val_buf[tid] = val_buf[tid + st];
                    idx_buf[tid] = idx_buf[tid + st];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        
        if (tid == 0) {{
            out[row] = idx_buf[0];
        }}
    """
    return metal_kernel(
        name=f"kk_argmax_D{D}_TG{TG}",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def argmax_lastdim(x: Any, *, threadgroup: int = 256) -> Any:
    """Argmax reduction over the last dimension. Returns uint32."""
    if x.ndim < 1:
        return mx.array(0, dtype=mx.uint32)
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _argmax_kernel(D, TG)
    rows = x.size // D
    # Argmax always uses float32 internal for comparison, returns uint32
    return k(
        x,
        template=[("T", mx.uint32)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape[:-1]],
        output_dtypes=[mx.uint32],
    )[0]


@cache
def _topk_kernel(d: int, k: int, tg: int) -> Any:
    D = int(d)
    K = int(k)
    TG = _validate_tg(tg)
    
    # Simple insertion sort in registers for small K
    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float val_buf[TG * K];
        threadgroup uint idx_buf[TG * K];

        // Local top-k for this thread's chunk
        float local_vals[K];
        uint local_idxs[K];
        for (uint i = 0; i < K; ++i) local_vals[i] = -INFINITY;

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            if (v > local_vals[K-1]) {{
                // insertion
                uint p = K - 1;
                while (p > 0 && v > local_vals[p-1]) {{
                    local_vals[p] = local_vals[p-1];
                    local_idxs[p] = local_idxs[p-1];
                    p--;
                }}
                local_vals[p] = v;
                local_idxs[p] = j;
            }}
        }}
        
        // Write to threadgroup for final merge
        for (uint i = 0; i < K; ++i) {{
            val_buf[tid * K + i] = local_vals[i];
            idx_buf[tid * K + i] = local_idxs[i];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Binary merge tree
        for (uint st = TG/2; st > 0; st >>= 1) {{
            if (tid < st) {{
                // Merge tid and tid + st
                float v_a[K], v_b[K];
                uint i_a[K], i_b[K];
                for (uint i = 0; i < K; ++i) {{
                    v_a[i] = val_buf[tid * K + i];
                    i_a[i] = idx_buf[tid * K + i];
                    v_b[i] = val_buf[(tid + st) * K + i];
                    i_b[i] = idx_buf[(tid + st) * K + i];
                }}
                
                uint pa = 0, pb = 0;
                for (uint i = 0; i < K; ++i) {{
                    if (v_a[pa] >= v_b[pb]) {{
                        val_buf[tid * K + i] = v_a[pa];
                        idx_buf[tid * K + i] = i_a[pa];
                        pa++;
                    }} else {{
                        val_buf[tid * K + i] = v_b[pb];
                        idx_buf[tid * K + i] = i_b[pb];
                        pb++;
                    }}
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0) {{
            for (uint i = 0; i < K; ++i) {{
                out_vals[row * K + i] = (T)val_buf[i];
                out_idxs[row * K + i] = idx_buf[i];
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_topk_D{D}_K{K}_TG{TG}",
        input_names=["inp"],
        output_names=["out_vals", "out_idxs"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def topk_lastdim(x: Any, k: int, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> tuple[Any, Any]:
    """Top-K reduction over the last dimension.
    
    Optimized for small K (e.g. <= 16).
    Returns (values, indices).
    """
    if k <= 0:
        raise ValueError("topk: k must be > 0")
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k_val = int(k)
    k_kernel = _topk_kernel(D, k_val, TG)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    
    v, i = k_kernel(
        x,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape[:-1] + (k_val,), x.shape[:-1] + (k_val,)],
        output_dtypes=[cd, mx.uint32],
    )
    return v, i

__all__ = [
    "sum_lastdim",
    "mean_lastdim",
    "max_lastdim",
    "var_lastdim",
    "std_lastdim",
    "argmax_lastdim",
    "topk_lastdim",
]
