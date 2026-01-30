from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .softmax import _validate_tg


@cache
def _top2_gating_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float val1_buf[TG];
        threadgroup float val2_buf[TG];
        threadgroup uint idx1_buf[TG];
        threadgroup uint idx2_buf[TG];

        float top1_v = -INFINITY;
        float top2_v = -INFINITY;
        uint top1_i = 0;
        uint top2_i = 0;

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            if (v > top1_v) {{
                top2_v = top1_v;
                top2_i = top1_i;
                top1_v = v;
                top1_i = j;
            }} else if (v > top2_v) {{
                top2_v = v;
                top2_i = j;
            }}
        }}
        
        val1_buf[tid] = top1_v;
        val2_buf[tid] = top2_v;
        idx1_buf[tid] = top1_i;
        idx2_buf[tid] = top2_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint st = TG/2; st > 0; st >>= 1) {{
            if (tid < st) {{
                float v1_a = val1_buf[tid];
                float v1_b = val1_buf[tid + st];
                float v2_a = val2_buf[tid];
                float v2_b = val2_buf[tid + st];
                
                uint i1_a = idx1_buf[tid];
                uint i1_b = idx1_buf[tid + st];
                uint i2_a = idx2_buf[tid];
                uint i2_b = idx2_buf[tid + st];

                // Merge two top-2 sets
                float res_v1, res_v2;
                uint res_i1, res_i2;
                
                if (v1_a > v1_b) {{
                    res_v1 = v1_a; res_i1 = i1_a;
                    if (v1_b > v2_a) {{
                        res_v2 = v1_b; res_i2 = i1_b;
                    }} else {{
                        res_v2 = v2_a; res_i2 = i2_a;
                    }}
                }} else {{
                    res_v1 = v1_b; res_i1 = i1_b;
                    if (v1_a > v2_b) {{
                        res_v2 = v1_a; res_i2 = i1_a;
                    }} else {{
                        res_v2 = v2_b; res_i2 = i2_b;
                    }}
                }}
                
                val1_buf[tid] = res_v1;
                idx1_buf[tid] = res_i1;
                val2_buf[tid] = res_v2;
                idx2_buf[tid] = res_i2;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0) {{
            float m = val1_buf[0];
            float v1 = metal::exp(val1_buf[0] - m);
            float v2 = metal::exp(val2_buf[0] - m);
            float s = v1 + v2;
            
            weights[row * 2] = (T)(v1 / s);
            weights[row * 2 + 1] = (T)(v2 / s);
            indices[row * 2] = idx1_buf[0];
            indices[row * 2 + 1] = idx2_buf[0];
        }}
    """
    return metal_kernel(
        name=f"kk_top2_gating_D{D}_TG{TG}",
        input_names=["inp"],
        output_names=["weights", "indices"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def top2_gating_softmax(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> tuple[Any, Any]:
    """Top-2 gating with softmax for Mixture of Experts.
    
    Returns:
      - weights: (..., 2) softmax probabilities
      - indices: (..., 2) expert indices (uint32)
    """
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _top2_gating_kernel(D, TG)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    
    weights, indices = k(
        x,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape[:-1] + (2,), x.shape[:-1] + (2,)],
        output_dtypes=[cd, mx.uint32],
    )
    return weights, indices

@cache
def _moe_dispatch_kernel(d: int, k: int) -> Any:
    D = int(d)
    K = int(k)
    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;
        uint k_idx = thread_position_in_grid.z;
        
        // This is a simple gather-like dispatch
        // x: (B, D)
        // indices: (B, K)
        // out: (B, K, D)
        out[(token_idx * K + k_idx) * D + d_idx] = x[token_idx * D + d_idx];
    """
    return metal_kernel(
        name=f"kk_moe_dispatch_D{D}_K{K}",
        input_names=["x", "indices"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def moe_dispatch(x: Any, indices: Any) -> Any:
    """Dispatch tokens to expert slots.
    
    x: (..., D)
    indices: (..., K)
    Returns: (..., K, D)
    """
    original_shape = x.shape[:-1]
    D = x.shape[-1]
    K = indices.shape[-1]
    
    x_flat = x.reshape(-1, D)
    indices_flat = indices.reshape(-1, K)
    B = x_flat.shape[0]
    
    k = _moe_dispatch_kernel(D, K)
    out = k(
        x_flat, indices_flat,
        template=[("T", x.dtype)],
        grid=(D, B, K),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, K, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape((*original_shape, K, D))


@cache
def _moe_combine_kernel(d: int, k: int) -> Any:
    D = int(d)
    K = int(k)
    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;
        
        float acc = 0.0f;
        for (uint i = 0; i < K; ++i) {{
            float w = (float)weights[token_idx * K + i];
            float v = (float)expert_outputs[(token_idx * K + i) * D + d_idx];
            acc += w * v;
        }}
        out[token_idx * D + d_idx] = (T)acc;
    """
    return metal_kernel(
        name=f"kk_moe_combine_D{D}_K{K}",
        input_names=["expert_outputs", "weights"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def moe_combine(expert_outputs: Any, weights: Any) -> Any:
    """Combine expert outputs using gating weights.
    
    expert_outputs: (..., K, D)
    weights: (..., K)
    Returns: (..., D)
    """
    original_shape = weights.shape[:-1]
    K = weights.shape[-1]
    D = expert_outputs.shape[-1]
    
    expert_outputs_flat = expert_outputs.reshape(-1, K, D)
    weights_flat = weights.reshape(-1, K)
    B = weights_flat.shape[0]
    
    k = _moe_combine_kernel(D, K)
    out = k(
        expert_outputs_flat, weights_flat,
        template=[("T", expert_outputs.dtype)],
        grid=(D, B, 1),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, D)],
        output_dtypes=[expert_outputs.dtype],
    )[0]
    return out.reshape((*original_shape, D))


__all__ = [
    "top2_gating_softmax",
    "moe_dispatch",
    "moe_combine",
]