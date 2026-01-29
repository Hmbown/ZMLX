from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def _linear_bias_act_kernel(m: int, n: int, k: int, act_expr: str = "sum") -> Any:
    # M: batch/rows, N: out_features, K: in_features
    # This is a naive dot-product kernel (one thread per output element)
    # Good for small N or non-GEMM cases.
    source = f"""
        constexpr uint M = {m};
        constexpr uint N = {n};
        constexpr uint K = {k};

        uint gid = thread_position_in_grid.x;
        uint row = gid / N;
        uint col = gid % N;

        if (row < M && col < N) {{
            float sum = 0.0f;
            for (uint i = 0; i < K; ++i) {{
                sum += (float)x[row * K + i] * (float)w[col * K + i];
            }}
            sum += (float)bias[col];
            out[gid] = (T)({act_expr});
        }}
    """
    return metal_kernel(
        name=f"kk_linear_bias_act_{hash(act_expr) % 10000}",
        input_names=["x", "w", "bias"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def fused_linear_bias_silu(x: Any, w: Any, bias: Any, *, compute_dtype: Any | None = None) -> Any:
    """Fused Linear + Bias + SiLU.

    Args:
        x: Input array with shape ``(M, K)``.
        w: Weight matrix with shape ``(N, K)`` (MLX convention).
        bias: Bias vector with shape ``(N,)``.
        compute_dtype: Dtype used for internal computation.

    Returns:
        Output array with shape ``(M, N)``.
    """
    M, K = x.shape
    N, K_w = w.shape
    if K != K_w:
        raise ValueError("linear: inner dimensions must match")
    if int(bias.ndim) != 1 or int(bias.shape[0]) != int(N):
        raise ValueError(f"linear: bias must have shape ({int(N)},)")
    
    cd = compute_dtype or mx.float32
    k = _linear_bias_act_kernel(M, N, K, act_expr="kk_silu(sum)")
    return k(
        x, w, bias,
        template=[("T", cd)],
        grid=(M * N, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[cd],
    )[0]

def fused_linear_bias_gelu(x: Any, w: Any, bias: Any, *, compute_dtype: Any | None = None) -> Any:
    """Fused Linear + Bias + GeLU (tanh approximation)."""
    M, K = x.shape
    N, K_w = w.shape
    if K != K_w:
        raise ValueError("linear: inner dimensions must match")
    if int(bias.ndim) != 1 or int(bias.shape[0]) != int(N):
        raise ValueError(f"linear: bias must have shape ({int(N)},)")
    cd = compute_dtype or mx.float32
    k = _linear_bias_act_kernel(M, N, K, act_expr="kk_gelu_tanh(sum)")
    return k(
        x, w, bias,
        template=[("T", cd)],
        grid=(M * N, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[cd],
    )[0]

@cache
def _linear_rmsnorm_kernel(m: int, n: int, k: int, eps: float) -> Any:
    M = int(m)
    N = int(n)
    K = int(k)
    eps_f = float(eps)
    
    # Each threadgroup handles ONE row of the output matrix (M, N)
    # This is necessary because RMSNorm needs a rowwise reduction.
    # TG size should be a power of two, matching N if possible, or we loop.
    TG = 256 # Default
    
    source = f"""
        constexpr uint M = {M};
        constexpr uint N = {N};
        constexpr uint K = {K};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint tid = thread_position_in_threadgroup.x;
        uint row = thread_position_in_grid.x / TG;
        
        threadgroup float buf[TG];

        // 1. Dot product loop + sumsq for RMS
        float sumsq = 0.0f;
        for (uint col = tid; col < N; col += TG) {{
            float dot = 0.0f;
            for (uint i = 0; i < K; ++i) {{
                dot += (float)x[row * K + i] * (float)w[col * K + i];
            }}
            // Store dot product in a temporary way or recompute
            // To avoid huge shared memory, we recompute dot later.
            // But we need the sumsq of ALL dots in this row.
            sumsq += dot * dot;
        }}
        
        buf[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float rms = metal::rsqrt(buf[0] / (float)N + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Final pass: recompute dot and normalize
        for (uint col = tid; col < N; col += TG) {{
            float dot = 0.0f;
            for (uint i = 0; i < K; ++i) {{
                dot += (float)x[row * K + i] * (float)w[col * K + i];
            }}
            float weight = (float)gamma[col];
            out[row * N + col] = (T)(dot * rms * weight);
        }}
    """
    return metal_kernel(
        name=f"kk_linear_rmsnorm_M{M}_N{N}_K{K}",
        input_names=["x", "w", "gamma"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def fused_linear_rmsnorm(x: Any, w: Any, gamma: Any, *, eps: float = 1e-6, compute_dtype: Any | None = None) -> Any:
    """Fused Linear + RMSNorm.

    Args:
        x: Input array with shape ``(M, K)``.
        w: Weight matrix with shape ``(N, K)``.
        gamma: RMSNorm scale vector with shape ``(N,)``.
        eps: RMSNorm epsilon for numerical stability.
        compute_dtype: Dtype used for internal computation.

    Returns:
        Output array with shape ``(M, N)``.
    """
    M, K = x.shape
    N, K_w = w.shape
    if K != K_w:
        raise ValueError("linear: inner dimensions must match")
    if int(gamma.ndim) != 1 or int(gamma.shape[0]) != int(N):
        raise ValueError(f"fused_linear_rmsnorm: gamma must have shape ({int(N)},)")
    cd = compute_dtype or mx.float32
    k = _linear_rmsnorm_kernel(M, N, K, eps)
    
    TG = 256
    return k(
        x, w, gamma,
        template=[("T", cd)],
        grid=(M * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[cd],
    )[0]


@cache
def _dequant_int4_matmul_kernel(m: int, n: int, k: int) -> Any:
    M, N, K = int(m), int(n), int(k)
    # K must be even for int4 packing (2 nibbles per byte)
    source = f"""
        constexpr uint M = {M};
        constexpr uint N = {N};
        constexpr uint K = {K};

        uint gid = thread_position_in_grid.x;
        uint row = gid / N;
        uint col = gid % N;

        if (row < M && col < N) {{
            float sum = 0.0f;
            float s = (float)scales[col];
            
            for (uint i = 0; i < K; i += 2) {{
                uint8_t packed = weights[col * (K/2) + (i/2)];
                
                // Extract two int4 values
                int8_t v0 = (int8_t)(packed & 0x0F);
                int8_t v1 = (int8_t)(packed >> 4);
                
                // Sign extend (assuming 0-15 unsigned or -8..7 signed)
                // Let's assume -8 to 7 for this demo
                if (v0 > 7) v0 -= 16;
                if (v1 > 7) v1 -= 16;

                sum += (float)x[row * K + i] * ((float)v0 * s);
                sum += (float)x[row * K + i + 1] * ((float)v1 * s);
            }}
            out[gid] = (T)sum;
        }}
    """
    return metal_kernel(
        name=f"kk_dequant_int4_matmul_M{M}_N{N}_K{K}",
        input_names=["x", "weights", "scales"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def dequantize_int4_matmul(x: Any, w_int4: Any, scales: Any, *, compute_dtype: Any | None = None) -> Any:
    """Fused int4 dequantize + matmul.

    Args:
        x: Input array with shape ``(M, K)``.
        w_int4: Packed int4 weights with shape ``(N, K/2)`` (uint8).
        scales: Per-row scales with shape ``(N,)``.
        compute_dtype: Dtype used for internal computation.

    Returns:
        Output array with shape ``(M, N)``.
    """
    M, K = x.shape
    N, K_half = w_int4.shape
    if K != K_half * 2:
        raise ValueError("dequantize_int4_matmul: weight shape must be (N, K/2)")
    
    cd = compute_dtype or mx.float32
    k = _dequant_int4_matmul_kernel(M, N, K)
    return k(
        x, w_int4, scales,
        template=[("T", cd)],
        grid=(M * N, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[cd],
    )[0]

__all__ = [
    "fused_linear_bias_silu",
    "fused_linear_bias_gelu",
    "fused_linear_rmsnorm",
    "dequantize_int4_matmul",
]
