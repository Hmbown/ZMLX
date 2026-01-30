"""Linear-related kernel reference implementations.

.. note::

    These kernels use naive dot-product matmul (one thread per output element)
    and are **not competitive** with MPS-accelerated ``mx.matmul``.  They exist
    as **reference implementations** showing how to fuse post-linear operations
    (bias + activation, RMSNorm) into a single Metal kernel.

    For production workloads, prefer MLX's built-in linear layers and compose
    post-linear ops separately.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def _linear_bias_act_kernel(m: int, n: int, k: int, act_expr: str = "sum") -> Any:
    # M: batch/rows, N: out_features, K: in_features
    # This is a naive dot-product kernel (one thread per output element).
    # Reference implementation only — MPS matmul is orders of magnitude faster.
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
    """Fused Linear + Bias + SiLU (reference implementation).

    .. warning::

        This uses a naive dot-product matmul and is **much slower** than
        ``mx.matmul`` + ``mx.nn.silu``.  Use only for codegen demonstration
        or testing.

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
    """Fused Linear + Bias + GeLU (reference implementation).

    .. warning::

        This uses a naive dot-product matmul and is **much slower** than
        ``mx.matmul`` + ``mx.nn.gelu``.  Use only for codegen demonstration
        or testing.
    """
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

    TG = 256

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

def fused_linear_rmsnorm(
    x: Any, w: Any, gamma: Any, *, eps: float = 1e-6, compute_dtype: Any | None = None
) -> Any:
    """Fused Linear + RMSNorm (reference implementation).

    .. warning::

        This uses a naive dot-product matmul and is **much slower** than
        ``mx.matmul`` followed by ``mx.fast.rms_norm``.  Use only for
        codegen demonstration or testing.

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


__all__ = [
    "fused_linear_bias_silu",
    "fused_linear_bias_gelu",
    "fused_linear_rmsnorm",
]
