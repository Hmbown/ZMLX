from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .softmax import _validate_tg

_COMPUTE_DTYPE_DEPRECATION = (
    "compute_dtype is deprecated and will be removed in a future release. "
    "All ZMLX kernels compute internally in float32 regardless of this parameter."
)


@cache
def _rmsnorm_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)

    src = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            sumsq += v * v;
        }}
        buf[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                buf[tid] += buf[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float inv = metal::rsqrt(buf[0] / (float)D + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            float w = (float)weight[j];
            out[base + j] = (T)(v * inv * w);
        }}
    """

    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    return metal_kernel(
        name=f"kk_rmsnorm_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp", "weight"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def rmsnorm(
    x: Any,
    weight: Any,
    *,
    eps: float = 1e-6,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """RMSNorm over the last dimension, with per-channel weights.

    Shapes:
      - x: (..., D)
      - weight: (D,)
    """
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    if x.ndim < 1:
        raise ValueError("rmsnorm: x must have rank >= 1")
    D = int(x.shape[-1])
    if int(weight.ndim) != 1 or int(weight.shape[0]) != D:
        raise ValueError(f"rmsnorm: weight must have shape ({D},)")

    TG = _validate_tg(threadgroup)
    rows = x.size // D
    k_fwd = _rmsnorm_kernel(D, TG, float(eps))

    @mx.custom_function
    def op(x_in, w_in):
        return k_fwd(
            x_in,
            w_in,
            template=[("T", x.dtype)],
            grid=(rows * TG, 1, 1),
            threadgroup=(TG, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]

    @op.vjp
    def op_vjp(primals, cotangents, outputs):
        x_in, w_in = primals
        cotan = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
        dx = rmsnorm_grad(x_in, w_in, cotan, eps=eps, threadgroup=TG)
        return (dx, None)

    return op(x, weight)


@cache
def _rmsnorm_bwd_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    # This kernel computes dL/dx and dL/dw
    # Since dL/dw is a global sum across rows, we might need a separate kernel or atomics.
    # For now, let's focus on dL/dx (the per-element grad).
    
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // 1. Compute RMS of the row
        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            sumsq += v * v;
        }}
        buf[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float rms = metal::rsqrt(buf[0] / (float)D + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Compute sum(dL/dy * x * w * rms^3 / D) which is part of the grad
        // Actually simpler: let y_raw = x * rms (normalized without weight)
        // dL/dx = (dL/dy * w * rms) - (y_raw * rms * mean(dL/dy * w * y_raw))
        
        float m_dot = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float x = (float)inp[base + j];
            float g = (float)cotan[base + j];
            float w = (float)weight[j];
            float y_raw = x * rms;
            m_dot += g * w * y_raw;
        }}
        buf[tid] = m_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float mean_dot = buf[0] / (float)D;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Compute final dL/dx
        for (uint j = tid; j < D; j += TG) {{
            float x = (float)inp[base + j];
            float g = (float)cotan[base + j];
            float w = (float)weight[j];
            float y_raw = x * rms;
            dinp[base + j] = (T)(rms * (g * w - y_raw * mean_dot));
        }}
    """

    return metal_kernel(
        name=f"kk_rmsnorm_bwd_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp", "weight", "cotan"],
        output_names=["dinp"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def rmsnorm_grad(
    x: Any,
    weight: Any,
    cotan: Any,
    *,
    eps: float = 1e-6,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """Gradient of RMSNorm with respect to input x."""
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    D = int(x.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = x.size // D
    k = _rmsnorm_bwd_kernel(D, TG, float(eps))
    return k(
        x, weight, cotan,
        template=[("T", x.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _rmsnorm_no_weight_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    src = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            sumsq += v * v;
        }}
        buf[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                buf[tid] += buf[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float inv = metal::rsqrt(buf[0] / (float)D + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            out[base + j] = (T)(v * inv);
        }}
    """

    return metal_kernel(
        name=f"kk_rmsnorm_no_w_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def rms_norm_no_weight(x: Any, *, eps: float = 1e-6, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Pure RMSNorm without weights."""
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    rows = x.size // D
    k = _rmsnorm_no_weight_kernel(D, TG, float(eps))
    return k(
        x,
        template=[("T", x.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _layernorm_no_weight_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    src = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf_sum[TG];
        threadgroup float buf_sumsq[TG];

        float sum = 0.0f;
        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            sum += v;
            sumsq += v * v;
        }}
        buf_sum[tid] = sum;
        buf_sumsq[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                buf_sum[tid] += buf_sum[tid + stride];
                buf_sumsq[tid] += buf_sumsq[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float mean = buf_sum[0] / (float)D;
        float var = buf_sumsq[0] / (float)D - mean * mean;
        float inv = metal::rsqrt(var + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            out[base + j] = (T)((v - mean) * inv);
        }}
    """

    return metal_kernel(
        name=f"kk_layernorm_no_w_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def layer_norm_no_weight(x: Any, *, eps: float = 1e-5, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Pure LayerNorm without weights."""
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    rows = x.size // D
    k = _layernorm_no_weight_kernel(D, TG, float(eps))
    return k(
        x,
        template=[("T", x.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _layernorm_dropout_kernel(d: int, tg: int, eps: float, p: float, seed: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    p_f = float(p)
    p_str = str(p_f).replace(".", "_").replace("-", "_")

    src = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;
        constexpr float P = {p_f}f;
        constexpr uint32_t BASE_SEED = {int(seed)}u;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf_sum[TG];
        threadgroup float buf_sumsq[TG];

        float sum = 0.0f;
        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            sum += v;
            sumsq += v * v;
        }}
        buf_sum[tid] = sum;
        buf_sumsq[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                buf_sum[tid] += buf_sum[tid + stride];
                buf_sumsq[tid] += buf_sumsq[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float mean = buf_sum[0] / (float)D;
        float var = buf_sumsq[0] / (float)D - mean * mean;
        float inv = metal::rsqrt(var + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            float g = (float)gamma[j];
            float b = (float)beta[j];
            float normed = (v - mean) * inv * g + b;
            
            // RNG per element
            uint32_t s = BASE_SEED + base + j;
            s = s * 1664525u + 1013904223u;
            float r = (float)(s & 0xFFFFFFu) / 16777216.0f;
            
            if (r < P) {{
                out[base + j] = (T)0;
            }} else {{
                out[base + j] = (T)(normed / (1.0f - P));
            }}
        }}
    """

    return metal_kernel(
        name=f"kk_layernorm_dropout_D{D}_P{p_str}_S{seed}",
        input_names=["inp", "gamma", "beta"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def layer_norm_dropout(
    x: Any,
    gamma: Any,
    beta: Any,
    p: float,
    seed: int = 0,
    *,
    eps: float = 1e-5,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """Fused LayerNorm + Dropout."""
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    D = int(x.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = x.size // D
    k = _layernorm_dropout_kernel(D, TG, float(eps), float(p), int(seed))
    return k(
        x, gamma, beta,
        template=[("T", x.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]



@cache
def _layernorm_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)

    src = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf_sum[TG];
        threadgroup float buf_sumsq[TG];

        float sum = 0.0f;
        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            sum += v;
            sumsq += v * v;
        }}
        buf_sum[tid] = sum;
        buf_sumsq[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                buf_sum[tid] += buf_sum[tid + stride];
                buf_sumsq[tid] += buf_sumsq[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float mean = buf_sum[0] / (float)D;
        float var = buf_sumsq[0] / (float)D - mean * mean;
        float inv = metal::rsqrt(var + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            float g = (float)gamma[j];
            float b = (float)beta[j];
            out[base + j] = (T)(((v - mean) * inv) * g + b);
        }}
    """

    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    return metal_kernel(
        name=f"kk_layernorm_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp", "gamma", "beta"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def layernorm(
    x: Any,
    gamma: Any,
    beta: Any,
    *,
    eps: float = 1e-5,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """LayerNorm over the last dimension.

    Shapes:
      - x: (..., D)
      - gamma: (D,)
      - beta: (D,)
    """
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    if x.ndim < 1:
        raise ValueError("layernorm: x must have rank >= 1")
    D = int(x.shape[-1])
    if int(gamma.ndim) != 1 or int(gamma.shape[0]) != D:
        raise ValueError(f"layernorm: gamma must have shape ({D},)")
    if int(beta.ndim) != 1 or int(beta.shape[0]) != D:
        raise ValueError(f"layernorm: beta must have shape ({D},)")

    TG = _validate_tg(threadgroup)
    rows = 1
    for s in x.shape[:-1]:
        rows *= int(s)

    k = _layernorm_kernel(D, TG, float(eps))
    out = k(
        x,
        gamma,
        beta,
        template=[("T", x.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
    return out

@cache
def _residual_rmsnorm_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    src = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j] + (float)residual[base + j];
            sumsq += v * v;
        }}
        buf[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                buf[tid] += buf[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float inv = metal::rsqrt(buf[0] / (float)D + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j] + (float)residual[base + j];
            float w = (float)weight[j];
            out[base + j] = (T)(v * inv * w);
        }}
    """

    return metal_kernel(
        name=f"kk_res_rmsnorm_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp", "residual", "weight"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def residual_rmsnorm(
    x: Any,
    residual: Any,
    weight: Any,
    *,
    eps: float = 1e-6,
    threadgroup: int = 256,
) -> Any:
    """Fused Add + RMSNorm: out = RMSNorm(x + residual, weight)."""
    if x.ndim < 1:
        raise ValueError("residual_rmsnorm: x must have rank >= 1")
    D = int(x.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = x.size // D
    k = _residual_rmsnorm_kernel(D, TG, float(eps))
    return k(
        x, residual, weight,
        template=[("T", x.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


__all__ = [
    "rmsnorm",
    "rmsnorm_grad",
    "rms_norm_no_weight",
    "layer_norm_no_weight",
    "layer_norm_dropout",
    "layernorm",
    "residual_rmsnorm",
]
