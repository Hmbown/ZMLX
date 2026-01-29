from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .softmax import _validate_tg


@cache
def _swiglu_fwd_kernel(d: int) -> Any:
    D = int(d)
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / {D};
        uint col = gid % {D};
        uint base = row * {2 * D};
        
        T a = inp[base + col];
        T b = inp[base + col + {D}];
        out[gid] = a * (T(1.0) / (T(1.0) + metal::exp(-a))) * b;
    """
    return metal_kernel(
        name=f"kk_swiglu_fwd_D{D}",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _swiglu_bwd_kernel(d: int) -> Any:
    D = int(d)
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / {D};
        uint col = gid % {D};
        uint base = row * {2 * D};
        
        T a = inp[base + col];
        T b = inp[base + col + {D}];
        T g = cotan[gid];
        
        T s = T(1.0) / (T(1.0) + metal::exp(-a));
        T swish = a * s;
        
        // d/da (a * s * b) = b * (s + a * s * (1 - s))
        // d/db (a * s * b) = a * s
        dinp[base + col] = g * b * (s + swish * (T(1.0) - s));
        dinp[base + col + {D}] = g * swish;
    """
    return metal_kernel(
        name=f"kk_swiglu_bwd_D{D}",
        input_names=["inp", "cotan"],
        output_names=["dinp"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def swiglu(x: Any, *, compute_dtype: Any | None = None) -> Any:
    """Fused SwiGLU activation.
    
    x: (..., 2*D)
    returns: (..., D)
    
    y = SiLU(x[..., :D]) * x[..., D:]
    """
    if x.ndim < 1:
        raise ValueError("swiglu: x must have rank >= 1")
    D2 = x.shape[-1]
    if D2 % 2 != 0:
        raise ValueError("swiglu: last dimension must be even")
    D = D2 // 2
    
    rows = x.size // D2
    cd = compute_dtype or mx.float32
    
    k_fwd = _swiglu_fwd_kernel(D)
    
    @mx.custom_function
    def op(inputs):
        return k_fwd(
            inputs,
            template=[("T", cd)],
            grid=(rows * D, 1, 1),
            output_shapes=[x.shape[:-1] + (D,)],
            output_dtypes=[x.dtype],
        )[0]

    @op.vjp
    def op_vjp(primals, cotangents, outputs):
        inp = primals[0]
        cotan = cotangents[0]
        k_bwd = _swiglu_bwd_kernel(D)
        
        dx = k_bwd(
            inp,
            cotan,
            template=[("T", cd)],
            grid=(rows * D, 1, 1),
            output_shapes=[inp.shape],
            output_dtypes=[inp.dtype],
        )[0]
        return (dx,)

    return op(x)


@cache
def _geglu_fwd_kernel(d: int) -> Any:
    D = int(d)
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / {D};
        uint col = gid % {D};
        uint base = row * {2 * D};
        
        T a = inp[base + col];
        T b = inp[base + col + {D}];
        out[gid] = kk_gelu_tanh(a) * b;
    """
    return metal_kernel(
        name=f"kk_geglu_fwd_D{D}",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _geglu_bwd_kernel(d: int) -> Any:
    D = int(d)
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / {D};
        uint col = gid % {D};
        uint base = row * {2 * D};
        
        T x = inp[base + col];
        T b = inp[base + col + {D}];
        T g = cotan[gid];
        
        // Gelu Tanh grad
        const T k0 = (T)0.7978845608028654;
        const T k1 = (T)0.044715;
        T x2 = x * x;
        T x3 = x2 * x;
        T u = k0 * (x + k1 * x3);
        T t = metal::tanh(u);
        T du = k0 * ((T)1 + (T)3 * k1 * x2);
        T dy = (T)0.5 * ((T)1 + t) + (T)0.5 * x * ((T)1 - t * t) * du;
        
        T gelu = (T)0.5 * x * ((T)1 + t);
        
        dinp[base + col] = g * b * dy;
        dinp[base + col + {D}] = g * gelu;
    """
    return metal_kernel(
        name=f"kk_geglu_bwd_D{D}",
        input_names=["inp", "cotan"],
        output_names=["dinp"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def geglu(x: Any, *, compute_dtype: Any | None = None) -> Any:
    """Fused GeGLU activation.
    
    x: (..., 2*D)
    returns: (..., D)
    
    y = GeLU(x[..., :D]) * x[..., D:]
    """
    if x.ndim < 1:
        raise ValueError("geglu: x must have rank >= 1")
    D2 = x.shape[-1]
    if D2 % 2 != 0:
        raise ValueError("geglu: last dimension must be even")
    D = D2 // 2
    
    rows = x.size // D2
    cd = compute_dtype or mx.float32
    
    k_fwd = _geglu_fwd_kernel(D)
    
    @mx.custom_function
    def op(inputs):
        return k_fwd(
            inputs,
            template=[("T", cd)],
            grid=(rows * D, 1, 1),
            output_shapes=[x.shape[:-1] + (D,)],
            output_dtypes=[x.dtype],
        )[0]

    @op.vjp
    def op_vjp(primals, cotangents, outputs):
        inp = primals[0]
        cotan = cotangents[0]
        k_bwd = _geglu_bwd_kernel(D)
        
        dx = k_bwd(
            inp,
            cotan,
            template=[("T", cd)],
            grid=(rows * D, 1, 1),
            output_shapes=[inp.shape],
            output_dtypes=[inp.dtype],
        )[0]
        return (dx,)

    return op(x)


@cache
def _rmsnorm_residual_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // x = x + residual
        // sumsq(x)
        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float val = (float)inp[base + j] + (float)res[base + j];
            // Write back updated residual if desired, or just use it
            // For this kernel, we output the normalized x. 
            // We'll also output the updated residual (x + res) so it can be used by next layer.
            updated_res[base + j] = (T)val;
            sumsq += val * val;
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
            float val = (float)updated_res[base + j];
            float w = (float)weight[j];
            out[base + j] = (T)(val * inv * w);
        }}
    """

    return metal_kernel(
        name=f"kk_rmsnorm_res_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp", "res", "weight"],
        output_names=["out", "updated_res"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def rmsnorm_residual(
    x: Any,
    residual: Any,
    weight: Any,
    *,
    eps: float = 1e-6,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """Fused RMSNorm(x + residual) and updated residual x + residual.
    
    Returns (normed_x, updated_residual).
    """
    if x.ndim < 1:
        raise ValueError("rmsnorm_residual: x must have rank >= 1")
    D = int(x.shape[-1])
    if x.shape != residual.shape:
        raise ValueError("rmsnorm_residual: x and residual must have same shape")
    if int(weight.ndim) != 1 or int(weight.shape[0]) != D:
        raise ValueError(f"rmsnorm_residual: weight must have shape ({D},)")

    TG = _validate_tg(threadgroup)
    rows = x.size // D

    cd = compute_dtype or mx.float32
    k = _rmsnorm_residual_kernel(D, TG, float(eps))
    
    out, updated_res = k(
        x,
        residual,
        weight,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return out, updated_res


@cache
def _fused_add_rmsnorm_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    source = f"""
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
            float val = (float)inp1[base + j] + (float)inp2[base + j];
            sumsq += val * val;
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
            float val = (float)inp1[base + j] + (float)inp2[base + j];
            float w = (float)weight[j];
            out[base + j] = (T)(val * inv * w);
        }}
    """

    return metal_kernel(
        name=f"kk_fused_add_rmsnorm_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp1", "inp2", "weight"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def fused_add_rmsnorm(
    x1: Any,
    x2: Any,
    weight: Any,
    *,
    eps: float = 1e-6,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """Fused RMSNorm(x1 + x2) returning only the normalized output."""
    D = int(x1.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = x1.size // D
    cd = compute_dtype or mx.float32
    k = _fused_add_rmsnorm_kernel(D, TG, float(eps))
    return k(
        x1, x2, weight,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x1.shape],
        output_dtypes=[x1.dtype],
    )[0]


@cache
def _layernorm_residual_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    source = f"""
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
            float val = (float)inp[base + j] + (float)res[base + j];
            updated_res[base + j] = (T)val;
            sum += val;
            sumsq += val * val;
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
            float val = (float)updated_res[base + j];
            float g = (float)gamma[j];
            float b = (float)beta[j];
            out[base + j] = (T)((val - mean) * inv * g + b);
        }}
    """

    return metal_kernel(
        name=f"kk_layernorm_res_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp", "res", "gamma", "beta"],
        output_names=["out", "updated_res"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def layernorm_residual(
    x: Any,
    residual: Any,
    gamma: Any,
    beta: Any,
    *,
    eps: float = 1e-5,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """Fused LayerNorm(x + residual) and updated residual x + residual."""
    D = int(x.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    k = _layernorm_residual_kernel(D, TG, float(eps))
    
    out, updated_res = k(
        x, residual, gamma, beta,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
    return out, updated_res


@cache
def _fused_add_layernorm_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    source = f"""
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
            float val = (float)inp1[base + j] + (float)inp2[base + j];
            sum += val;
            sumsq += val * val;
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
            float val = (float)inp1[base + j] + (float)inp2[base + j];
            float g = (float)gamma[j];
            float b = (float)beta[j];
            out[base + j] = (T)((val - mean) * inv * g + b);
        }}
    """

    return metal_kernel(
        name=f"kk_fused_add_layernorm_D{D}_TG{TG}_E{eps_str}",
        input_names=["inp1", "inp2", "gamma", "beta"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def fused_add_layernorm(
    x1: Any,
    x2: Any,
    gamma: Any,
    beta: Any,
    *,
    eps: float = 1e-5,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """Fused LayerNorm(x1 + x2) returning only the normalized output."""
    D = int(x1.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = x1.size // D
    cd = compute_dtype or mx.float32
    k = _fused_add_layernorm_kernel(D, TG, float(eps))
    return k(
        x1, x2, gamma, beta,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x1.shape],
        output_dtypes=[x1.dtype],
    )[0]


@cache
def _rmsnorm_dropout_kernel(d: int, tg: int, eps: float, p: float, seed: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)
    p_f = float(p)
    p_str = str(p_f).replace(".", "_").replace("-", "_")

    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;
        constexpr float P = {p_f}f;
        constexpr uint32_t BASE_SEED = {int(seed)}u;

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

        // 2. Normalize and Apply Dropout
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            float w = (float)weight[j];
            float normed = v * rms * w;
            
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
        name=f"kk_rmsnorm_dropout_D{D}_P{p_str}_S{seed}",
        input_names=["inp", "weight"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def rms_norm_dropout(
    x: Any,
    weight: Any,
    p: float,
    seed: int = 0,
    *,
    eps: float = 1e-6,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """Fused RMSNorm + Dropout."""
    D = int(x.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    k = _rmsnorm_dropout_kernel(D, TG, float(eps), float(p), int(seed))
    return k(
        x, weight,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _dropout_kernel(p: float, seed: int) -> Any:
    # Simple LCG for demonstration; in production use better RNG
    source = f"""
        uint gid = thread_position_in_grid.x;
        float p = {float(p)}f;
        uint32_t seed = {int(seed)}u + gid;
        
        // Simple LCG
        seed = seed * 1664525u + 1013904223u;
        float r = (float)(seed & 0xFFFFFFu) / 16777216.0f;
        
        T x = inp[gid];
        if (r < p) {{
            out[gid] = (T)0;
        }} else {{
            out[gid] = (T)(x / (1.0f - p));
        }}
    """
    p_str = str(p).replace(".", "_").replace("-", "_")
    return metal_kernel(
        name=f"kk_dropout_p{p_str}_s{seed}",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def dropout(x: Any, p: float, seed: int = 0, *, compute_dtype: Any | None = None) -> Any:
    """Dropout with scaling.
    
    p: probability of dropping an element.
    """
    if p < 0 or p >= 1:
        raise ValueError("dropout: p must be in [0, 1)")
    
    cd = compute_dtype or mx.float32
    k = _dropout_kernel(float(p), int(seed))
    
    return k(
        x,
        template=[("T", cd)],
        grid=(x.size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _bias_swiglu_kernel(d: int) -> Any:
    D = int(d)
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / {D};
        uint col = gid % {D};
        uint base = row * {2 * D};
        
        T a = inp[base + col] + bias[col];
        T b = inp[base + col + {D}] + bias[col + {D}];
        out[gid] = a * (T(1.0) / (T(1.0) + metal::exp(-a))) * b;
    """
    return metal_kernel(
        name=f"kk_bias_swiglu_D{D}",
        input_names=["inp", "bias"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _bias_swiglu_bwd_kernel(d: int) -> Any:
    D = int(d)
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / {D};
        uint col = gid % {D};
        uint base = row * {2 * D};
        
        T a = inp[base + col] + bias[col];
        T b = inp[base + col + {D}] + bias[col + {D}];
        T g = cotan[gid];
        
        T s = T(1.0) / (T(1.0) + metal::exp(-a));
        T swish = a * s;
        
        T da = g * b * (s + swish * (T(1.0) - s));
        T db = g * swish;
        
        dinp[base + col] = da;
        dinp[base + col + {D}] = db;
    """
    return metal_kernel(
        name=f"kk_bias_swiglu_bwd_D{D}",
        input_names=["inp", "bias", "cotan"],
        output_names=["dinp"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def bias_swiglu(x: Any, bias: Any, *, compute_dtype: Any | None = None) -> Any:
    """Fused Bias + SwiGLU (differentiable wrt x)."""
    D2 = x.shape[-1]
    D = D2 // 2
    rows = x.size // D2
    cd = compute_dtype or mx.float32
    k_fwd = _bias_swiglu_kernel(D)
    
    @mx.custom_function
    def op(x_in, b_in):
        return k_fwd(
            x_in, b_in,
            template=[("T", cd)],
            grid=(rows * D, 1, 1),
            output_shapes=[x.shape[:-1] + (D,)],
            output_dtypes=[x.dtype],
        )[0]

    @op.vjp
    def op_vjp(primals, cotangents, outputs):
        x_in, b_in = primals
        cotan = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
        k_bwd = _bias_swiglu_bwd_kernel(D)
        dx = k_bwd(
            x_in, b_in, cotan,
            template=[("T", cd)],
            grid=(rows * D, 1, 1),
            output_shapes=[x_in.shape],
            output_dtypes=[x_in.dtype],
        )[0]
        # return dx, None (no grad for bias yet to keep it simple)
        return (dx, None)

    return op(x, bias)


@cache
def _bias_geglu_kernel(d: int) -> Any:
    D = int(d)
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / {D};
        uint col = gid % {D};
        uint base = row * {2 * D};
        
        T a = inp[base + col] + bias[col];
        T b = inp[base + col + {D}] + bias[col + {D}];
        out[gid] = kk_gelu_tanh(a) * b;
    """
    return metal_kernel(
        name=f"kk_bias_geglu_D{D}",
        input_names=["inp", "bias"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _bias_geglu_bwd_kernel(d: int) -> Any:
    D = int(d)
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint row = gid / {D};
        uint col = gid % {D};
        uint base = row * {2 * D};
        
        T x_val = inp[base + col] + bias[col];
        T b_val = inp[base + col + {D}] + bias[col + {D}];
        T g = cotan[gid];
        
        // Gelu Tanh grad
        const T k0 = (T)0.7978845608028654;
        const T k1 = (T)0.044715;
        T x2 = x_val * x_val;
        T x3 = x2 * x_val;
        T u = k0 * (x_val + k1 * x3);
        T t = metal::tanh(u);
        T du = k0 * ((T)1 + (T)3 * k1 * x2);
        T dy = (T)0.5 * ((T)1 + t) + (T)0.5 * x_val * ((T)1 - t * t) * du;
        
        T gelu = (T)0.5 * x_val * ((T)1 + t);
        
        dinp[base + col] = g * b_val * dy;
        dinp[base + col + {D}] = g * gelu;
    """
    return metal_kernel(
        name=f"kk_bias_geglu_bwd_D{D}",
        input_names=["inp", "bias", "cotan"],
        output_names=["dinp"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def bias_geglu(x: Any, bias: Any, *, compute_dtype: Any | None = None) -> Any:
    """Fused Bias + GeGLU (differentiable wrt x)."""
    D2 = x.shape[-1]
    D = D2 // 2
    rows = x.size // D2
    cd = compute_dtype or mx.float32
    k_fwd = _bias_geglu_kernel(D)
    
    @mx.custom_function
    def op(x_in, b_in):
        return k_fwd(
            x_in, b_in,
            template=[("T", cd)],
            grid=(rows * D, 1, 1),
            output_shapes=[x.shape[:-1] + (D,)],
            output_dtypes=[x.dtype],
        )[0]

    @op.vjp
    def op_vjp(primals, cotangents, outputs):
        x_in, b_in = primals
        cotan = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
        k_bwd = _bias_geglu_bwd_kernel(D)
        dx = k_bwd(
            x_in, b_in, cotan,
            template=[("T", cd)],
            grid=(rows * D, 1, 1),
            output_shapes=[x_in.shape],
            output_dtypes=[x_in.dtype],
        )[0]
        return (dx, None)

    return op(x, bias)

__all__ = [
    "swiglu",
    "geglu",
    "rmsnorm_residual",
    "fused_add_rmsnorm",
    "layernorm_residual",
    "fused_add_layernorm",
    "rms_norm_dropout",
    "dropout",
    "bias_swiglu",
    "bias_geglu",
]
