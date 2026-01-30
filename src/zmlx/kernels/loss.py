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
def _softmax_ce_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // 1. Max pass
        float m = -INFINITY;
        for (uint j = tid; j < D; j += TG) {{
            m = metal::max(m, (float)logits[base + j]);
        }}
        buf[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] = metal::max(buf[tid], buf[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float max_val = buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Sum(exp) pass
        float sum_exp = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            sum_exp += metal::exp((float)logits[base + j] - max_val);
        }}
        buf[tid] = sum_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float log_sum_exp = max_val + metal::log(buf[0]);
        
        // 3. Final loss calculation (only thread 0)
        if (tid == 0) {{
            uint target_idx = targets[row];
            float target_logit = (float)logits[base + target_idx];
            out[row] = (T)(log_sum_exp - target_logit);
        }}
    """
    return metal_kernel(
        name=f"kk_softmax_ce_D{D}_TG{TG}",
        input_names=["logits", "targets"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

@cache
def _softmax_ce_bwd_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)

    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // 1. Max pass
        float m = -INFINITY;
        for (uint j = tid; j < D; j += TG) {{
            m = metal::max(m, (float)logits[base + j]);
        }}
        buf[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] = metal::max(buf[tid], buf[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float max_val = buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Sum(exp) pass
        float sum_exp = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            sum_exp += metal::exp((float)logits[base + j] - max_val);
        }}
        buf[tid] = sum_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float inv_sum = 1.0f / buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Compute gradient: (softmax - one_hot) * cotan
        uint target_idx = targets[row];
        float c = (float)cotan[row];
        for (uint j = tid; j < D; j += TG) {{
            float y = metal::exp((float)logits[base + j] - max_val) * inv_sum;
            float grad = (j == target_idx) ? (y - 1.0f) : y;
            dlogits[base + j] = (T)(grad * c);
        }}
    """

    return metal_kernel(
        name=f"kk_softmax_ce_bwd_D{D}_TG{TG}",
        input_names=["logits", "targets", "cotan"],
        output_names=["dlogits"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def softmax_cross_entropy(
    logits: Any,
    targets: Any,
    *,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """Fused Softmax + Cross Entropy Loss with custom VJP.

    logits: (..., D)
    targets: (...) - indices (uint32)
    """
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    D = logits.shape[-1]
    TG = _validate_tg(threadgroup)
    rows = logits.size // D
    cd = compute_dtype or mx.float32
    targets_u32 = targets.astype(mx.uint32)
    k_fwd = _softmax_ce_kernel(D, TG)

    @mx.custom_function
    def op(logits_in, targets_in):
        return k_fwd(
            logits_in,
            targets_in,
            template=[("T", cd)],
            grid=(rows * TG, 1, 1),
            threadgroup=(TG, 1, 1),
            output_shapes=[logits.shape[:-1]],
            output_dtypes=[cd],
        )[0]

    @op.vjp
    def op_vjp(primals, cotangents, outputs):
        logits_in, targets_in = primals
        cotan = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
        dlogits = softmax_cross_entropy_grad(
            logits_in,
            targets_in,
            cotan,
            threadgroup=TG,
        )
        return (dlogits, None)

    return op(logits, targets_u32)


def softmax_cross_entropy_grad(
    logits: Any,
    targets: Any,
    cotan: Any,
    *,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """Gradient of softmax cross-entropy with respect to logits."""
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    D = logits.shape[-1]
    TG = _validate_tg(threadgroup)
    rows = logits.size // D
    k = _softmax_ce_bwd_kernel(D, TG)
    return k(
        logits,
        targets,
        cotan,
        template=[("T", logits.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
    )[0]

__all__ = [
    "softmax_cross_entropy",
    "softmax_cross_entropy_grad",
]
