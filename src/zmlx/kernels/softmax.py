from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import mlx.core as mx

from ..codegen import rowwise_mapreduce_source
from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER

_COMPUTE_DTYPE_DEPRECATION = (
    "compute_dtype is deprecated and will be removed in a future release. "
    "All ZMLX kernels compute internally in float32 regardless of this parameter."
)


def _validate_tg(tg: int) -> int:
    tg = int(tg)
    if tg <= 0 or tg > 1024:
        raise ValueError("threadgroup must be in (0, 1024]")
    # require power of two for the reduction loop
    if tg & (tg - 1) != 0:
        raise ValueError("threadgroup must be a power of two")
    return tg


@cache
def _softmax_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)

    src = rowwise_mapreduce_source(
        d=D,
        tg=TG,
        pass1_init="-INFINITY",
        pass1_update="metal::max(acc1, x)",
        pass1_reduce_op="metal::max(a, b)",
        pass2_init="0.0f",
        pass2_update="acc2 + metal::exp(x - s1)",
        pass2_reduce_op="a + b",
        write_expr="metal::exp(x - s1) / s2",
        inp="inp",
        out="out",
    )

    return metal_kernel(
        name=f"kk_softmax_D{D}_TG{TG}",
        input_names=["inp"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def softmax_lastdim(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Rowwise softmax over the last dimension.

    This kernel assumes `x` is row-contiguous (or will be made so by MLX if possible).
    """
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    if x.ndim < 1:
        raise ValueError("softmax_lastdim: x must have rank >= 1")
    D = int(x.shape[-1])
    if D <= 0:
        raise ValueError("softmax_lastdim: last dimension must be > 0")

    TG = _validate_tg(threadgroup)
    rows = x.size // D
    k_fwd = _softmax_kernel(D, TG)

    @mx.custom_function
    def op(x_in):
        return k_fwd(
            x_in,
            template=[("T", x.dtype)],
            grid=(rows * TG, 1, 1),
            threadgroup=(TG, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]

    @op.vjp
    def op_vjp(primals, cotangents, outputs):
        y = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        cotan = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
        dx = softmax_grad(y, cotan, threadgroup=TG)
        return (dx,)

    return op(x)


@cache
def _log_softmax_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)

    src = rowwise_mapreduce_source(
        d=D,
        tg=TG,
        pass1_init="-INFINITY",
        pass1_update="metal::max(acc1, x)",
        pass1_reduce_op="metal::max(a, b)",
        pass2_init="0.0f",
        pass2_update="acc2 + metal::exp(x - s1)",
        pass2_reduce_op="a + b",
        write_expr="(x - s1) - metal::log(s2)",
        inp="inp",
        out="out",
    )

    return metal_kernel(
        name=f"kk_log_softmax_D{D}_TG{TG}",
        input_names=["inp"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def log_softmax_lastdim(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Rowwise log-softmax over the last dimension."""
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    if x.ndim < 1:
        raise ValueError("log_softmax_lastdim: x must have rank >= 1")
    D = int(x.shape[-1])
    if D <= 0:
        raise ValueError("log_softmax_lastdim: last dimension must be > 0")

    TG = _validate_tg(threadgroup)
    rows = x.size // D
    k = _log_softmax_kernel(D, TG)

    return k(
        x,
        template=[("T", x.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _softmax_bwd_kernel(d: int, tg: int) -> Any:
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

        float dot = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float y = (float)out_val[base + j];
            float g = (float)cotan[base + j];
            dot += g * y;
        }}
        buf[tid] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float sum_gy = buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < D; j += TG) {{
            float y = (float)out_val[base + j];
            float g = (float)cotan[base + j];
            dinp[base + j] = (T)(y * (g - sum_gy));
        }}
    """
    return metal_kernel(
        name=f"kk_softmax_bwd_D{D}_TG{TG}",
        input_names=["out_val", "cotan"],
        output_names=["dinp"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def softmax_grad(y: Any, cotan: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Gradient of Softmax with respect to its input."""
    if compute_dtype is not None:
        warnings.warn(_COMPUTE_DTYPE_DEPRECATION, DeprecationWarning, stacklevel=2)
    D = int(y.shape[-1])
    TG = _validate_tg(threadgroup)
    rows = y.size // D
    k = _softmax_bwd_kernel(D, TG)
    return k(
        y, cotan,
        template=[("T", y.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[y.shape],
        output_dtypes=[y.dtype],
    )[0]

__all__ = [
    "softmax_lastdim",
    "log_softmax_lastdim",
    "softmax_grad",
]
