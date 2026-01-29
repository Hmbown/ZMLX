from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

from .._compat import import_mx
from ..autograd import binary_from_expr
from ..elementwise import binary as binary_kernel
from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def add(*, compute_dtype_key: str = "float32") -> Callable[[Any, Any], Any]:
    """Build a Metal kernel for elementwise addition.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(a, b) -> a + b`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return binary_kernel(
        name=f"kk_add_{compute_dtype_key}",
        expr="a + b",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def mul(*, compute_dtype_key: str = "float32") -> Callable[[Any, Any], Any]:
    """Build a Metal kernel for elementwise multiplication.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(a, b) -> a * b`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return binary_kernel(
        name=f"kk_mul_{compute_dtype_key}",
        expr="a * b",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def silu_mul_grad(*, compute_dtype_key: str = "float32") -> Callable[[Any, Any], Any]:
    """Fused silu(a) * b with custom VJP for both inputs.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(a, b) -> silu(a) * b`` that supports ``mx.grad``.

    Notes:
        ``out = (a * sigmoid(a)) * b``.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16

    prelude = "T s = kk_sigmoid(a);"

    # forward uses a, b
    fwd_expr = "(a * s) * b"
    # grads:
    # d/da = g * b * (s + a*s*(1-s))
    # d/db = g * (a*s)
    vjp_lhs = "g * b * (s + a * s * ((T)1 - s))"
    vjp_rhs = "g * (a * s)"

    return binary_from_expr(
        name=f"kk_silu_mul_{compute_dtype_key}",
        fwd_expr=fwd_expr,
        vjp_lhs_expr=vjp_lhs,
        vjp_rhs_expr=vjp_rhs,
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
        vjp_prelude=prelude,
    )


def _bias_kernel_source(*, act_expr: str, c: int) -> str:
    C = int(c)
    return f"""
        constexpr uint C = {C};
        uint elem = thread_position_in_grid.x;
        uint col = elem % C;
        T x = inp[elem];
        T b = bias[col];
        T z = x + b;
        out[elem] = {act_expr};
    """


def add_bias(*, c: int, compute_dtype: Any) -> Callable[[Any, Any], Any]:
    """Add a 1D bias vector over the last dimension.

    Args:
        c: Last-dimension size (bias length).
        compute_dtype: MLX dtype for the template parameter ``T``.

    Returns:
        A callable ``f(x, bias) -> x + bias`` that runs on the GPU.

    Notes:
        - ``x`` has shape ``(..., C)`` and is flattened by MLX
          (``ensure_row_contiguous=True``).
        - ``bias`` has shape ``(C,)``.
    """
    source = _bias_kernel_source(act_expr="z", c=c)
    k = metal_kernel(
        name=f"kk_add_bias_C{int(c)}",
        input_names=["inp", "bias"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

    def op(x: Any, bias: Any) -> Any:
        if bias.ndim != 1:
            raise ValueError("add_bias: bias must be 1D")
        if int(bias.shape[0]) != int(c):
            raise ValueError(f"add_bias: expected bias length {int(c)} got {int(bias.shape[0])}")
        if x.shape[-1] != int(c):
            raise ValueError(f"add_bias: expected x.shape[-1]=={int(c)} got {x.shape[-1]}")

        out = k(
            x,
            bias,
            template=[("T", compute_dtype)],
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]
        return out

    return op


def bias_gelu_tanh(*, c: int, compute_dtype: Any) -> Callable[[Any, Any], Any]:
    """Build a fused bias-add + GeLU (tanh approx) kernel.

    Computes ``gelu_tanh(x + bias)`` in a single Metal dispatch.

    Args:
        c: Last-dimension size (bias length).
        compute_dtype: MLX dtype for the template parameter ``T``.

    Returns:
        A callable ``f(x, bias) -> gelu_tanh(x + bias)`` that runs on the GPU.
    """
    source = _bias_kernel_source(act_expr="kk_gelu_tanh(z)", c=c)
    k = metal_kernel(
        name=f"kk_bias_gelu_tanh_C{int(c)}",
        input_names=["inp", "bias"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

    def op(x: Any, bias: Any) -> Any:
        if bias.ndim != 1:
            raise ValueError("bias_gelu_tanh: bias must be 1D")
        if int(bias.shape[0]) != int(c):
            raise ValueError(f"bias_gelu_tanh: expected bias length {int(c)} got {int(bias.shape[0])}")
        if x.shape[-1] != int(c):
            raise ValueError(f"bias_gelu_tanh: expected x.shape[-1]=={int(c)} got {x.shape[-1]}")

        out = k(
            x,
            bias,
            template=[("T", compute_dtype)],
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]
        return out

    return op


def bias_silu(*, c: int, compute_dtype: Any) -> Callable[[Any, Any], Any]:
    """Build a fused bias-add + SiLU kernel.

    Computes ``silu(x + bias)`` in a single Metal dispatch.

    Args:
        c: Last-dimension size (bias length).
        compute_dtype: MLX dtype for the template parameter ``T``.

    Returns:
        A callable ``f(x, bias) -> silu(x + bias)`` that runs on the GPU.
    """
    source = _bias_kernel_source(act_expr="kk_silu(z)", c=c)
    k = metal_kernel(
        name=f"kk_bias_silu_C{int(c)}",
        input_names=["inp", "bias"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

    def op(x: Any, bias: Any) -> Any:
        if bias.ndim != 1:
            raise ValueError("bias_silu: bias must be 1D")
        if int(bias.shape[0]) != int(c):
            raise ValueError(f"bias_silu: expected bias length {int(c)} got {int(bias.shape[0])}")
        if x.shape[-1] != int(c):
            raise ValueError(f"bias_silu: expected x.shape[-1]=={int(c)} got {x.shape[-1]}")

        out = k(
            x,
            bias,
            template=[("T", compute_dtype)],
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]
        return out

    return op

__all__ = [
    "add",
    "mul",
    "silu_mul_grad",
    "add_bias",
    "bias_gelu_tanh",
    "bias_silu",
]
