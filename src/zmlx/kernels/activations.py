from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

from .._compat import import_mx
from ..autograd import unary_from_expr
from ..elementwise import unary as unary_kernel
from ..msl import DEFAULT_HEADER


def _dtype_key(dt: Any) -> str:
    return getattr(dt, "name", str(dt))


@cache
def exp(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise exponential.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> exp(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_kernel(
        name=f"kk_exp_{compute_dtype_key}",
        expr="metal::exp(x)",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def log(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise natural logarithm.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> log(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_kernel(
        name=f"kk_log_{compute_dtype_key}",
        expr="metal::log(x)",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def tanh(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise hyperbolic tangent.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> tanh(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_kernel(
        name=f"kk_tanh_{compute_dtype_key}",
        expr="metal::tanh(x)",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def sigmoid(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise sigmoid.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> 1 / (1 + exp(-x))`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_kernel(
        name=f"kk_sigmoid_{compute_dtype_key}",
        expr="kk_sigmoid(x)",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def relu(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise ReLU.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> max(x, 0)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_kernel(
        name=f"kk_relu_{compute_dtype_key}",
        expr="metal::max(x, (T)0)",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def silu(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise SiLU (swish).

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> x * sigmoid(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_kernel(
        name=f"kk_silu_{compute_dtype_key}",
        expr="kk_silu(x)",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def gelu_tanh(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise GeLU (tanh approximation).

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> gelu_tanh(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_kernel(
        name=f"kk_gelu_tanh_{compute_dtype_key}",
        expr="kk_gelu_tanh(x)",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def softplus(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise softplus.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> log(exp(x) + 1)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_kernel(
        name=f"kk_softplus_{compute_dtype_key}",
        expr="metal::log(metal::exp(x) + (T)1.0)",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def mish(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise mish activation.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> x * tanh(softplus(x))`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    # mish(x) = x * tanh(softplus(x))
    return unary_kernel(
        name=f"kk_mish_{compute_dtype_key}",
        expr="x * metal::tanh(metal::log(metal::exp(x) + (T)1.0))",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


@cache
def elu(*, alpha: float = 1.0, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a Metal kernel for elementwise ELU.

    Args:
        alpha: Scale for the negative region. Default ``1.0``.
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> x if x > 0 else alpha * (exp(x) - 1)``
        that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    alpha_str = str(alpha).replace(".", "_").replace("-", "_")
    return unary_kernel(
        name=f"kk_elu_{alpha_str}_{compute_dtype_key}",
        expr=f"(x > (T)0) ? x : (T){float(alpha)} * (metal::exp(x) - (T)1.0)",
        compute_dtype=compute_dtype,
        header=DEFAULT_HEADER,
    )


# ---- Gradient-enabled variants (custom VJP) ----

@cache
def exp_grad(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a differentiable Metal kernel for elementwise exponential.

    Supports ``mx.grad`` via a custom VJP.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> exp(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_from_expr(
        name=f"kk_exp_grad_{compute_dtype_key}",
        fwd_expr="metal::exp(x)",
        vjp_expr="g * y",
        compute_dtype=compute_dtype,
        use_output=True,
        header=DEFAULT_HEADER,
    )


@cache
def tanh_grad(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a differentiable Metal kernel for elementwise tanh.

    Supports ``mx.grad`` via a custom VJP.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> tanh(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_from_expr(
        name=f"kk_tanh_grad_{compute_dtype_key}",
        fwd_expr="metal::tanh(x)",
        vjp_expr="g * ((T)1 - y * y)",
        compute_dtype=compute_dtype,
        use_output=True,
        header=DEFAULT_HEADER,
    )


@cache
def sigmoid_grad(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a differentiable Metal kernel for elementwise sigmoid.

    Supports ``mx.grad`` via a custom VJP.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> sigmoid(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_from_expr(
        name=f"kk_sigmoid_grad_{compute_dtype_key}",
        fwd_expr="kk_sigmoid(x)",
        vjp_expr="g * y * ((T)1 - y)",
        compute_dtype=compute_dtype,
        use_output=True,
        header=DEFAULT_HEADER,
    )


@cache
def relu_grad(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a differentiable Metal kernel for elementwise ReLU.

    Supports ``mx.grad`` via a custom VJP.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> max(x, 0)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    # Use input x (not output) to avoid ambiguity at x==0
    return unary_from_expr(
        name=f"kk_relu_grad_{compute_dtype_key}",
        fwd_expr="metal::max(x, (T)0)",
        vjp_expr="(x > (T)0) ? g : (T)0",
        compute_dtype=compute_dtype,
        use_output=False,
        header=DEFAULT_HEADER,
    )


@cache
def silu_grad(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a differentiable Metal kernel for elementwise SiLU.

    Supports ``mx.grad`` via a custom VJP.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> x * sigmoid(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    prelude = "T s = kk_sigmoid(x);"
    return unary_from_expr(
        name=f"kk_silu_grad_{compute_dtype_key}",
        fwd_expr="x * kk_sigmoid(x)",
        vjp_expr="g * (s + x * s * ((T)1 - s))",
        compute_dtype=compute_dtype,
        use_output=False,
        header=DEFAULT_HEADER,
        vjp_prelude=prelude,
    )


@cache
def gelu_tanh_grad(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a differentiable Metal kernel for elementwise GeLU (tanh approx).

    Supports ``mx.grad`` via a custom VJP.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> gelu_tanh(x)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    # d/dx [0.5*x*(1+tanh(u))] where u = k0*(x + k1*x^3)
    prelude = r"""
        const T k0 = (T)0.7978845608028654;
        const T k1 = (T)0.044715;
        T x2 = x * x;
        T x3 = x2 * x;
        T u = k0 * (x + k1 * x3);
        T t = metal::tanh(u);
        T du = k0 * ((T)1 + (T)3 * k1 * x2);
        T dy = (T)0.5 * ((T)1 + t) + (T)0.5 * x * ((T)1 - t * t) * du;
    """
    return unary_from_expr(
        name=f"kk_gelu_tanh_grad_{compute_dtype_key}",
        fwd_expr="kk_gelu_tanh(x)",
        vjp_expr="g * dy",
        compute_dtype=compute_dtype,
        use_output=False,
        header=DEFAULT_HEADER,
        vjp_prelude=prelude,
    )


@cache
def softplus_grad(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a differentiable Metal kernel for elementwise softplus.

    Supports ``mx.grad`` via a custom VJP.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> log(exp(x) + 1)`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    return unary_from_expr(
        name=f"kk_softplus_grad_{compute_dtype_key}",
        fwd_expr="metal::log(metal::exp(x) + (T)1.0)",
        vjp_expr="g * kk_sigmoid(x)",
        compute_dtype=compute_dtype,
        use_output=False,
        header=DEFAULT_HEADER,
    )


@cache
def mish_grad(*, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a differentiable Metal kernel for elementwise mish.

    Supports ``mx.grad`` via a custom VJP.

    Args:
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> x * tanh(softplus(x))`` that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    # d/dx mish(x) = exp(x)*omega / delta^2
    # where omega = 4(x+1) + 4exp(2x) + exp(3x) + exp(x)(4x+6)
    # delta = 2exp(x) + exp(2x) + 2
    # Simple version: use sigmoid and tanh
    prelude = """
        T sp = metal::log(metal::exp(x) + (T)1.0);
        T tsp = metal::tanh(sp);
        T s = kk_sigmoid(x);
        T dy = tsp + x * s * ((T)1.0 - tsp * tsp);
    """
    return unary_from_expr(
        name=f"kk_mish_grad_{compute_dtype_key}",
        fwd_expr="x * metal::tanh(metal::log(metal::exp(x) + (T)1.0))",
        vjp_expr="g * dy",
        compute_dtype=compute_dtype,
        use_output=False,
        header=DEFAULT_HEADER,
        vjp_prelude=prelude,
    )


@cache
def elu_grad(*, alpha: float = 1.0, compute_dtype_key: str = "float32") -> Callable[[Any], Any]:
    """Build a differentiable Metal kernel for elementwise ELU.

    Supports ``mx.grad`` via a custom VJP.

    Args:
        alpha: Scale for the negative region. Default ``1.0``.
        compute_dtype_key: Precision for internal computation.
            ``"float32"`` (default) or ``"float16"``.

    Returns:
        A callable ``f(x) -> x if x > 0 else alpha * (exp(x) - 1)``
        that runs on the GPU.
    """
    mx = import_mx()
    compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
    alpha_str = str(alpha).replace(".", "_").replace("-", "_")
    return unary_from_expr(
        name=f"kk_elu_grad_{alpha_str}_{compute_dtype_key}",
        fwd_expr=f"(x > (T)0) ? x : (T){float(alpha)} * (metal::exp(x) - (T)1.0)",
        vjp_expr=f"(x > (T)0) ? g : g * (T){float(alpha)} * metal::exp(x)",
        compute_dtype=compute_dtype,
        use_output=False,
        header=DEFAULT_HEADER,
    )

__all__ = [
    "exp",
    "log",
    "tanh",
    "sigmoid",
    "relu",
    "silu",
    "gelu_tanh",
    "softplus",
    "mish",
    "elu",
    "exp_grad",
    "tanh_grad",
    "sigmoid_grad",
    "relu_grad",
    "silu_grad",
    "gelu_tanh_grad",
    "softplus_grad",
    "mish_grad",
    "elu_grad",
]
