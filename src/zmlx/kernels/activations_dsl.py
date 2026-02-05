"""Activations defined via the DSL — replaces ~500 lines with ~60.

Every kernel here produces identical Metal to the legacy ``activations.py``
builders.  The ``_legacy`` prefix is available in ``activations.py`` for
side-by-side comparison during migration validation.
"""

from __future__ import annotations

from ..dsl.defkernel import defkernel

# ---------------------------------------------------------------------------
# Forward-only activations
# ---------------------------------------------------------------------------

exp = defkernel("kk_exp", "metal::exp(x)")
log = defkernel("kk_log", "metal::log(x)")
tanh = defkernel("kk_tanh", "metal::tanh(x)")
sigmoid = defkernel("kk_sigmoid", "kk_sigmoid(x)")
relu = defkernel("kk_relu", "metal::max(x, (T)0)")
relu2 = defkernel("kk_relu2", "metal::max(x, (T)0) * metal::max(x, (T)0)")
silu = defkernel("kk_silu", "kk_silu(x)")
gelu_tanh = defkernel("kk_gelu_tanh", "kk_gelu_tanh(x)")
softplus = defkernel("kk_softplus", "metal::log(metal::exp(x) + (T)1.0)")
mish = defkernel(
    "kk_mish",
    "x * metal::tanh(metal::log(metal::exp(x) + (T)1.0))",
)
elu = defkernel(
    "kk_elu",
    "(x > (T)0) ? x : (T)1.0 * (metal::exp(x) - (T)1.0)",
)

# ---------------------------------------------------------------------------
# Gradient-enabled activations (custom VJP)
# ---------------------------------------------------------------------------

exp_grad = defkernel(
    "kk_exp_grad",
    "metal::exp(x)",
    vjp_expr="g * y",
    use_output=True,
)

tanh_grad = defkernel(
    "kk_tanh_grad",
    "metal::tanh(x)",
    vjp_expr="g * ((T)1 - y * y)",
    use_output=True,
)

sigmoid_grad = defkernel(
    "kk_sigmoid_grad",
    "kk_sigmoid(x)",
    vjp_expr="g * y * ((T)1 - y)",
    use_output=True,
)

relu_grad = defkernel(
    "kk_relu_grad",
    "metal::max(x, (T)0)",
    vjp_expr="(x > (T)0) ? g : (T)0",
    use_output=False,
)

relu2_grad = defkernel(
    "kk_relu2_grad",
    "metal::max(x, (T)0) * metal::max(x, (T)0)",
    vjp_expr="(x > (T)0) ? (T)2 * x * g : (T)0",
    use_output=False,
)

silu_grad = defkernel(
    "kk_silu_grad",
    "x * kk_sigmoid(x)",
    vjp_expr="g * (s + x * s * ((T)1 - s))",
    vjp_prelude="T s = kk_sigmoid(x);",
    use_output=False,
)

gelu_tanh_grad = defkernel(
    "kk_gelu_tanh_grad",
    "kk_gelu_tanh(x)",
    vjp_expr="g * dy",
    vjp_prelude=r"""
        const T k0 = (T)0.7978845608028654;
        const T k1 = (T)0.044715;
        T x2 = x * x;
        T x3 = x2 * x;
        T u = k0 * (x + k1 * x3);
        T t = metal::tanh(u);
        T du = k0 * ((T)1 + (T)3 * k1 * x2);
        T dy = (T)0.5 * ((T)1 + t) + (T)0.5 * x * ((T)1 - t * t) * du;
    """,
    use_output=False,
)

softplus_grad = defkernel(
    "kk_softplus_grad",
    "metal::log(metal::exp(x) + (T)1.0)",
    vjp_expr="g * kk_sigmoid(x)",
    use_output=False,
)

mish_grad = defkernel(
    "kk_mish_grad",
    "x * metal::tanh(metal::log(metal::exp(x) + (T)1.0))",
    vjp_expr="g * dy",
    vjp_prelude="""
        T sp = metal::log(metal::exp(x) + (T)1.0);
        T tsp = metal::tanh(sp);
        T s = kk_sigmoid(x);
        T dy = tsp + x * s * ((T)1.0 - tsp * tsp);
    """,
    use_output=False,
)

elu_grad = defkernel(
    "kk_elu_grad",
    "(x > (T)0) ? x : (T)1.0 * (metal::exp(x) - (T)1.0)",
    vjp_expr="(x > (T)0) ? g : g * (T)1.0 * metal::exp(x)",
    use_output=False,
)


__all__ = [
    "exp",
    "log",
    "tanh",
    "sigmoid",
    "relu",
    "relu2",
    "silu",
    "gelu_tanh",
    "softplus",
    "mish",
    "elu",
    "exp_grad",
    "tanh_grad",
    "sigmoid_grad",
    "relu_grad",
    "relu2_grad",
    "silu_grad",
    "gelu_tanh_grad",
    "softplus_grad",
    "mish_grad",
    "elu_grad",
]
