"""Verify DSL activations produce identical output to legacy activations.

For every activation in the catalog, this test:
1. Builds both the DSL and legacy kernels
2. Runs them on the same random input
3. Asserts numerical identity (atol=1e-6)
"""

from __future__ import annotations

import pytest

from zmlx._compat import is_metal_available

pytestmark = pytest.mark.skipif(
    not is_metal_available(),
    reason="Metal GPU not available",
)


# ---------------------------------------------------------------------------
# Forward-only activations
# ---------------------------------------------------------------------------

_FORWARD_PAIRS = [
    ("exp", {}),
    ("log", {}),
    ("tanh", {}),
    ("sigmoid", {}),
    ("relu", {}),
    ("relu2", {}),
    ("silu", {}),
    ("gelu_tanh", {}),
    ("softplus", {}),
    ("mish", {}),
]


@pytest.mark.parametrize("name,kwargs", _FORWARD_PAIRS, ids=[p[0] for p in _FORWARD_PAIRS])
def test_forward_identity(name, kwargs):
    """DSL forward kernel matches legacy kernel numerically."""
    import mlx.core as mx

    from zmlx.kernels import activations as legacy
    from zmlx.kernels import activations_dsl as dsl

    legacy_builder = getattr(legacy, name)
    dsl_builder = getattr(dsl, name)

    legacy_k = legacy_builder(**kwargs)
    dsl_k = dsl_builder(**kwargs)

    # Use positive values for log to avoid NaN
    if name == "log":
        x = mx.abs(mx.random.normal((256,))) + 0.01
    else:
        x = mx.random.normal((256,))
    mx.eval(x)

    legacy_out = legacy_k(x)
    dsl_out = dsl_k(x)
    mx.eval(legacy_out, dsl_out)

    assert mx.allclose(legacy_out, dsl_out, atol=1e-6).item(), (
        f"{name}: max diff = {mx.max(mx.abs(legacy_out - dsl_out)).item()}"
    )


# ---------------------------------------------------------------------------
# Gradient-enabled activations
# ---------------------------------------------------------------------------

_GRAD_PAIRS = [
    ("exp_grad", {}),
    ("tanh_grad", {}),
    ("sigmoid_grad", {}),
    ("relu_grad", {}),
    ("relu2_grad", {}),
    ("silu_grad", {}),
    ("gelu_tanh_grad", {}),
    ("softplus_grad", {}),
    ("mish_grad", {}),
]


@pytest.mark.parametrize("name,kwargs", _GRAD_PAIRS, ids=[p[0] for p in _GRAD_PAIRS])
def test_grad_forward_identity(name, kwargs):
    """DSL grad kernel forward pass matches legacy kernel numerically."""
    import mlx.core as mx

    from zmlx.kernels import activations as legacy
    from zmlx.kernels import activations_dsl as dsl

    legacy_builder = getattr(legacy, name)
    dsl_builder = getattr(dsl, name)

    legacy_k = legacy_builder(**kwargs)
    dsl_k = dsl_builder(**kwargs)

    x = mx.random.normal((256,))
    mx.eval(x)

    legacy_out = legacy_k(x)
    dsl_out = dsl_k(x)
    mx.eval(legacy_out, dsl_out)

    assert mx.allclose(legacy_out, dsl_out, atol=1e-6).item(), (
        f"{name}: max diff = {mx.max(mx.abs(legacy_out - dsl_out)).item()}"
    )


@pytest.mark.parametrize("name,kwargs", _GRAD_PAIRS, ids=[p[0] for p in _GRAD_PAIRS])
def test_grad_backward_identity(name, kwargs):
    """DSL grad kernel backward pass matches legacy kernel numerically."""
    import mlx.core as mx

    from zmlx.kernels import activations as legacy
    from zmlx.kernels import activations_dsl as dsl

    legacy_builder = getattr(legacy, name)
    dsl_builder = getattr(dsl, name)

    legacy_k = legacy_builder(**kwargs)
    dsl_k = dsl_builder(**kwargs)

    x = mx.random.normal((128,))
    mx.eval(x)

    # Compute gradients
    legacy_grad_fn = mx.grad(lambda inp: mx.sum(legacy_k(inp)))
    dsl_grad_fn = mx.grad(lambda inp: mx.sum(dsl_k(inp)))

    legacy_grad = legacy_grad_fn(x)
    dsl_grad = dsl_grad_fn(x)
    mx.eval(legacy_grad, dsl_grad)

    assert mx.allclose(legacy_grad, dsl_grad, atol=1e-5).item(), (
        f"{name} grad: max diff = {mx.max(mx.abs(legacy_grad - dsl_grad)).item()}"
    )


# ---------------------------------------------------------------------------
# ELU has an extra parameter (alpha)
# ---------------------------------------------------------------------------

def test_elu_forward_identity():
    """DSL elu matches legacy elu (default alpha=1.0)."""
    import mlx.core as mx

    from zmlx.kernels import activations_dsl as dsl
    from zmlx.kernels.activations import elu as legacy_elu

    legacy_k = legacy_elu()
    dsl_k = dsl.elu()

    x = mx.random.normal((256,))
    mx.eval(x)

    legacy_out = legacy_k(x)
    dsl_out = dsl_k(x)
    mx.eval(legacy_out, dsl_out)

    assert mx.allclose(legacy_out, dsl_out, atol=1e-6).item()


def test_elu_grad_identity():
    """DSL elu_grad matches legacy elu_grad (default alpha=1.0)."""
    import mlx.core as mx

    from zmlx.kernels import activations_dsl as dsl
    from zmlx.kernels.activations import elu_grad as legacy_elu_grad

    legacy_k = legacy_elu_grad()
    dsl_k = dsl.elu_grad()

    x = mx.random.normal((128,))
    mx.eval(x)

    legacy_grad_fn = mx.grad(lambda inp: mx.sum(legacy_k(inp)))
    dsl_grad_fn = mx.grad(lambda inp: mx.sum(dsl_k(inp)))

    legacy_grad = legacy_grad_fn(x)
    dsl_grad = dsl_grad_fn(x)
    mx.eval(legacy_grad, dsl_grad)

    assert mx.allclose(legacy_grad, dsl_grad, atol=1e-5).item()
