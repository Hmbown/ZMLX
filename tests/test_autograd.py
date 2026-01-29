import mlx.core as mx

from zmlx import autograd


def test_autograd_exp_vjp():
    exp_metal = autograd.unary_from_expr(
        name="test_exp_vjp",
        fwd_expr="metal::exp(x)",
        vjp_expr="g * y",
        compute_dtype=mx.float32,
        use_output=True,
    )

    x = mx.random.normal((2048,)).astype(mx.float32)

    def loss(z):
        return exp_metal(z).sum()

    gx = mx.grad(loss)(x)
    gx_ref = mx.exp(x)
    mx.eval(gx, gx_ref)
    assert mx.allclose(gx, gx_ref, rtol=1e-4, atol=1e-4).item()

def test_autograd_silu_vjp():
    silu_metal = autograd.unary_from_expr(
        name="test_silu_vjp",
        fwd_expr="x * (T(1.0) / (T(1.0) + metal::exp(-x)))",
        vjp_prelude="T sig = T(1.0) / (T(1.0) + metal::exp(-x));",
        vjp_expr="g * (sig + x * sig * (T(1.0) - sig))",
        compute_dtype=mx.float32,
        use_output=False,
    )

    x = mx.random.normal((4096,)).astype(mx.float32)

    def loss(z):
        return silu_metal(z).mean()

    gx = mx.grad(loss)(x)

    def silu_ref(z):
        return z * mx.sigmoid(z)

    gx_ref = mx.grad(lambda z: silu_ref(z).mean())(x)

    mx.eval(gx, gx_ref)
    assert mx.allclose(gx, gx_ref, rtol=1e-4, atol=1e-4).item()
