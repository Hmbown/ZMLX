import mlx.core as mx

from zmlx import autograd, msl


def main():
    silu_metal = autograd.unary_from_expr(
        name="kk_silu",
        fwd_expr="x * (T(1.0) / (T(1.0) + metal::exp(-x)))",
        vjp_prelude="T sig = T(1.0) / (T(1.0) + metal::exp(-x));",
        # d/dx [x*sigmoid(x)] = sig + x*sig*(1-sig)
        vjp_expr="g * (sig + x * sig * (T(1.0) - sig))",
        compute_dtype=mx.float32,
        use_output=False,
        header=msl.DEFAULT_HEADER,
    )

    x = mx.random.normal((4096,)).astype(mx.float32)

    def loss(z):
        return silu_metal(z).mean()

    gx = mx.grad(loss)(x)
    mx.eval(gx)

    # Reference gradient using MLX ops
    def silu_ref(z):
        return z * mx.sigmoid(z)

    gx_ref = mx.grad(lambda z: silu_ref(z).mean())(x)
    mx.eval(gx_ref)

    ok = mx.allclose(gx, gx_ref, rtol=1e-4, atol=1e-4).item()
    print("fused_silu grad allclose:", bool(ok))

if __name__ == "__main__":
    main()
