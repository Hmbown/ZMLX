import mlx.core as mx

from zmlx import autograd, msl


def main():
    exp_metal = autograd.unary_from_expr(
        name="kk_exp",
        fwd_expr="metal::exp(x)",
        vjp_expr="g * y",          # derivative of exp is exp(x), reuse forward output y
        compute_dtype=mx.float32,
        use_output=True,
        header=msl.DEFAULT_HEADER,
    )

    x = mx.random.normal((2048,)).astype(mx.float32)

    def loss(z):
        return exp_metal(z).sum()

    gx = mx.grad(loss)(x)
    mx.eval(gx)

    gx_ref = mx.exp(x)  # d/dx sum(exp(x)) = exp(x)
    mx.eval(gx_ref)

    ok = mx.allclose(gx, gx_ref, rtol=1e-4, atol=1e-4).item()
    print("autograd_exp grad allclose:", bool(ok))

if __name__ == "__main__":
    main()
