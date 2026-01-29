import mlx.core as mx

from zmlx import elementwise, msl


def main():
    exp_fast = elementwise.unary(
        name="kk_exp",
        expr="metal::exp(x)",
        compute_dtype=mx.float32,
        header=msl.DEFAULT_HEADER,
    )

    x = mx.random.normal((1024,)).astype(mx.float16)
    y = exp_fast(x)
    mx.eval(y)

    y_ref = mx.exp(x)
    mx.eval(y_ref)

    ok = mx.allclose(y, y_ref, rtol=1e-3, atol=1e-3).item()
    print("exp_elementwise allclose:", bool(ok))

if __name__ == "__main__":
    main()
