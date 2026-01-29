import mlx.core as mx

from zmlx import elementwise


def test_unary_exp_matches_mx():
    exp_fast = elementwise.unary(
        name="test_exp",
        expr="metal::exp(x)",
        compute_dtype=mx.float32,
    )
    x = mx.random.normal((1024,)).astype(mx.float16)
    y = exp_fast(x)
    y_ref = mx.exp(x)
    mx.eval(y, y_ref)
    assert mx.allclose(y, y_ref, rtol=1e-3, atol=1e-3).item()

def test_binary_add_matches_mx():
    add_fast = elementwise.binary(
        name="test_add",
        expr="a + b",
        compute_dtype=mx.float32,
    )
    a = mx.random.normal((1024,)).astype(mx.float16)
    b = mx.random.normal((1024,)).astype(mx.float16)
    y = add_fast(a, b)
    y_ref = a + b
    mx.eval(y, y_ref)
    assert mx.allclose(y, y_ref, rtol=1e-3, atol=1e-3).item()
