import mlx.core as mx

from zmlx.kernels import activations, fused, norms, rope, softmax


def test_softmax_matches_reference():
    x = mx.random.normal((4, 128)).astype(mx.float16)
    y = softmax.softmax_lastdim(x, threadgroup=128)

    # reference
    m = mx.max(x, axis=-1, keepdims=True)
    ex = mx.exp(x - m)
    y_ref = ex / mx.sum(ex, axis=-1, keepdims=True)

    mx.eval(y, y_ref)
    assert mx.allclose(y, y_ref, rtol=2e-3, atol=2e-3).item()


def test_rmsnorm_matches_reference():
    eps = 1e-6
    x = mx.random.normal((2, 64)).astype(mx.float16)
    w = mx.random.normal((64,)).astype(mx.float16)

    y = norms.rmsnorm(x, w, eps=eps, threadgroup=64)

    # ref: x * rsqrt(mean(x^2)+eps) * w
    x2 = x * x
    mean2 = mx.mean(x2, axis=-1, keepdims=True)
    inv = mx.rsqrt(mean2 + eps)
    y_ref = x * inv * w  # broadcast w over the first dim

    mx.eval(y, y_ref)
    assert mx.allclose(y, y_ref, rtol=3e-3, atol=3e-3).item()


def test_layernorm_matches_reference():
    eps = 1e-5
    x = mx.random.normal((2, 64)).astype(mx.float16)
    gamma = mx.random.normal((64,)).astype(mx.float16)
    beta = mx.random.normal((64,)).astype(mx.float16)

    y = norms.layernorm(x, gamma, beta, eps=eps, threadgroup=64)

    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    inv = mx.rsqrt(var + eps)
    y_ref = (x - mean) * inv * gamma + beta

    mx.eval(y, y_ref)
    assert mx.allclose(y, y_ref, rtol=3e-3, atol=3e-3).item()


def test_rope_matches_reference():
    S, D = 32, 64
    x = mx.random.normal((2, S, D)).astype(mx.float16)
    cos = mx.random.normal((S, D // 2)).astype(mx.float16)
    sin = mx.random.normal((S, D // 2)).astype(mx.float16)

    y = rope.apply_rope(x, cos, sin)

    x0 = x[..., : D // 2]
    x1 = x[..., D // 2 :]
    cos_b = mx.reshape(cos, (1, S, D // 2))
    sin_b = mx.reshape(sin, (1, S, D // 2))
    y0 = x0 * cos_b - x1 * sin_b
    y1 = x0 * sin_b + x1 * cos_b
    y_ref = mx.concatenate([y0, y1], axis=-1)

    mx.eval(y, y_ref)
    assert mx.allclose(y, y_ref, rtol=3e-3, atol=3e-3).item()


def test_fused_silu_mul_grad_matches_reference():
    op = fused.silu_mul_grad()

    a = mx.random.normal((1024,)).astype(mx.float32)
    b = mx.random.normal((1024,)).astype(mx.float32)

    def loss(u, v):
        return mx.mean(op(u, v))

    ga, gb = mx.grad(loss, argnums=(0, 1))(a, b)

    def ref(u, v):
        return mx.mean((u * mx.sigmoid(u)) * v)

    ga_ref, gb_ref = mx.grad(ref, argnums=(0, 1))(a, b)

    mx.eval(ga, gb, ga_ref, gb_ref)
    assert mx.allclose(ga, ga_ref, rtol=2e-4, atol=2e-4).item()
    assert mx.allclose(gb, gb_ref, rtol=2e-4, atol=2e-4).item()


def test_activation_grad_variant_matches_reference():
    exp_op = activations.exp_grad()

    x = mx.random.normal((2048,)).astype(mx.float32)

    def loss(z):
        return mx.mean(exp_op(z))

    gx = mx.grad(loss)(x)

    gx_ref = mx.grad(lambda z: mx.mean(mx.exp(z)))(x)

    mx.eval(gx, gx_ref)
    assert mx.allclose(gx, gx_ref, rtol=1e-4, atol=1e-4).item()
