import mlx.core as mx

from zmlx.kernels import (
    attention,
    linear,
    loss,
    norms,
    reductions,
    rope,
    scan,
    transformer,
)


def test_swiglu_correctness():
    D = 128
    x = mx.random.normal((2, 2 * D)).astype(mx.float32)
    y = transformer.swiglu(x)
    a, b = x[..., :D], x[..., D:]
    y_ref = (a * mx.sigmoid(a)) * b
    assert mx.allclose(y, y_ref, atol=1e-5)

def test_swiglu_grad():
    D = 64
    x = mx.random.normal((1, 2 * D)).astype(mx.float32)
    def loss_fn(z): return transformer.swiglu(z).sum()
    def loss_ref(z):
        a, b = z[..., :D], z[..., D:]
        return ((a * mx.sigmoid(a)) * b).sum()
    gx = mx.grad(loss_fn)(x)
    gx_ref = mx.grad(loss_ref)(x)
    assert mx.allclose(gx, gx_ref, atol=1e-5)

def test_geglu_correctness():
    D = 128
    x = mx.random.normal((2, 2 * D)).astype(mx.float32)
    y = transformer.geglu(x)
    a, b = x[..., :D], x[..., D:]
    def gelu_ref(z): return 0.5 * z * (1 + mx.tanh(0.79788456 * (z + 0.044715 * z**3)))
    y_ref = gelu_ref(a) * b
    assert mx.allclose(y, y_ref, atol=1e-5)

def test_rmsnorm_residual():
    D = 64
    x = mx.random.normal((2, D)).astype(mx.float32)
    res = mx.random.normal((2, D)).astype(mx.float32)
    w = mx.random.normal((D,)).astype(mx.float32)
    y, updated_res = transformer.rmsnorm_residual(x, res, w, eps=1e-6)
    ref_res = x + res
    rms = mx.sqrt(mx.mean(ref_res**2, axis=-1, keepdims=True) + 1e-6)
    y_ref = (ref_res / rms) * w
    assert mx.allclose(y, y_ref, atol=1e-5)
    assert mx.allclose(updated_res, ref_res, atol=1e-5)

def test_rmsnorm_residual_grad():
    D = 64
    eps = 1e-6
    x = mx.random.normal((4, D)).astype(mx.float32)
    res = mx.random.normal((4, D)).astype(mx.float32)
    w = mx.random.normal((D,)).astype(mx.float32)

    # Reference forward using pure MLX ops
    def ref_fwd(x_in, res_in, w_in):
        val = x_in + res_in
        rms = mx.rsqrt(mx.mean(val * val, axis=-1, keepdims=True) + eps)
        y = val * rms * w_in
        updated_res = val
        return y, updated_res

    # Loss that depends on both outputs
    def loss_kernel(x_in, res_in, w_in):
        y, updated_res = transformer.rmsnorm_residual(x_in, res_in, w_in, eps=eps)
        return (y.sum() + updated_res.sum())

    def loss_ref(x_in, res_in, w_in):
        y, updated_res = ref_fwd(x_in, res_in, w_in)
        return (y.sum() + updated_res.sum())

    gx, gres, gw = mx.grad(loss_kernel, argnums=(0, 1, 2))(x, res, w)
    gx_ref, gres_ref, gw_ref = mx.grad(loss_ref, argnums=(0, 1, 2))(x, res, w)
    mx.eval(gx, gres, gw, gx_ref, gres_ref, gw_ref)

    assert mx.allclose(gx, gx_ref, atol=1e-4).item(), "d_x mismatch"
    assert mx.allclose(gres, gres_ref, atol=1e-4).item(), "d_residual mismatch"
    assert mx.allclose(gw, gw_ref, atol=1e-4).item(), "d_weight mismatch"


def test_reductions():
    shape = (4, 1024)
    x = mx.random.normal(shape).astype(mx.float32)
    assert mx.allclose(reductions.sum_lastdim(x), x.sum(axis=-1), atol=1e-4)
    assert mx.allclose(reductions.mean_lastdim(x), x.mean(axis=-1), atol=1e-4)
    assert mx.allclose(reductions.max_lastdim(x), x.max(axis=-1), atol=1e-4)

def test_dropout():
    x = mx.ones((10000,)).astype(mx.float32)
    p = 0.5
    y = transformer.dropout(x, p, seed=42)
    zeros = (y == 0).sum().item()
    assert 4500 < zeros < 5500
    expected = 1.0 / (1.0 - p)
    is_zero = mx.abs(y) < 1e-5
    is_expected = mx.abs(y - expected) < 1e-5
    assert mx.all(mx.logical_or(is_zero, is_expected))

def test_attention_kernels():
    x = mx.random.normal((2, 128)).astype(mx.float32)
    y = attention.logsumexp_lastdim(x)
    assert mx.allclose(y, mx.logsumexp(x, axis=-1), atol=1e-4)
    
    x = mx.array([[1.0, 2.0, 10.0], [1.0, 2.0, 3.0]]).astype(mx.float32)
    mask = mx.array([[True, True, False], [True, True, True]])
    y = attention.masked_softmax(x, mask)
    row0_ref = mx.softmax(x[0, :2])
    assert mx.allclose(y[0, :2], row0_ref, atol=1e-5)
    assert y[0, 2] == 0

def test_rmsnorm_grad():
    D = 64
    x = mx.random.normal((2, D)).astype(mx.float32)
    w = mx.random.normal((D,)).astype(mx.float32)
    def loss_fn(z): return norms.rmsnorm(z, w, eps=1e-6).sum()
    gx_ref = mx.grad(loss_fn)(x)
    y = norms.rmsnorm(x, w, eps=1e-6)
    gx = norms.rmsnorm_grad(x, w, mx.ones_like(y), eps=1e-6)
    assert mx.allclose(gx, gx_ref, atol=1e-5)

def test_rope_interleaved():
    S, D = 16, 32
    x = mx.random.normal((1, S, D)).astype(mx.float32)
    cos = mx.random.normal((S, D // 2)).astype(mx.float32)
    sin = mx.random.normal((S, D // 2)).astype(mx.float32)
    y = rope.apply_rope_interleaved(x, cos, sin)
    x_v = x.reshape(1, S, D // 2, 2)
    y1 = x_v[..., 0] * cos - x_v[..., 1] * sin
    y2 = x_v[..., 0] * sin + x_v[..., 1] * cos
    y_ref = mx.stack([y1, y2], axis=-1).reshape(1, S, D)
    assert mx.allclose(y, y_ref, atol=1e-5)

def test_var_lastdim():
    x = mx.random.normal((4, 1024)).astype(mx.float32)
    assert mx.allclose(reductions.var_lastdim(x), mx.var(x, axis=-1), atol=1e-4)

def test_argmax_cumsum():
    x = mx.array([[1.0, 5.0, 2.0], [10.0, 2.0, 3.0]]).astype(mx.float32)
    assert mx.array_equal(reductions.argmax_lastdim(x), mx.array([1, 0], dtype=mx.uint32))
    y = scan.cumsum_lastdim(x)
    assert mx.allclose(y, mx.cumsum(x, axis=-1))

def test_softmax_ce():
    logits = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).astype(mx.float32)
    targets = mx.array([2, 0], dtype=mx.uint32)
    y = loss.softmax_cross_entropy(logits, targets)
    ref = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, targets[:, None], axis=-1).squeeze()
    assert mx.allclose(y, ref, atol=1e-5)

def test_softmax_ce_grad():
    logits = mx.random.normal((4, 16)).astype(mx.float32)
    targets = mx.array([0, 1, 2, 3], dtype=mx.uint32)

    def loss_fn(lgts):
        return loss.softmax_cross_entropy(lgts, targets).mean()

    def ref_loss(lgts):
        t = targets.astype(mx.int32)
        logsumexp = mx.logsumexp(lgts, axis=-1)
        target_logits = mx.take_along_axis(lgts, t[:, None], axis=-1).squeeze(-1)
        return mx.mean(logsumexp - target_logits)

    g = mx.grad(loss_fn)(logits)
    g_ref = mx.grad(ref_loss)(logits)
    mx.eval(g, g_ref)
    assert mx.allclose(g, g_ref, atol=1e-4).item()

def test_topk():
    x = mx.array([[1., 5., 2., 10., 3.], [9., 1., 4., 3., 8.]]).astype(mx.float32)
    v, i = reductions.topk_lastdim(x, 3)
    assert mx.array_equal(i[0], mx.array([3, 1, 4], dtype=mx.uint32))
    assert mx.array_equal(i[1], mx.array([0, 4, 2], dtype=mx.uint32))

def test_layernorm_residual():
    D = 64
    x = mx.random.normal((2, D)).astype(mx.float32)
    res = mx.random.normal((2, D)).astype(mx.float32)
    g = mx.random.normal((D,)).astype(mx.float32)
    b = mx.random.normal((D,)).astype(mx.float32)
    y, ur = transformer.layernorm_residual(x, res, g, b)
    ref_res = x + res
    mean = mx.mean(ref_res, axis=-1, keepdims=True)
    var = mx.var(ref_res, axis=-1, keepdims=True)
    y_ref = (ref_res - mean) * mx.rsqrt(var + 1e-5) * g + b
    assert mx.allclose(y, y_ref, atol=1e-5)
    assert mx.allclose(ur, ref_res, atol=1e-5)

def test_linear_bias_silu():
    M, K, N = 2, 16, 8
    x = mx.random.normal((M, K)).astype(mx.float32)
    w = mx.random.normal((N, K)).astype(mx.float32)
    b = mx.random.normal((N,)).astype(mx.float32)
    y = linear.fused_linear_bias_silu(x, w, b)
    assert y.shape == (M, N)
