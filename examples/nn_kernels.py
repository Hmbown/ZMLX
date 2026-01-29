import mlx.core as mx

from zmlx.kernels import activations, norms, softmax


def main():
    x = mx.random.normal((4, 1024)).astype(mx.float16)

    y = softmax.softmax_lastdim(x)
    mx.eval(y)
    print("softmax ok:", y.shape, y.dtype)

    w = mx.ones((1024,), dtype=mx.float16)
    z = norms.rmsnorm(x, w, eps=1e-6)
    mx.eval(z)
    print("rmsnorm ok:", z.shape, z.dtype)

    # gradient-enabled activation
    exp_trainable = activations.exp_grad()
    g = mx.grad(lambda t: exp_trainable(t).sum())(x.astype(mx.float32))
    mx.eval(g)
    print("exp_grad ok:", g.shape, g.dtype)


if __name__ == "__main__":
    main()
