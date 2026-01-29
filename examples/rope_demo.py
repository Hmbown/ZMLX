import mlx.core as mx

from zmlx.kernels import rope


def main():
    S, D = 128, 64
    x = mx.random.normal((2, S, D)).astype(mx.float16)
    cos = mx.random.normal((S, D // 2)).astype(mx.float16)
    sin = mx.random.normal((S, D // 2)).astype(mx.float16)

    y = rope.apply_rope(x, cos, sin)
    mx.eval(y)
    print("rope ok:", y.shape, y.dtype)


if __name__ == "__main__":
    main()
