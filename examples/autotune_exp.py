import mlx.core as mx

from zmlx import autotune_threadgroup, metal, msl


def main():
    k = metal.kernel(
        name="kk_exp_autotune",
        input_names=["inp"],
        output_names=["out"],
        source='''
            uint elem = thread_position_in_grid.x;
            T x = inp[elem];
            out[elem] = metal::exp(x);
        ''',
        header=msl.DEFAULT_HEADER,
    )

    x = mx.random.normal((1_000_000,)).astype(mx.float16)
    grid = (x.size, 1, 1)

    result = autotune_threadgroup(
        k,
        inputs=[x],
        template=[("T", mx.float32)],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=grid,
        candidates=[(32, 1, 1), (64, 1, 1), (128, 1, 1), (256, 1, 1), (512, 1, 1)],
        warmup=5,
        iters=30,
    )

    print("best threadgroup:", result.best_threadgroup)
    # Print a small summary
    for tg, ms in sorted(result.timings_ms.items(), key=lambda kv: kv[1]):
        print(f"  {tg}: {ms:.4f} ms")

if __name__ == "__main__":
    main()
