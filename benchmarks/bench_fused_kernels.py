"""Benchmark suite for fused Metal kernels: add_rms_norm and gather_qmm_combine.

Usage:
    python benchmarks/bench_fused_kernels.py

NOTE: MoE benchmarks use E=8, D=2048, float16 by default. Larger configs
(E=64, D=4096, float32) can monopolize the GPU long enough to trigger the
macOS watchdog timer and cause a kernel panic — Apple Silicon has no GPU
preemption, so long-running Metal command buffers starve the window server.
"""

from __future__ import annotations

import time
from functools import partial

import mlx.core as mx


def _warmup_and_time(fn, *, warmup: int = 5, iters: int = 20) -> float:
    """Run warmup iterations then return median time in microseconds."""
    for _ in range(warmup):
        result = fn()
        mx.eval(result)
        mx.synchronize()  # yield GPU between warmup iters to avoid watchdog

    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        result = fn()
        mx.eval(result)
        mx.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)  # ns -> us

    times.sort()
    return times[len(times) // 2]


def _run_decomposed(x, res, w, eps):
    h = x + res
    return mx.fast.rms_norm(h, w, eps), h


def _run_existing(rmsnorm_residual_fn, x, res, w, eps):
    return rmsnorm_residual_fn(x, res, w, eps=eps)


def _run_new(add_rms_norm_fn, x, res, w, eps):
    return add_rms_norm_fn(x, res, w, eps=eps)


def _run_residual_rmsnorm(residual_rmsnorm_fn, x, res, w):
    return residual_rmsnorm_fn(x, res, w, eps=1e-5)


def _run_gqc(gather_qmm_combine_fn, act, weights, gate, indices, stream):
    return gather_qmm_combine_fn(act, weights, gate, indices, streaming=stream)


def _run_decomposed_moe(moe_combine_fn, act, weights, gate, indices, K):
    proj_all = []
    for k_idx in range(K):
        a_k = act[:, k_idx, :]
        w_k = weights[indices[:, k_idx]]
        proj_k = mx.matmul(mx.expand_dims(a_k, axis=1), w_k).squeeze(axis=1)
        proj_all.append(proj_k)
    proj_stacked = mx.stack(proj_all, axis=1)
    return moe_combine_fn(proj_stacked, gate)


def bench_add_rms_norm():
    """Compare add_rms_norm variants."""
    from zmlx.kernels.norms import add_rms_norm, residual_rmsnorm
    from zmlx.kernels.transformer import rmsnorm_residual

    print("=" * 72)
    print("  add_rms_norm benchmarks")
    print("=" * 72)
    header = (
        f"{'Shape':>20s}  {'Decomposed':>12s}  "
        f"{'Existing':>12s}  {'New':>12s}  {'Speedup':>8s}"
    )
    print(header)
    print("-" * len(header))

    for B in [1, 4, 16, 64]:
        for D in [2048, 4096]:
            shape_str = f"({B}, {D})"
            x = mx.random.normal((B, D))
            res = mx.random.normal((B, D))
            w = mx.ones((D,))
            eps = 1e-5

            fn_dec = partial(_run_decomposed, x, res, w, eps)
            fn_exist = partial(_run_existing, rmsnorm_residual, x, res, w, eps)
            fn_new = partial(_run_new, add_rms_norm, x, res, w, eps)

            t_decomposed = _warmup_and_time(fn_dec)
            t_existing = _warmup_and_time(fn_exist)
            t_new = _warmup_and_time(fn_new)

            speedup = t_decomposed / t_new if t_new > 0 else float("inf")
            print(
                f"{shape_str:>20s}  {t_decomposed:>10.1f}us  {t_existing:>10.1f}us  "
                f"{t_new:>10.1f}us  {speedup:>7.2f}x"
            )

    print()
    print("Single-output comparison (residual_rmsnorm from norms.py):")
    for B in [1, 16]:
        for D in [2048, 4096]:
            shape_str = f"({B}, {D})"
            x = mx.random.normal((B, D))
            res = mx.random.normal((B, D))
            w = mx.ones((D,))

            fn = partial(_run_residual_rmsnorm, residual_rmsnorm, x, res, w)
            t = _warmup_and_time(fn)
            print(f"  {shape_str:>20s}  residual_rmsnorm: {t:.1f}us")


def bench_gather_qmm_combine():
    """Compare gather_qmm_combine streaming vs decomposed.

    Uses E=8, D=2048, float16 — realistic for 8B-class MoE inference.
    Larger configs (E=64, D=4096, float32) can trigger macOS watchdog panics
    because Apple Silicon has no GPU preemption.
    """
    from zmlx.kernels.moe import gather_qmm_combine, moe_combine

    print()
    print("=" * 72)
    print("  gather_qmm_combine benchmarks (E=8, D=2048, float16)")
    print("=" * 72)
    header = (
        f"{'B':>4s}  {'K':>3s}  {'E':>3s}  {'D_in':>5s}  {'D_out':>5s}  "
        f"{'Decomposed':>12s}  {'Streaming':>12s}  {'NonStream':>12s}  {'Speedup':>8s}"
    )
    print(header)
    print("-" * len(header))

    E = 8
    D_in = 2048
    D_out = 2048
    K = 4

    # Allocate weights once (E=8, 2048x2048, fp16 = 64 MB)
    mx.random.seed(42)
    weights = (mx.random.normal((E, D_in, D_out)) * 0.01).astype(mx.float16)
    mx.eval(weights)

    for B in [1, 2, 4, 8, 16, 32]:
        act = mx.random.normal((B, K, D_in)).astype(mx.float16)
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1).astype(mx.float16)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E
        mx.eval(act, gate, indices)

        fn_dec = partial(_run_decomposed_moe, moe_combine, act, weights, gate, indices, K)
        fn_stream = partial(_run_gqc, gather_qmm_combine, act, weights, gate, indices, True)
        fn_batch = partial(_run_gqc, gather_qmm_combine, act, weights, gate, indices, False)

        t_decomposed = _warmup_and_time(fn_dec)
        t_streaming = _warmup_and_time(fn_stream)
        t_nonstreaming = _warmup_and_time(fn_batch)

        speedup = t_decomposed / t_streaming if t_streaming > 0 else float("inf")
        print(
            f"{B:>4d}  {K:>3d}  {E:>3d}  {D_in:>5d}  {D_out:>5d}  "
            f"{t_decomposed:>10.1f}us  {t_streaming:>10.1f}us  "
            f"{t_nonstreaming:>10.1f}us  {speedup:>7.2f}x"
        )
        mx.synchronize()  # yield GPU between configs


def bench_crossover():
    """Find streaming vs non-streaming crossover point."""
    from zmlx.kernels.moe import gather_qmm_combine

    print()
    print("=" * 72)
    print("  Streaming crossover analysis (K=4, E=8, D=2048, float16)")
    print("=" * 72)
    header = f"{'B':>6s}  {'Streaming':>12s}  {'NonStream':>12s}  {'Winner':>10s}"
    print(header)
    print("-" * len(header))

    E, K, D_in, D_out = 8, 4, 2048, 2048

    mx.random.seed(42)
    weights = (mx.random.normal((E, D_in, D_out)) * 0.01).astype(mx.float16)
    mx.eval(weights)

    for B in [1, 2, 4, 8, 16, 32, 64]:
        act = mx.random.normal((B, K, D_in)).astype(mx.float16)
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1).astype(mx.float16)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E
        mx.eval(act, gate, indices)

        fn_s = partial(_run_gqc, gather_qmm_combine, act, weights, gate, indices, True)
        fn_ns = partial(_run_gqc, gather_qmm_combine, act, weights, gate, indices, False)

        t_s = _warmup_and_time(fn_s)
        t_ns = _warmup_and_time(fn_ns)
        winner = "streaming" if t_s < t_ns else "batch"
        print(f"{B:>6d}  {t_s:>10.1f}us  {t_ns:>10.1f}us  {winner:>10s}")
        mx.synchronize()


if __name__ == "__main__":
    bench_add_rms_norm()
    bench_gather_qmm_combine()
    bench_crossover()
