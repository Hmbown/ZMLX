"""Micro-benchmark: fused gather_qmm_swiglu vs naive two-pass approach.

Measures kernel-level latency and throughput without model overhead.

Usage:
    python benchmarks/bench_gather_qmm_swiglu.py
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx
import mlx.nn as nn


def _quantize_experts(n_experts, N, K, bits, group_size, dtype=mx.float16):
    """Create properly quantized expert weight matrices."""
    w_list, s_list, b_list = [], [], []
    for _ in range(n_experts):
        fp = mx.random.normal((N, K)).astype(dtype) * 0.02
        w, s, b = mx.quantize(fp, group_size=group_size, bits=bits)
        w_list.append(w)
        s_list.append(s)
        b_list.append(b)
    return mx.stack(w_list), mx.stack(s_list), mx.stack(b_list)


def bench_one(name, fn, warmup=20, iters=200):
    """Benchmark a single function, return median ms."""
    # Warmup
    for _ in range(warmup):
        out = fn()
        if isinstance(out, (tuple, list)):
            mx.eval(*out)
        else:
            mx.eval(out)

    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, (tuple, list)):
            mx.eval(*out)
        else:
            mx.eval(out)
        if callable(sync):
            sync()
        times.append((time.perf_counter() - t0) * 1e6)  # microseconds

    times.sort()
    median = times[len(times) // 2]
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    print(f"  {name:.<45} {median:8.1f} us  (p10={p10:.1f}, p90={p90:.1f})")
    return median


def bench_shape(
    n_experts: int,
    M: int,
    K: int,
    N: int,
    bits: int = 4,
    group_size: int = 64,
    n_selected: int = 2,
    dtype=mx.float16,
):
    """Benchmark fused vs naive for a specific shape."""
    gate_w, gate_s, gate_b = _quantize_experts(n_experts, N, K, bits, group_size, dtype)
    up_w, up_s, up_b = _quantize_experts(n_experts, N, K, bits, group_size, dtype)
    x = mx.random.normal((1, M, K)).astype(dtype) * 0.1

    lhs_indices = mx.zeros((n_selected,), dtype=mx.uint32)
    rhs_indices = mx.arange(n_selected).astype(mx.uint32)

    # Ensure weights are materialized
    mx.eval(gate_w, gate_s, gate_b, up_w, up_s, up_b, x)

    def naive():
        g = mx.gather_qmm(
            x, gate_w, gate_s, gate_b,
            lhs_indices=lhs_indices, rhs_indices=rhs_indices,
            transpose=True, group_size=group_size, bits=bits,
        )
        u = mx.gather_qmm(
            x, up_w, up_s, up_b,
            lhs_indices=lhs_indices, rhs_indices=rhs_indices,
            transpose=True, group_size=group_size, bits=bits,
        )
        return nn.silu(g) * u

    def fused():
        return mx.gather_qmm_swiglu(
            x, gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            lhs_indices=lhs_indices, rhs_indices=rhs_indices,
            transpose=True, group_size=group_size, bits=bits,
        )

    t_naive = bench_one("naive (2x gather_qmm + SwiGLU)", naive)
    t_fused = bench_one("fused (gather_qmm_swiglu)", fused)
    speedup = t_naive / t_fused if t_fused > 0 else float("inf")
    print(f"  => Speedup: {speedup:.2f}x")
    return t_naive, t_fused, speedup


def main():
    if not hasattr(mx, "gather_qmm_swiglu"):
        print("ERROR: mx.gather_qmm_swiglu not available in this MLX build")
        sys.exit(1)

    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print()

    # -----------------------------------------------------------------------
    # Decode shapes (B=1, M=1) — most latency-sensitive
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("DECODE (M=1) — single-token latency")
    print("=" * 70)

    configs = [
        # (label, n_experts, M, K, N, bits, group_size, n_selected)
        ("Small (K=512, N=512)", 8, 1, 512, 512, 4, 64, 2),
        ("Medium (K=2048, N=1024)", 8, 1, 2048, 1024, 4, 64, 2),
        ("Qwen3-30B-A3B (K=2048, N=2048)", 8, 1, 2048, 2048, 4, 64, 2),
        ("Large (K=4096, N=2048)", 8, 1, 4096, 2048, 4, 64, 2),
    ]

    results = []
    for label, *args in configs:
        print(f"\n{label}:")
        t_n, t_f, s = bench_shape(*args)
        results.append((label, t_n, t_f, s))

    # -----------------------------------------------------------------------
    # Prefill shapes (M > 1) — throughput-sensitive
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PREFILL (M>1) — throughput")
    print("=" * 70)

    prefill_configs = [
        ("M=4, K=2048, N=1024", 8, 4, 2048, 1024, 4, 64, 2),
        ("M=16, K=2048, N=1024", 8, 16, 2048, 1024, 4, 64, 2),
        ("M=64, K=2048, N=1024", 8, 64, 2048, 1024, 4, 64, 2),
    ]

    for label, *args in prefill_configs:
        print(f"\n{label}:")
        t_n, t_f, s = bench_shape(*args)
        results.append((label, t_n, t_f, s))

    # -----------------------------------------------------------------------
    # Quantization variants
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("QUANTIZATION VARIANTS (M=1, K=2048, N=1024)")
    print("=" * 70)

    quant_configs = [
        ("4-bit, gs=32", 8, 1, 2048, 1024, 4, 32, 2),
        ("4-bit, gs=64", 8, 1, 2048, 1024, 4, 64, 2),
        ("4-bit, gs=128", 8, 1, 2048, 1024, 4, 128, 2),
        ("8-bit, gs=64", 8, 1, 2048, 1024, 8, 64, 2),
    ]

    for label, *args in quant_configs:
        print(f"\n{label}:")
        t_n, t_f, s = bench_shape(*args)
        results.append((label, t_n, t_f, s))

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<40} {'Naive':>10} {'Fused':>10} {'Speedup':>10}")
    print("-" * 70)
    for label, t_n, t_f, s in results:
        print(f"{label:<40} {t_n:>9.1f}us {t_f:>9.1f}us {s:>9.2f}x")


if __name__ == "__main__":
    main()
