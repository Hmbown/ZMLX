#!/usr/bin/env python3
"""Progressive affine dequant microbench (hi2/lo2 refinement)."""

from __future__ import annotations

import time

import mlx.core as mx

from zmlx.kernels import quant


def _warmup_and_time(fn, *, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
        mx.synchronize()

    times = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        out = fn()
        mx.eval(out)
        mx.synchronize()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1e3)
    times.sort()
    return times[len(times) // 2]


def _bench_case(B: int, D: int, *, group_size: int) -> None:
    w = mx.random.normal((B, D)).astype(mx.float32)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=4, mode="affine")

    def baseline():
        return mx.dequantize(q, scales, biases, group_size=group_size, bits=4, mode="affine")

    def hi2():
        return quant.dequantize_affine_packed_hi2(q, scales, biases, group_size=group_size)

    def refine():
        return quant.dequantize_affine_packed_lo2_delta(q, scales, biases, group_size=group_size)

    def full_progressive():
        return hi2() + refine()

    t_base = _warmup_and_time(baseline)
    t_hi2 = _warmup_and_time(hi2)
    t_refine = _warmup_and_time(refine)
    t_prog = _warmup_and_time(full_progressive)

    full = baseline()
    approx = hi2()
    mx.eval(full, approx)
    rel_err = float(mx.mean(mx.abs(full - approx)) / (mx.mean(mx.abs(full)) + 1e-6))

    print(
        f"B={B:>2d} D={D:<5d}  base {t_base:>8.1f}us  hi2 {t_hi2:>8.1f}us  "
        f"lo2 {t_refine:>8.1f}us  prog {t_prog:>8.1f}us  rel_err {rel_err:>7.4f}"
    )


def main() -> None:
    mx.random.seed(0)
    print("Progressive affine dequant (hi2 + lo2)")
    print("-" * 88)
    for B in (1, 16):
        _bench_case(B, 4096, group_size=64)


if __name__ == "__main__":
    main()
