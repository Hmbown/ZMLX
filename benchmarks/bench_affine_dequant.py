#!/usr/bin/env python3
"""Microbench affine-packed dequant + activation (baseline vs fused).

Uses mx.quantize (affine) to generate packed weights, then compares:
- mx.dequantize + activation
- zmlx.kernels.quant fused dequant + activation
"""

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


def _bench_case(B: int, D: int, *, bits: int, group_size: int) -> None:
    w = mx.random.normal((B, D)).astype(mx.float32)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")

    def baseline_silu():
        deq = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits, mode="affine")
        return deq * mx.sigmoid(deq)

    def fused_silu():
        return quant.dequantize_affine_packed_silu(q, scales, biases, bits=bits, group_size=group_size)

    def baseline_gelu():
        deq = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits, mode="affine")
        k0 = 0.7978845608028654
        k1 = 0.044715
        return 0.5 * deq * (1.0 + mx.tanh(k0 * (deq + k1 * deq * deq * deq)))

    def fused_gelu():
        return quant.dequantize_affine_packed_gelu(q, scales, biases, bits=bits, group_size=group_size)

    t_base_silu = _warmup_and_time(baseline_silu)
    t_fused_silu = _warmup_and_time(fused_silu)
    t_base_gelu = _warmup_and_time(baseline_gelu)
    t_fused_gelu = _warmup_and_time(fused_gelu)

    silu_speedup = t_base_silu / t_fused_silu if t_fused_silu else 0.0
    gelu_speedup = t_base_gelu / t_fused_gelu if t_fused_gelu else 0.0

    print(
        f"B={B:>2d} D={D:<5d} bits={bits} group={group_size}  "
        f"SiLU {t_base_silu:>8.1f}us -> {t_fused_silu:>8.1f}us ({silu_speedup:>5.2f}x)  "
        f"GELU {t_base_gelu:>8.1f}us -> {t_fused_gelu:>8.1f}us ({gelu_speedup:>5.2f}x)"
    )


def main() -> None:
    mx.random.seed(0)
    print("Affine-packed dequant + activation microbench")
    print("-" * 88)
    for bits in (4, 8):
        for B in (1, 16):
            _bench_case(B, 4096, bits=bits, group_size=64)


if __name__ == "__main__":
    main()
