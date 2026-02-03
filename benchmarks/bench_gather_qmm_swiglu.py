"""Micro-benchmark: fused gather_qmm_swiglu vs naive two-pass approach.

Measures kernel-level latency and throughput without model overhead.

Usage:
    python benchmarks/bench_gather_qmm_swiglu.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

try:
    from zmlx.device import detect_device
except Exception:  # pragma: no cover - optional logging only
    detect_device = None


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


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _device_meta() -> dict[str, object]:
    if detect_device is None:
        return {}
    device = detect_device()
    return {
        "family": device.family,
        "variant": device.variant,
        "gpu_cores": device.gpu_cores,
        "memory_gb": device.memory_gb,
        "has_ray_tracing": device.has_ray_tracing,
    }


def _append_json(path: Path, records: list[dict]) -> None:
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    if not isinstance(data, list):
        data = []
    data.extend(records)
    path.write_text(json.dumps(data, indent=2))


def _bench_once(fn, *, warmup: int, iters: int) -> float:
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
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    return times[len(times) // 2]


def _multi_run(fn, *, outer_runs: int, warmup: int, iters: int) -> tuple[float, list[float]]:
    medians = []
    for _ in range(outer_runs):
        medians.append(_bench_once(fn, warmup=warmup, iters=iters))
    medians.sort()
    return medians[len(medians) // 2], medians


def bench_shape(
    n_experts: int,
    M: int,
    K: int,
    N: int,
    bits: int = 4,
    group_size: int = 64,
    n_selected: int = 2,
    dtype=mx.float16,
    outer_runs: int = 3,
    warmup: int = 20,
    iters: int = 200,
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

    t_naive, naive_runs = _multi_run(naive, outer_runs=outer_runs, warmup=warmup, iters=iters)
    t_fused, fused_runs = _multi_run(fused, outer_runs=outer_runs, warmup=warmup, iters=iters)
    speedup = t_naive / t_fused if t_fused > 0 else float("inf")
    print(
        f"  naive  {t_naive:8.1f} us (median of {outer_runs}, min={min(naive_runs):.1f}, max={max(naive_runs):.1f})"
    )
    print(
        f"  fused  {t_fused:8.1f} us (median of {outer_runs}, min={min(fused_runs):.1f}, max={max(fused_runs):.1f})"
    )
    print(f"  => Speedup: {speedup:.2f}x")
    return t_naive, t_fused, speedup, naive_runs, fused_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--log-json", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model", type=str, default="microbench")
    parser.add_argument("--skip-prefill", action="store_true")
    parser.add_argument("--skip-quant", action="store_true")
    args = parser.parse_args()

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
    records = []
    timestamp = datetime.now(timezone.utc).isoformat()
    sha = _git_sha()
    device_meta = _device_meta()
    for label, *cfg in configs:
        print(f"\n{label}:")
        t_n, t_f, s, n_runs, f_runs = bench_shape(
            *cfg,
            outer_runs=args.outer_runs,
            warmup=args.warmup,
            iters=args.iters,
        )
        results.append((label, t_n, t_f, s))
        n_experts, M, K, N, bits, group_size, n_selected = cfg
        records.append(
            {
                "timestamp": timestamp,
                "kind": "gather_qmm_swiglu",
                "model": args.model,
                "label": label,
                "shape": {"M": M, "K": K, "N": N},
                "n_experts": n_experts,
                "n_selected": n_selected,
                "bits": bits,
                "group_size": group_size,
                "epsilon": None,
                "threadgroup": 0,
                "naive_time_us": t_n,
                "fused_time_us": t_f,
                "speedup_x": s,
                "naive_times_us": n_runs,
                "fused_times_us": f_runs,
                "outer_runs": args.outer_runs,
                "warmup": args.warmup,
                "iters": args.iters,
                "mlx_version": mx.__version__,
                "git_sha": sha,
                "device": device_meta,
                "tag": args.tag or None,
            }
        )

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

    if not args.skip_prefill:
        for label, *cfg in prefill_configs:
            print(f"\n{label}:")
            t_n, t_f, s, n_runs, f_runs = bench_shape(
                *cfg,
                outer_runs=args.outer_runs,
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append((label, t_n, t_f, s))
            n_experts, M, K, N, bits, group_size, n_selected = cfg
            records.append(
                {
                    "timestamp": timestamp,
                    "kind": "gather_qmm_swiglu",
                    "model": args.model,
                    "label": label,
                    "shape": {"M": M, "K": K, "N": N},
                    "n_experts": n_experts,
                    "n_selected": n_selected,
                    "bits": bits,
                    "group_size": group_size,
                    "epsilon": None,
                    "threadgroup": 0,
                    "naive_time_us": t_n,
                    "fused_time_us": t_f,
                    "speedup_x": s,
                    "naive_times_us": n_runs,
                    "fused_times_us": f_runs,
                    "outer_runs": args.outer_runs,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "mlx_version": mx.__version__,
                    "git_sha": sha,
                    "device": device_meta,
                    "tag": args.tag or None,
                }
            )

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

    if not args.skip_quant:
        for label, *cfg in quant_configs:
            print(f"\n{label}:")
            t_n, t_f, s, n_runs, f_runs = bench_shape(
                *cfg,
                outer_runs=args.outer_runs,
                warmup=args.warmup,
                iters=args.iters,
            )
            results.append((label, t_n, t_f, s))
            n_experts, M, K, N, bits, group_size, n_selected = cfg
            records.append(
                {
                    "timestamp": timestamp,
                    "kind": "gather_qmm_swiglu",
                    "model": args.model,
                    "label": label,
                    "shape": {"M": M, "K": K, "N": N},
                    "n_experts": n_experts,
                    "n_selected": n_selected,
                    "bits": bits,
                    "group_size": group_size,
                    "epsilon": None,
                    "threadgroup": 0,
                    "naive_time_us": t_n,
                    "fused_time_us": t_f,
                    "speedup_x": s,
                    "naive_times_us": n_runs,
                    "fused_times_us": f_runs,
                    "outer_runs": args.outer_runs,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "mlx_version": mx.__version__,
                    "git_sha": sha,
                    "device": device_meta,
                    "tag": args.tag or None,
                }
            )

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

    if args.log_json:
        path = Path(args.log_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        _append_json(path, records)


if __name__ == "__main__":
    main()
