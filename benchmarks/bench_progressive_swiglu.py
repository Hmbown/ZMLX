#!/usr/bin/env python3
"""Sweep progressive SwiGLU epsilon on quantized GEMV (decode-like)."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

from zmlx.device import detect_device
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


def _multi_run(fn, *, outer_runs: int, warmup: int, iters: int) -> tuple[float, list[float]]:
    runs = []
    for _ in range(outer_runs):
        runs.append(_warmup_and_time(fn, warmup=warmup, iters=iters))
    runs.sort()
    return runs[len(runs) // 2], runs


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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


def _bench_case(
    B: int,
    K: int,
    N: int,
    *,
    bits: int,
    group_size: int,
    epsilons: list[float],
    threadgroup: int,
    outer_runs: int,
    warmup: int,
    iters: int,
    log_json: Path | None,
    tag: str | None,
    model_name: str,
) -> None:
    mx.random.seed(0)
    x = mx.random.normal((B, K)).astype(mx.float16)
    w_gate = mx.random.normal((N, K)).astype(mx.float32)
    w_up = mx.random.normal((N, K)).astype(mx.float32)

    gate_w, gate_scales, gate_biases = mx.quantize(w_gate, group_size=group_size, bits=bits, mode="affine")
    up_w, up_scales, up_biases = mx.quantize(w_up, group_size=group_size, bits=bits, mode="affine")

    def baseline():
        gate = mx.quantized_matmul(
            x,
            gate_w,
            scales=gate_scales,
            biases=gate_biases,
            group_size=group_size,
            bits=bits,
            mode="affine",
            transpose=True,
        )
        up = mx.quantized_matmul(
            x,
            up_w,
            scales=up_scales,
            biases=up_biases,
            group_size=group_size,
            bits=bits,
            mode="affine",
            transpose=True,
        )
        return gate * mx.sigmoid(gate) * up

    base = baseline()
    mx.eval(base)

    base_time, base_runs = _multi_run(baseline, outer_runs=outer_runs, warmup=warmup, iters=iters)
    print(f"Baseline: {base_time:.1f}us (median of {outer_runs} runs)")

    records: list[dict] = []
    device = detect_device()
    timestamp = datetime.now(timezone.utc).isoformat()
    sha = _git_sha()
    mlx_version = mx.__version__

    records.append(
        {
            "timestamp": timestamp,
            "kind": "progressive_swiglu_baseline",
            "model": model_name,
            "shape": {"B": B, "K": K, "N": N},
            "bits": bits,
            "group_size": group_size,
            "time_us": base_time,
            "times_us": base_runs,
            "outer_runs": outer_runs,
            "warmup": warmup,
            "iters": iters,
            "mlx_version": mlx_version,
            "git_sha": sha,
            "device": {
                "family": device.family,
                "variant": device.variant,
                "gpu_cores": device.gpu_cores,
                "memory_gb": device.memory_gb,
                "has_ray_tracing": device.has_ray_tracing,
            },
            "tag": tag,
        }
    )

    for eps in epsilons:
        def prog(eps=eps):
            return quant.fused_quantized_swiglu_gemv_progressive(
                x,
                gate_w,
                gate_scales,
                gate_biases,
                up_w,
                up_scales,
                up_biases,
                group_size=group_size,
                bits=bits,
                epsilon=eps,
            )

        out = prog()
        mx.eval(out)
        rel_err = float(mx.mean(mx.abs(out - base)) / (mx.mean(mx.abs(base)) + 1e-6))
        t, runs = _multi_run(prog, outer_runs=outer_runs, warmup=warmup, iters=iters)
        speedup = base_time / t if t else 0.0
        print(
            f"  eps={eps:<6g} time={t:>8.1f}us speedup={speedup:>5.2f}x rel_err={rel_err:>7.4f}"
        )
        records.append(
            {
                "timestamp": timestamp,
                "kind": "progressive_swiglu_gemv",
                "model": model_name,
                "shape": {"B": B, "K": K, "N": N},
                "bits": bits,
                "group_size": group_size,
                "epsilon": eps,
                "threadgroup": 0,
                "baseline_time_us": base_time,
                "time_us": t,
                "speedup_x": speedup,
                "rel_err": rel_err,
                "times_us": runs,
                "outer_runs": outer_runs,
                "warmup": warmup,
                "iters": iters,
                "mlx_version": mlx_version,
                "git_sha": sha,
                "device": {
                    "family": device.family,
                    "variant": device.variant,
                    "gpu_cores": device.gpu_cores,
                    "memory_gb": device.memory_gb,
                    "has_ray_tracing": device.has_ray_tracing,
                },
                "tag": tag,
            }
        )

    tg = threadgroup
    print(f"\nThreadgroup={tg}:")
    for eps in epsilons:
        def prog_tg(eps=eps):
            return quant.fused_quantized_swiglu_gemv_progressive(
                x,
                gate_w,
                gate_scales,
                gate_biases,
                up_w,
                up_scales,
                up_biases,
                group_size=group_size,
                bits=bits,
                epsilon=eps,
                threadgroup=tg,
            )

        out = prog_tg()
        mx.eval(out)
        rel_err = float(mx.mean(mx.abs(out - base)) / (mx.mean(mx.abs(base)) + 1e-6))
        t, runs = _multi_run(prog_tg, outer_runs=outer_runs, warmup=warmup, iters=iters)
        speedup = base_time / t if t else 0.0
        print(
            f"  eps={eps:<6g} time={t:>8.1f}us speedup={speedup:>5.2f}x rel_err={rel_err:>7.4f}"
        )
        records.append(
            {
                "timestamp": timestamp,
                "kind": "progressive_swiglu_gemv",
                "model": model_name,
                "shape": {"B": B, "K": K, "N": N},
                "bits": bits,
                "group_size": group_size,
                "epsilon": eps,
                "threadgroup": tg,
                "baseline_time_us": base_time,
                "time_us": t,
                "speedup_x": speedup,
                "rel_err": rel_err,
                "times_us": runs,
                "outer_runs": outer_runs,
                "warmup": warmup,
                "iters": iters,
                "mlx_version": mlx_version,
                "git_sha": sha,
                "device": {
                    "family": device.family,
                    "variant": device.variant,
                    "gpu_cores": device.gpu_cores,
                    "memory_gb": device.memory_gb,
                    "has_ray_tracing": device.has_ray_tracing,
                },
                "tag": tag,
            }
        )

    if log_json is not None:
        _append_json(log_json, records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--threadgroup", type=int, default=64)
    parser.add_argument("--eps", type=str, default="0,0.01,0.1,1,10,100")
    parser.add_argument("--outer-runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--log-json", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model", type=str, default="microbench")
    args = parser.parse_args()

    epsilons = [float(e.strip()) for e in args.eps.split(",") if e.strip()]
    log_json = Path(args.log_json) if args.log_json else None
    tag = args.tag or None

    if log_json is not None:
        log_json.parent.mkdir(parents=True, exist_ok=True)

    print("Progressive SwiGLU GEMV epsilon sweep")
    print("-" * 72)
    _bench_case(
        B=args.B,
        K=args.K,
        N=args.N,
        bits=args.bits,
        group_size=args.group_size,
        epsilons=epsilons,
        threadgroup=args.threadgroup,
        outer_runs=args.outer_runs,
        warmup=args.warmup,
        iters=args.iters,
        log_json=log_json,
        tag=tag,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
