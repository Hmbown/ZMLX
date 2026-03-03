#!/usr/bin/env python3
"""Microbench for K=8 fp32/no-FMA weighted-sum MoE combine specialization."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx

from zmlx.kernels.moe import (
    moe_combine_fp32_no_fma,
    moe_combine_weighted_sum_fp32_no_fma,
)


def _git_short_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _bench_case(
    *,
    expert_outputs: Any,
    weights: Any,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        mx.eval(moe_combine_fp32_no_fma(expert_outputs, weights))
        mx.eval(moe_combine_weighted_sum_fp32_no_fma(expert_outputs, weights))

    base_times_us: list[float] = []
    spec_times_us: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        out_base = moe_combine_fp32_no_fma(expert_outputs, weights)
        mx.eval(out_base)
        t1 = time.perf_counter_ns()
        out_spec = moe_combine_weighted_sum_fp32_no_fma(expert_outputs, weights)
        mx.eval(out_spec)
        t2 = time.perf_counter_ns()
        base_times_us.append((t1 - t0) / 1e3)
        spec_times_us.append((t2 - t1) / 1e3)

    out_base = moe_combine_fp32_no_fma(expert_outputs, weights)
    out_spec = moe_combine_weighted_sum_fp32_no_fma(expert_outputs, weights)
    mx.eval(out_base, out_spec)
    max_abs = float(mx.max(mx.abs(out_base - out_spec)).item())

    base_med = float(statistics.median(base_times_us))
    spec_med = float(statistics.median(spec_times_us))
    return {
        "base_median_us": round(base_med, 3),
        "specialized_median_us": round(spec_med, 3),
        "specialized_over_base": round(spec_med / base_med, 6) if base_med > 0 else 0.0,
        "max_abs_diff": max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark moe_combine_weighted_sum_fp32_no_fma against "
            "moe_combine_fp32_no_fma for K=8."
        )
    )
    parser.add_argument("--json-out", required=True, help="Output capsule JSON path.")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["1x8x1024", "1x8x2048", "1x8x4096", "2x8x2048", "4x8x4096"],
        help="Case list in BxKxD form (K must be 8).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.json_out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for case in args.cases:
        try:
            b_raw, k_raw, d_raw = case.lower().split("x")
            B = int(b_raw)
            K = int(k_raw)
            D = int(d_raw)
        except Exception as exc:  # pragma: no cover - CLI guard
            raise SystemExit(f"invalid case {case!r}: expected BxKxD") from exc
        if K != 8:
            raise SystemExit(f"invalid case {case!r}: this bench requires K=8")
        if B <= 0 or D <= 0:
            raise SystemExit(f"invalid case {case!r}: B and D must be > 0")

        mx.random.seed(20260211 + B + D)
        expert_outputs = mx.random.normal((B, K, D)).astype(mx.float16)
        weights = mx.softmax(mx.random.normal((B, K)), axis=-1).astype(mx.float32)
        metrics = _bench_case(
            expert_outputs=expert_outputs,
            weights=weights,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
        )
        rows.append({"B": B, "K": K, "D": D, **metrics})

    ratios = [float(r["specialized_over_base"]) for r in rows]
    capsule = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "platform": platform.platform(),
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "git_commit": _git_short_commit(),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
        },
        "summary": {
            "cases": len(rows),
            "median_specialized_over_base": (
                round(float(statistics.median(ratios)), 6) if ratios else None
            ),
            "wins_specialized": sum(1 for x in ratios if x < 1.0),
            "max_abs_diff": max(float(r["max_abs_diff"]) for r in rows) if rows else 0.0,
        },
        "rows": rows,
    }
    out_path.write_text(json.dumps(capsule, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote capsule: {_safe_relpath(out_path, repo_root)}")


if __name__ == "__main__":
    main()
