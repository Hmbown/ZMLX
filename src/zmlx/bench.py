"""Benchmarking utilities for custom Metal kernels.

Provides a ``compare()`` function that benchmarks multiple implementations
side-by-side across shapes and dtypes, printing a formatted comparison table.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from ._compat import import_mx


def _maybe_sync(mx: Any) -> None:
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()


def _time_fn(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    mx: Any,
    warmup: int,
    iters: int,
) -> float:
    """Time a function and return median latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        mx.eval(result) if hasattr(result, "shape") else mx.eval(*result)
    _maybe_sync(mx)

    # Timed iterations
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn(*args)
        mx.eval(result) if hasattr(result, "shape") else mx.eval(*result)
        _maybe_sync(mx)
        times.append((time.perf_counter_ns() - t0) / 1e3)  # ns -> us

    times.sort()
    # Return median
    mid = len(times) // 2
    return times[mid]


def compare(
    implementations: dict[str, Callable[..., Any]],
    shapes: Sequence[tuple[int, ...]],
    *,
    dtypes: Sequence[Any] | None = None,
    n_inputs: int = 1,
    warmup: int = 5,
    iters: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Benchmark multiple implementations across shapes and dtypes.

    Prints a formatted comparison table and returns raw results.

    Args:
        implementations: Mapping of name to callable (e.g.
            ``{"ZMLX": my_op, "MLX": mx.nn.mish}``).
        shapes: Input shapes to benchmark.
        dtypes: Dtypes to benchmark. Defaults to ``[mx.float32, mx.float16]``.
        n_inputs: Number of input arrays to generate per test case.
        warmup: Warm-up iterations before timing.
        iters: Timed iterations per measurement.
        seed: Random seed for reproducibility.

    Returns:
        List of result dicts with keys: shape, dtype, and per-implementation
        latency_us and speedup fields.
    """
    mx = import_mx()
    if dtypes is None:
        dtypes = [mx.float32, mx.float16]

    mx.random.seed(seed)

    names = list(implementations.keys())
    results: list[dict[str, Any]] = []

    # Print header
    name_cols = "  ".join(f"{n:>12s}" for n in names)
    print(f"{'Shape':>20s}  {'Dtype':>10s}  {name_cols}  {'Speedup':>10s}")
    print("-" * (20 + 10 + len(names) * 14 + 14 + 4))

    baseline_name = names[0]

    for shape in shapes:
        for dtype in dtypes:
            inputs = [mx.random.normal(shape).astype(dtype) for _ in range(n_inputs)]
            args = tuple(inputs)

            timings: dict[str, float] = {}
            for name, fn in implementations.items():
                try:
                    us = _time_fn(fn, args, mx, warmup=warmup, iters=iters)
                    timings[name] = us
                except Exception as e:
                    timings[name] = float("nan")
                    print(f"  WARNING: {name} failed for shape={shape}, dtype={dtype}: {e}")

            # Calculate speedup (baseline / challenger)
            baseline_us = timings.get(baseline_name, float("nan"))
            if len(names) >= 2:
                challenger_us = timings.get(names[1], float("nan"))
                if challenger_us > 0 and baseline_us > 0:
                    speedup = baseline_us / challenger_us
                else:
                    speedup = float("nan")
            else:
                speedup = 1.0

            # Format row
            dtype_str = str(dtype).replace("mlx.core.", "")
            timing_strs = "  ".join(
                f"{timings[n]:>10.1f}us" if timings[n] == timings[n] else f"{'ERR':>12s}"
                for n in names
            )
            speedup_str = f"{speedup:.2f}x" if speedup == speedup else "N/A"
            print(f"{str(shape):>20s}  {dtype_str:>10s}  {timing_strs}  {speedup_str:>10s}")

            row: dict[str, Any] = {"shape": shape, "dtype": str(dtype)}
            for name in names:
                row[f"{name}_us"] = timings[name]
            row["speedup"] = speedup
            results.append(row)

    return results


__all__ = [
    "compare",
]
