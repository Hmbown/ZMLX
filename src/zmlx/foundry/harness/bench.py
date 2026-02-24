"""Adaptive benchmarking with timeout, warmup, and percentile stats.

Adapted from DataFoundry's harness/bench.py.  Runs a kernel repeatedly,
adaptively increasing the repeat count until timing noise stabilises or
the timeout is hit.
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable


def _percentile(xs: list[float], q: float) -> float:
    """Simple percentile on a pre-sorted list (nearest-rank)."""
    xs_sorted = sorted(xs)
    idx = int(round((len(xs_sorted) - 1) * q))
    return float(xs_sorted[max(0, min(len(xs_sorted) - 1, idx))])


def bench(
    *,
    run_once: Callable[[], None],
    sync: Callable[[], None],
    warmup: int,
    repeats: int,
    timeout_s: float,
    adaptive: bool = True,
) -> tuple[bool, dict[str, float | None], bool]:
    """Benchmark a kernel and return ``(ok, latency_ms_dict, timed_out)``.

    Parameters:
        run_once: Callable that dispatches the kernel once.
        sync: Callable that forces GPU completion (e.g. ``mx.synchronize``).
        warmup: Number of warmup iterations (not timed).
        repeats: Target number of timed iterations.
        timeout_s: Wall-clock budget in seconds.
        adaptive: If ``True``, adaptively increase repeats until noise < 10%.

    Returns:
        A 3-tuple:
        * ``ok`` -- ``True`` if at least one timed sample was collected.
        * ``latency_ms`` -- dict with keys ``p50``, ``p90``, ``min``, ``mean``
          (values are ``None`` on failure).
        * ``timed_out`` -- ``True`` if the timeout was hit before completing.
    """
    # Warmup
    for _ in range(max(0, warmup)):
        run_once()
    sync()

    times: list[float] = []
    t_start = time.perf_counter()
    timed_out = False

    r = max(1, repeats)
    while True:
        times.clear()
        for _ in range(r):
            t0 = time.perf_counter()
            run_once()
            sync()
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)
            if (time.perf_counter() - t_start) > timeout_s:
                timed_out = True
                break
        if timed_out:
            break
        if not adaptive or r >= repeats:
            break
        # Crude noise estimate: if CV < 10% we have enough samples.
        if len(times) >= 5:
            mean = statistics.mean(times)
            stdev = statistics.pstdev(times)
            if mean > 0 and (stdev / mean) < 0.10:
                break
        r = min(r * 2, 256)
        if r >= repeats:
            break

    if timed_out or not times:
        return False, {"p50": None, "p90": None, "min": None, "mean": None}, True

    p50 = _percentile(times, 0.50)
    p90 = _percentile(times, 0.90)
    mn = min(times)
    mean = statistics.mean(times)
    return True, {"p50": p50, "p90": p90, "min": mn, "mean": mean}, False
