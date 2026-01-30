"""Profiling utilities for custom Metal kernels.

Provides tools for timing individual kernels, inspecting generated Metal source,
and reviewing compilation/cache statistics.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ._compat import import_mx


def _maybe_sync(mx: Any) -> None:
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()


def time_kernel(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 5,
    iters: int = 20,
) -> dict[str, float]:
    """Time a kernel function and return detailed timing statistics.

    Args:
        fn: Callable to time (kernel or any MLX function).
        *args: Arguments to pass to ``fn``.
        warmup: Number of warm-up iterations.
        iters: Number of timed iterations.

    Returns:
        Dict with keys: ``median_us``, ``mean_us``, ``min_us``, ``max_us``,
        ``std_us``, ``iters``.
    """
    mx = import_mx()

    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if isinstance(result, (list, tuple)):
            mx.eval(*result)
        else:
            mx.eval(result)
    _maybe_sync(mx)

    # Timed
    times_us: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn(*args)
        if isinstance(result, (list, tuple)):
            mx.eval(*result)
        else:
            mx.eval(result)
        _maybe_sync(mx)
        times_us.append((time.perf_counter_ns() - t0) / 1e3)

    times_us.sort()
    n = len(times_us)
    mean = sum(times_us) / n
    variance = sum((t - mean) ** 2 for t in times_us) / n
    std = variance**0.5
    median = times_us[n // 2]

    return {
        "median_us": median,
        "mean_us": mean,
        "min_us": times_us[0],
        "max_us": times_us[-1],
        "std_us": std,
        "iters": float(iters),
    }


def memory_usage(fn: Callable[..., Any], *args: Any) -> dict[str, Any]:
    """Estimate peak memory for a kernel invocation.

    Computes input and output sizes from the MLX arrays. This is an estimate
    based on array metadata, not a GPU-level memory trace.

    Args:
        fn: Callable to profile.
        *args: Arguments to pass to ``fn``.

    Returns:
        Dict with keys: ``input_bytes``, ``output_bytes``, ``total_bytes``.
    """
    mx = import_mx()

    input_bytes = 0
    for a in args:
        if hasattr(a, "nbytes"):
            input_bytes += a.nbytes
        elif hasattr(a, "size") and hasattr(a, "dtype"):
            input_bytes += int(a.size) * a.dtype.size

    result = fn(*args)
    if isinstance(result, (list, tuple)):
        mx.eval(*result)
    else:
        mx.eval(result)
        result = [result]

    output_bytes = 0
    for r in result:
        if hasattr(r, "nbytes"):
            output_bytes += r.nbytes
        elif hasattr(r, "size") and hasattr(r, "dtype"):
            output_bytes += int(r.size) * r.dtype.size

    return {
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "total_bytes": input_bytes + output_bytes,
    }


def dump_msl(kernel: Any) -> str:
    """Extract the Metal Shading Language source from a MetalKernel.

    Args:
        kernel: A :class:`~zmlx.metal.MetalKernel` instance.

    Returns:
        The Metal source string. Returns a descriptive message if the kernel
        type is not recognized.
    """
    from .metal import MetalKernel

    if isinstance(kernel, MetalKernel):
        header = kernel.spec.header or ""
        source = kernel.spec.source or ""
        parts = []
        if header.strip():
            parts.append(f"// --- Header ---\n{header}")
        parts.append(f"// --- Source ---\n{source}")
        return "\n".join(parts)

    return f"<unrecognized kernel type: {type(kernel).__name__}>"


def kernel_stats() -> list[dict[str, Any]]:
    """Collect runtime statistics from all cached MetalKernel instances.

    Returns:
        List of dicts, each with keys: ``name``, ``compile_time_ms``,
        ``run_count``, ``total_run_time_ms``.
    """
    from .cache import GLOBAL_KERNEL_CACHE
    from .metal import MetalKernel

    stats: list[dict[str, Any]] = []
    for key in GLOBAL_KERNEL_CACHE.keys():
        k = GLOBAL_KERNEL_CACHE.get(key)
        if isinstance(k, MetalKernel):
            stats.append({
                "name": k.spec.name,
                "compile_time_ms": k.stats.compile_time_ms,
                "run_count": k.stats.run_count,
                "total_run_time_ms": k.stats.total_run_time_ms,
            })
    return stats


def analyze_bottlenecks(fn: Callable[..., Any], *args: Any, verbose: bool = True) -> dict[str, Any]:
    """Analyze a function/model pass for potential bottlenecks and fusions.

    Identifies sequences of operations that could be replaced by fused
    ZMLX kernels or jitted.

    Args:
        fn: Function or model forward pass to analyze.
        *args: Arguments to pass to ``fn``.
        verbose: If True, prints a summary report.

    Returns:
        Dict containing identified bottlenecks and suggested fusions.
    """
    mx = import_mx()
    
    # 1. Capture kernel stats before
    before_stats = {s["name"]: s for s in kernel_stats()}
    
    # 2. Run the function
    t0 = time.perf_counter_ns()
    result = fn(*args)
    if isinstance(result, (list, tuple)):
        mx.eval(*result)
    else:
        mx.eval(result)
    _maybe_sync(mx)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    
    # 3. Capture kernel stats after
    after_stats = {s["name"]: s for s in kernel_stats()}
    
    # 4. Compare
    new_kernels = []
    for name, stat in after_stats.items():
        if name not in before_stats:
            new_kernels.append(stat)
        elif stat["run_count"] > before_stats[name]["run_count"]:
            diff = stat.copy()
            diff["run_count"] -= before_stats[name]["run_count"]
            new_kernels.append(diff)
            
    # 5. Analyze patterns (Heuristic)
    suggestions = []
    
    # Check for many small kernels
    if len(new_kernels) > 50:
        suggestions.append("Found many small kernel launches. Consider using 'zmlx.jit' to fuse elementwise sequences.")
    
    # Check for common patterns if 'fn' is an nn.Module
    if hasattr(fn, "parameters"):
        # Suggest patching if not already patched
        if not hasattr(fn, "_zmlx_patch_result"):
            suggestions.append("Model is not patched. Try 'zmlx.patch.patch(model)' to fuse activations and norms.")
        else:
            res = fn._zmlx_patch_result
            if res.replacements == 0:
                 suggestions.append("ZMLX patch was applied but zero replacements were made. Check your pattern list.")

    report = {
        "elapsed_ms": elapsed_ms,
        "kernel_count": len(new_kernels),
        "suggestions": suggestions,
    }

    if verbose:
        print("\n=== ZMLX Bottleneck Analysis ===")
        print(f"Total time: {elapsed_ms:.2f} ms")
        print(f"Active kernels: {len(new_kernels)}")
        if suggestions:
            print("\nSuggestions:")
            for s in suggestions:
                print(f"  - {s}")
        else:
            print("\nNo obvious bottlenecks found. Your model is likely well-optimized!")
        print("================================\n")

    return report


__all__ = [
    "time_kernel",
    "memory_usage",
    "dump_msl",
    "kernel_stats",
    "analyze_bottlenecks",
]
