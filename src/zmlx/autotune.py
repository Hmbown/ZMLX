from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ._compat import import_mx
from .metal import MetalKernel


@dataclass(frozen=True)
class AutotuneKey:
    kernel_name: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    grid: tuple[int, int, int]


@dataclass(frozen=True)
class AutotuneResult:
    best_threadgroup: tuple[int, int, int]
    timings_ms: dict[tuple[int, int, int], float]


GLOBAL_AUTOTUNE_CACHE: dict[AutotuneKey, tuple[int, int, int]] = {}


def _maybe_sync(mx: Any) -> None:
    # MLX may expose different sync mechanisms across versions/backends.
    # `mx.eval` is the primary barrier we rely on.
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()


def get_autotuned_threadgroup(
    kernel: MetalKernel,
    *,
    inputs: Sequence[Any],
    template: list[tuple[str, Any]] | None,
    output_shapes: list[Sequence[int]],
    output_dtypes: list[Any],
    grid: tuple[int, int, int],
    candidates: Sequence[tuple[int, int, int]] | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> tuple[int, int, int]:
    """Get the best threadgroup size, either from cache or by running a search.

    Looks up the kernel + input signature in ``GLOBAL_AUTOTUNE_CACHE``. On a
    cache miss, runs :func:`autotune_threadgroup` and stores the winner.

    Args:
        kernel: The :class:`MetalKernel` to tune.
        inputs: Input arrays to pass to the kernel.
        template: Metal template specializations (e.g. ``[("T", mx.float32)]``).
        output_shapes: Shape of each output array.
        output_dtypes: Dtype of each output array.
        grid: Metal grid dimensions ``(x, y, z)``.
        candidates: Threadgroup sizes to try.  Defaults to common 1-D values
            ``[(32,1,1) .. (1024,1,1)]``.
        warmup: Number of untimed warm-up iterations per candidate.
        iters: Number of timed iterations per candidate.

    Returns:
        The ``(x, y, z)`` threadgroup tuple that achieved the lowest average time.
    """
    key = AutotuneKey(
        kernel_name=kernel.spec.name,
        input_shapes=tuple(tuple(x.shape) for x in inputs),
        input_dtypes=tuple(str(x.dtype) for x in inputs),
        grid=grid,
    )
    
    if key in GLOBAL_AUTOTUNE_CACHE:
        return GLOBAL_AUTOTUNE_CACHE[key]
    
    # Default 1D candidates if none provided.
    if candidates is None:
        candidates = [(x, 1, 1) for x in (32, 64, 128, 256, 512, 1024)]
    
    res = autotune_threadgroup(
        kernel,
        inputs=inputs,
        template=template,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        grid=grid,
        candidates=candidates,
        warmup=warmup,
        iters=iters,
    )
    
    GLOBAL_AUTOTUNE_CACHE[key] = res.best_threadgroup
    return res.best_threadgroup


class KernelSearch:
    """Utility to search across DIFFERENT kernel implementations for the same math."""

    def __init__(self, name: str):
        self.name = name
        self.candidates: list[Callable] = []
        self.best_fn: Callable | None = None

    def add_candidate(self, fn: Callable):
        """Register a candidate implementation to benchmark.

        Args:
            fn: Callable implementing the same math as other candidates.
        """
        self.candidates.append(fn)

    def find_best(self, *inputs: Any, iters: int = 10) -> Callable:
        """Return the fastest candidate based on repeated timing.

        Args:
            *inputs: Inputs to pass to each candidate.
            iters: Number of timed iterations per candidate.

        Returns:
            The fastest candidate callable.
        """
        mx = import_mx()
        best_time = float("inf")
        
        for fn in self.candidates:
            # Warmup
            mx.eval(fn(*inputs))
            _maybe_sync(mx)
            
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                mx.eval(fn(*inputs))
            _maybe_sync(mx)
            elapsed = (time.perf_counter_ns() - t0) / iters
            
            if elapsed < best_time:
                best_time = elapsed
                self.best_fn = fn
        
        return self.best_fn or self.candidates[0]


def autotune_threadgroup(
    kernel: MetalKernel,
    *,
    inputs: Sequence[Any],
    template: list[tuple[str, Any]] | None,
    output_shapes: list[Sequence[int]],
    output_dtypes: list[Any],
    grid: tuple[int, int, int],
    candidates: Sequence[tuple[int, int, int]],
    warmup: int = 3,
    iters: int = 10,
) -> AutotuneResult:
    """Search for the best threadgroup size among the provided candidates.

    Runs the kernel with every candidate threadgroup, measures wall-clock
    time per iteration, and returns the fastest.

    Args:
        kernel: The :class:`MetalKernel` to benchmark.
        inputs: Input arrays to pass to the kernel.
        template: Metal template specializations (e.g. ``[("T", mx.float32)]``).
        output_shapes: Shape of each output array.
        output_dtypes: Dtype of each output array.
        grid: Metal grid dimensions ``(x, y, z)``.
        candidates: Sequence of ``(x, y, z)`` threadgroup sizes to evaluate.
        warmup: Number of untimed warm-up iterations per candidate. Default 3.
        iters: Number of timed iterations per candidate. Default 10.

    Returns:
        An :class:`AutotuneResult` with ``best_threadgroup`` and per-candidate
        ``timings_ms``.
    """
    mx = import_mx()
    timings: dict[tuple[int, int, int], float] = {}

    for tg in candidates:
        # Check if TG is valid for this grid (Metal constraint: TG size <= Grid size)
        # Note: actually Grid can be smaller than TG in dispatchThreads, 
        # but for simplicity we clamp or skip if TG is clearly too big for the device limits.
        if tg[0] * tg[1] * tg[2] > 1024:
            continue

        try:
            # Warmup
            for _ in range(max(0, warmup)):
                outs = kernel(
                    *inputs,
                    template=template,
                    grid=grid,
                    threadgroup=tg,
                    output_shapes=output_shapes,
                    output_dtypes=output_dtypes,
                )
                mx.eval(*outs)
            _maybe_sync(mx)

            # Timed
            start = time.perf_counter()
            for _ in range(max(1, iters)):
                outs = kernel(
                    *inputs,
                    template=template,
                    grid=grid,
                    threadgroup=tg,
                    output_shapes=output_shapes,
                    output_dtypes=output_dtypes,
                )
                mx.eval(*outs)
            _maybe_sync(mx)
            elapsed = time.perf_counter() - start
            timings[tg] = (elapsed / max(1, iters)) * 1000.0
        except Exception:
            # Skip candidates that fail (e.g. out of resources)
            continue

    if not timings:
        # Fallback
        return AutotuneResult(best_threadgroup=(1, 1, 1), timings_ms={(1, 1, 1): 0.0})

    best = min(timings.items(), key=lambda kv: kv[1])[0]
    return AutotuneResult(best_threadgroup=best, timings_ms=timings)


def _cache_file_path() -> str | None:
    """Return the path to the persistent autotune cache file.

    Returns ``~/.cache/zmlx/autotune_v1.json`` on macOS, or ``None`` if
    the directory cannot be determined.
    """
    import os
    from pathlib import Path

    cache_dir = os.environ.get("ZMLX_CACHE_DIR")
    if cache_dir is None:
        home = Path.home()
        cache_dir = str(home / ".cache" / "zmlx")

    return str(Path(cache_dir) / "autotune_v1.json")


def _device_cache_key() -> str:
    """Return a key incorporating device family and MLX version."""
    try:
        from .device import detect_device

        dev = detect_device()
        family = f"{dev.family}_{dev.variant}".rstrip("_")
    except Exception:
        family = "unknown"

    try:
        import mlx.core as mx

        mlx_version = mx.__version__
    except Exception:
        mlx_version = "unknown"

    return f"{family}_{mlx_version}"


def save_autotune_cache(path: str | None = None) -> None:
    """Save the global autotune cache to disk.

    Args:
        path: File path. Defaults to ``~/.cache/zmlx/autotune_v1.json``.
    """
    import json
    from pathlib import Path

    if path is None:
        path = _cache_file_path()
    if path is None:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    device_key = _device_cache_key()

    # Load existing file to preserve other device profiles
    existing: dict[str, Any] = {}
    if Path(path).exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Serialize current cache
    entries: dict[str, str] = {}
    for key, tg in GLOBAL_AUTOTUNE_CACHE.items():
        key_str = json.dumps({
            "name": key.kernel_name,
            "shapes": key.input_shapes,
            "dtypes": key.input_dtypes,
            "grid": key.grid,
        })
        entries[key_str] = json.dumps(tg)

    existing[device_key] = entries

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def load_autotune_cache(path: str | None = None) -> int:
    """Load the global autotune cache from disk.

    Args:
        path: File path. Defaults to ``~/.cache/zmlx/autotune_v1.json``.

    Returns:
        Number of cache entries loaded.
    """
    import json
    from pathlib import Path

    if path is None:
        path = _cache_file_path()
    if path is None or not Path(path).exists():
        return 0

    device_key = _device_cache_key()

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0

    entries = data.get(device_key, {})
    count = 0

    for key_str, tg_str in entries.items():
        try:
            kd = json.loads(key_str)
            tg = tuple(json.loads(tg_str))
            key = AutotuneKey(
                kernel_name=kd["name"],
                input_shapes=tuple(tuple(s) for s in kd["shapes"]),
                input_dtypes=tuple(kd["dtypes"]),
                grid=tuple(kd["grid"]),
            )
            GLOBAL_AUTOTUNE_CACHE[key] = tg  # type: ignore[assignment]
            count += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return count


__all__ = [
    "AutotuneResult",
    "autotune_threadgroup",
    "get_autotuned_threadgroup",
    "save_autotune_cache",
    "load_autotune_cache",
]
