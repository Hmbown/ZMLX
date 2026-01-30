from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._compat import import_mx
from .metal import MetalKernel


@dataclass(frozen=True)
class AutotuneKey:
    kernel_name: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    grid: tuple[int, int, int]
    template_params: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class AutotuneConfig:
    threadgroup: tuple[int, int, int]
    template: tuple[tuple[str, Any], ...] = ()

    def to_list(self) -> list[tuple[str, Any]]:
        return list(self.template)


@dataclass(frozen=True)
class AutotuneResult:
    best_config: AutotuneConfig
    timings_ms: dict[AutotuneConfig, float]

    @property
    def best_threadgroup(self) -> tuple[int, int, int]:
        return self.best_config.threadgroup


GLOBAL_AUTOTUNE_CACHE: dict[AutotuneKey, AutotuneConfig] = {}

# ---------------------------------------------------------------------------
# Fast cache: keyed on (kernel_name, id(input0), id(input1), ...)
# Avoids constructing a full AutotuneKey (which copies shapes/dtypes) on the
# hot path.  Bounded to _FAST_CACHE_MAX entries; cleared entirely on overflow
# to avoid stale id() references (Python can reuse object ids).
# ---------------------------------------------------------------------------
_FAST_CACHE: dict[tuple, AutotuneConfig] = {}
_FAST_CACHE_MAX = 512


def _fast_cache_lookup(
    kernel_name: str, inputs: Sequence[Any]
) -> AutotuneConfig | None:
    key = (kernel_name,) + tuple(id(x) for x in inputs)
    return _FAST_CACHE.get(key)


def _fast_cache_store(
    kernel_name: str, inputs: Sequence[Any], config: AutotuneConfig
) -> None:
    global _FAST_CACHE
    if len(_FAST_CACHE) >= _FAST_CACHE_MAX:
        _FAST_CACHE = {}
    key = (kernel_name,) + tuple(id(x) for x in inputs)
    _FAST_CACHE[key] = config


def _maybe_sync(mx: Any) -> None:
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()


def get_autotuned_config(
    kernel: MetalKernel,
    *,
    inputs: Sequence[Any],
    grid: tuple[int, int, int] | Callable[[tuple[int, int, int]], tuple[int, int, int]],
    template_candidates: Sequence[list[tuple[str, Any]]] | None = None,
    threadgroup_candidates: Sequence[tuple[int, int, int]] | None = None,
    output_shapes: list[Sequence[int]] | None = None,
    output_dtypes: list[Any] | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> AutotuneConfig:
    """Get the best configuration (threadgroup + template), either from cache or by running a search.
    """
    # Fast path: id-based lookup avoids constructing AutotuneKey entirely
    fast = _fast_cache_lookup(kernel.spec.name, inputs)
    if fast is not None:
        return fast

    # We use a simplified key for template params in the AutotuneKey
    # to avoid issues with non-hashable types.
    t_params: list[tuple[str, str]] = []
    if template_candidates and len(template_candidates) > 1:
        t_params.append(("tuning", "templates"))

    # If grid is a callable, we use a placeholder in the key
    # but ideally we should evaluate it with a default.
    grid_val = grid if isinstance(grid, tuple) else grid((1, 1, 1))

    key = AutotuneKey(
        kernel_name=kernel.spec.name,
        input_shapes=tuple(tuple(x.shape) for x in inputs),
        input_dtypes=tuple(str(x.dtype) for x in inputs),
        grid=grid_val,
        template_params=tuple(t_params),
    )

    if key in GLOBAL_AUTOTUNE_CACHE:
        config = GLOBAL_AUTOTUNE_CACHE[key]
        _fast_cache_store(kernel.spec.name, inputs, config)
        return config

    # Defaults
    if threadgroup_candidates is None:
        threadgroup_candidates = [(x, 1, 1) for x in (32, 64, 128, 256, 512, 1024)]

    if template_candidates is None:
        template_candidates = [[]]

    res = autotune_kernel(
        kernel,
        inputs=inputs,
        grid=grid,
        template_candidates=template_candidates,
        threadgroup_candidates=threadgroup_candidates,
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        warmup=warmup,
        iters=iters,
    )
    
    GLOBAL_AUTOTUNE_CACHE[key] = res.best_config
    _fast_cache_store(kernel.spec.name, inputs, res.best_config)
    return res.best_config


def autotune_kernel(
    kernel: MetalKernel,
    *,
    inputs: Sequence[Any],
    grid: tuple[int, int, int] | Callable[[tuple[int, int, int]], tuple[int, int, int]],
    template_candidates: Sequence[list[tuple[str, Any]]],
    threadgroup_candidates: Sequence[tuple[int, int, int]],
    output_shapes: list[Sequence[int]] | None = None,
    output_dtypes: list[Any] | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> AutotuneResult:
    """Search for the best (threadgroup, template) pair among provided candidates.
    """
    mx = import_mx()
    timings: dict[AutotuneConfig, float] = {}

    for template_list in template_candidates:
        template = tuple(template_list)
        for tg in threadgroup_candidates:
            if tg[0] * tg[1] * tg[2] > 1024:
                continue
            
            # Evaluate grid if it's a callable
            current_grid = grid(tg) if callable(grid) else grid

            config = AutotuneConfig(threadgroup=tg, template=template)
            try:
                # Warmup
                for _ in range(max(0, warmup)):
                    outs = kernel(
                        *inputs,
                        template=list(template),
                        grid=current_grid,
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
                        template=list(template),
                        grid=current_grid,
                        threadgroup=tg,
                        output_shapes=output_shapes,
                        output_dtypes=output_dtypes,
                    )
                    mx.eval(*outs)
                _maybe_sync(mx)
                elapsed = time.perf_counter() - start
                timings[config] = (elapsed / max(1, iters)) * 1000.0
            except Exception:
                continue

    if not timings:
        fallback = AutotuneConfig(threadgroup=(1, 1, 1), template=tuple(template_candidates[0]))
        return AutotuneResult(best_config=fallback, timings_ms={fallback: 0.0})

    best = min(timings.items(), key=lambda kv: kv[1])[0]
    return AutotuneResult(best_config=best, timings_ms=timings)


def autotune_threadgroup(
    kernel: MetalKernel,
    *,
    inputs: Sequence[Any],
    template: list[tuple[str, Any]],
    output_shapes: list[Sequence[int]] | None = None,
    output_dtypes: list[Any] | None = None,
    grid: tuple[int, int, int],
    candidates: Sequence[tuple[int, int, int]],
    warmup: int = 3,
    iters: int = 10,
) -> AutotuneResult:
    """Search for the best threadgroup size among *candidates*.

    Convenience wrapper around :func:`autotune_kernel` that fixes a single
    template and only varies the threadgroup.
    """
    return autotune_kernel(
        kernel,
        inputs=inputs,
        grid=grid,
        template_candidates=[template],
        threadgroup_candidates=list(candidates),
        output_shapes=output_shapes,
        output_dtypes=output_dtypes,
        warmup=warmup,
        iters=iters,
    )


def autotune(
    threadgroups: Sequence[tuple[int, int, int]] | None = None,
    templates: Sequence[dict[str, Any]] | None = None,
    warmup: int = 3,
    iters: int = 10,
):
    """Decorator to automatically autotune a kernel-launching function.
    """
    def decorator(fn: Callable):
        def wrapper(*args, **kwargs):
            # For now, this is a placeholder for a more complex implementation
            # that might inspect the kernel being launched.
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def _cache_file_path() -> str | None:
    cache_dir = os.environ.get("ZMLX_CACHE_DIR")
    if cache_dir is None:
        home = Path.home()
        cache_dir = str(home / ".cache" / "zmlx")
    return str(Path(cache_dir) / "autotune_v2.json")


def _device_cache_key() -> str:
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
    if path is None:
        path = _cache_file_path()
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    device_key = _device_cache_key()
    existing: dict[str, Any] = {}
    if Path(path).exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            pass
    entries: dict[str, Any] = {}
    for key, config in GLOBAL_AUTOTUNE_CACHE.items():
        key_str = json.dumps({
            "name": key.kernel_name,
            "shapes": key.input_shapes,
            "dtypes": key.input_dtypes,
            "grid": key.grid,
            "t_params": key.template_params,
        })
        entries[key_str] = {
            "tg": config.threadgroup,
            "template": config.template
        }
    existing[device_key] = entries
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def load_autotune_cache(path: str | None = None) -> int:
    if path is None:
        path = _cache_file_path()
    if path is None or not Path(path).exists():
        return 0
    device_key = _device_cache_key()
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return 0
    entries = data.get(device_key, {})
    count = 0
    for key_str, val in entries.items():
        try:
            kd = json.loads(key_str)
            tg = tuple(val["tg"])
            template = tuple(tuple(t) for t in val["template"])
            key = AutotuneKey(
                kernel_name=kd["name"],
                input_shapes=tuple(tuple(s) for s in kd["shapes"]),
                input_dtypes=tuple(kd["dtypes"]),
                grid=tuple(kd["grid"]),
                template_params=tuple(tuple(tp) for tp in kd.get("t_params", ())),
            )
            GLOBAL_AUTOTUNE_CACHE[key] = AutotuneConfig(threadgroup=tg, template=template)  # type: ignore
            count += 1
        except Exception:
            continue
    return count

__all__ = [
    "AutotuneResult",
    "AutotuneConfig",
    "autotune_kernel",
    "get_autotuned_config",
    "save_autotune_cache",
    "load_autotune_cache",
    "autotune",
]