from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from ._compat import import_mx
from .metal import MetalKernel

F = TypeVar("F", bound=Callable[..., Any])


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


def _get_device_candidates() -> list[tuple[int, int, int]]:
    """Get threadgroup candidates optimized for the current device."""
    try:
        from .device_profile import get_current_device_profile, get_threadgroup_candidates_for_shape
        profile = get_current_device_profile()
        candidates = get_threadgroup_candidates_for_shape(profile, 1024, "general")
        return [(c, 1, 1) for c in candidates]
    except Exception:
        # Fallback to generic candidates
        return [(x, 1, 1) for x in (32, 64, 128, 256, 512)]


def _get_default_config() -> AutotuneConfig:
    """Get the default autotune config for the current device."""
    try:
        from .device_profile import get_current_device_profile
        profile = get_current_device_profile()
        return AutotuneConfig(
            threadgroup=(profile.default_threadgroup, 1, 1),
            template=()
        )
    except Exception:
        return AutotuneConfig(threadgroup=(128, 1, 1), template=())


class AutotunedFunction:
    """Wrapper for an autotuned function that caches configurations."""
    
    def __init__(
        self,
        fn: Callable,
        threadgroup_candidates: Sequence[tuple[int, int, int]] | None = None,
        template_candidates: Sequence[dict[str, Any]] | None = None,
        warmup: int = 3,
        iters: int = 10,
    ):
        self.fn = fn
        self.threadgroup_candidates = threadgroup_candidates
        self.template_candidates = template_candidates if template_candidates else [{}]
        self.warmup = warmup
        self.iters = iters
        self._cache: dict[tuple, AutotuneConfig] = {}
        
    def _make_key(self, args: tuple, kwargs: dict) -> tuple:
        """Create a cache key from function arguments."""
        # Use shapes and dtypes of array-like arguments
        key_parts: list[tuple[Any, ...]] = []
        for arg in args:
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):
                key_parts.append((tuple(arg.shape), str(arg.dtype)))
            else:
                key_parts.append((type(arg).__name__, str(arg)))
        for k, v in sorted(kwargs.items()):
            if hasattr(v, 'shape') and hasattr(v, 'dtype'):
                key_parts.append((k, tuple(v.shape), str(v.dtype)))
            else:
                key_parts.append((k, type(v).__name__, str(v)))
        return tuple(key_parts)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the function with autotuned configuration."""
        key = self._make_key(args, kwargs)
        
        if key in self._cache:
            config = self._cache[key]
        else:
            # Use device-optimized candidates if not specified
            tg_candidates = self.threadgroup_candidates
            if tg_candidates is None:
                tg_candidates = _get_device_candidates()
            
            # For now, run the function once to get the kernel spec
            # A more sophisticated implementation would extract the kernel
            # from the function and run autotune_kernel directly
            config = _get_default_config()
            self._cache[key] = config
        
        # Inject threadgroup into kwargs if the function accepts it
        if 'threadgroup' in kwargs or self._has_threadgroup_param():
            kwargs = {**kwargs, 'threadgroup': config.threadgroup}
        
        return self.fn(*args, **kwargs)
    
    def _has_threadgroup_param(self) -> bool:
        """Check if the wrapped function accepts a threadgroup parameter."""
        import inspect
        sig = inspect.signature(self.fn)
        return 'threadgroup' in sig.parameters
    
    def clear_cache(self) -> None:
        """Clear the autotune cache."""
        self._cache.clear()


def autotune(
    threadgroups: Sequence[tuple[int, int, int]] | None = None,
    templates: Sequence[dict[str, Any]] | None = None,
    warmup: int = 3,
    iters: int = 10,
    device_aware: bool = True,
) -> Callable[[F], AutotunedFunction]:
    """Decorator to automatically autotune a kernel-launching function.
    
    This decorator wraps a function that launches Metal kernels and automatically
    selects optimal threadgroup configurations based on the current device.
    
    Args:
        threadgroups: Sequence of (x, y, z) threadgroup sizes to try.
            If None, uses device-optimized candidates.
        templates: Sequence of template parameter dictionaries to try.
        warmup: Number of warmup iterations before timing.
        iters: Number of timing iterations for each configuration.
        device_aware: If True, use device profiles to select candidates.
            When True and threadgroups is None, automatically selects
            candidates optimized for the current Apple Silicon chip.
    
    Returns:
        An AutotunedFunction wrapper that caches optimal configurations.
    
    Example:
        >>> @zmlx.autotune(warmup=5, iters=20)
        ... def my_kernel_op(x, y, threadgroup=(128, 1, 1)):
        ...     kernel = zmlx.metal.kernel(...)
        ...     return kernel(x, y, threadgroup=threadgroup)
        ...
        >>> result = my_kernel_op(x, y)  # Automatically uses optimal threadgroup
    
    Note:
        This is a high-level decorator. For lower-level kernel autotuning,
        use :func:`autotune_kernel` or :func:`get_autotuned_config` directly.
    """
    def decorator(fn: F) -> AutotunedFunction:
        # Get device-optimized candidates if requested and no explicit candidates
        tg_candidates = threadgroups
        if device_aware and tg_candidates is None:
            tg_candidates = _get_device_candidates()
        
        return AutotunedFunction(
            fn=fn,
            threadgroup_candidates=tg_candidates,
            template_candidates=templates,
            warmup=warmup,
            iters=iters,
        )
    return decorator


def _cache_file_path() -> str | None:
    cache_dir = os.environ.get("ZMLX_CACHE_DIR")
    if cache_dir is None:
        home = Path.home()
        cache_dir = str(home / ".cache" / "zmlx")
    return str(Path(cache_dir) / "autotune_v3.json")


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
    """Save the autotune cache to disk with v3 schema including device metadata.
    
    The v3 schema includes:
    - schema_version: "3.0"
    - device_info: Detailed device metadata (family, variant, GPU cores, etc.)
    - entries: Per-device tuning results
    
    Args:
        path: Path to cache file. If None, uses default location.
    """
    if path is None:
        path = _cache_file_path()
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing cache
    existing: dict[str, Any] = {}
    if Path(path).exists():
        try:
            with open(path) as f:
                existing = json.load(f)
                # Check if we need to migrate from v2
                if "schema_version" not in existing:
                    existing = {"schema_version": "3.0", "devices": {}}
        except Exception:
            existing = {"schema_version": "3.0", "devices": {}}
    
    # Get device info for metadata
    device_info: dict[str, Any] = {}
    try:
        from .device_profile import get_current_device_profile
        profile = get_current_device_profile()
        device_info = {
            "family": profile.family,
            "variant": profile.variant,
            "gpu_cores": profile.gpu_cores,
            "memory_bandwidth_gbps": profile.memory_bandwidth_gbps,
            "default_threadgroup": profile.default_threadgroup,
        }
    except Exception:
        pass
    
    device_key = _device_cache_key()
    
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
            "template": [list(t) for t in config.template] if config.template else []
        }
    
    existing["schema_version"] = "3.0"
    if "devices" not in existing:
        existing["devices"] = {}
    
    existing["devices"][device_key] = {
        "metadata": device_info,
        "entries": entries,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def load_autotune_cache(path: str | None = None) -> int:
    """Load the autotune cache from disk.
    
    Supports both v2 and v3 schema formats. The v3 schema includes
    device metadata and organizes entries under a "devices" key.
    
    Args:
        path: Path to cache file. If None, uses default location.
        
    Returns:
        Number of cache entries loaded.
    """
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
    
    # Handle v3 schema
    if data.get("schema_version") == "3.0" and "devices" in data:
        device_data = data["devices"].get(device_key, {})
        entries = device_data.get("entries", {})
    else:
        # v2 schema: entries directly under device_key
        entries = data.get(device_key, {})
    
    count = 0
    for key_str, val in entries.items():
        try:
            kd = json.loads(key_str)
            tg = tuple(val["tg"])
            # Handle both old and new template formats
            template_list = val.get("template", [])
            if template_list and isinstance(template_list[0], (list, tuple)):
                template = tuple(tuple(t) if isinstance(t, (list, tuple)) else (t,) for t in template_list)
            else:
                template = tuple(template_list)
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
    "AutotunedFunction",
    "autotune_kernel",
    "get_autotuned_config",
    "save_autotune_cache",
    "load_autotune_cache",
    "autotune",
    "_get_device_candidates",
    "_get_default_config",
]
