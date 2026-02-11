"""Compile a Metal kernel via the backend, with caching.

Adapted from DataFoundry's harness/compile.py.  Wraps the backend's
``metal_kernel()`` call with ``CompileCache`` lookup/insert and timing.
"""
from __future__ import annotations

import time
from typing import Any

from ..taxonomy import BuildResult
from .cache import CompileCache


def compile_kernel(
    *,
    backend: Any,
    cache: CompileCache,
    cache_key: str,
    name: str,
    input_names: list[str],
    output_names: list[str],
    source: str,
    header: str,
    ensure_row_contiguous: bool,
    atomic_outputs: bool = False,
) -> tuple[Any, BuildResult]:
    """Compile (or cache-hit) a Metal kernel and return ``(kernel_obj, BuildResult)``.

    On compile failure ``kernel_obj`` is ``None`` and ``BuildResult.ok`` is
    ``False`` with error details populated.
    """
    # Cache hit?
    ent = cache.get(cache_key)
    if ent is not None:
        if ent.ok:
            return ent.kernel, BuildResult(ok=True, ms=0.0, cache_hit=True)
        return None, BuildResult(
            ok=False,
            ms=0.0,
            error_type="compile_error",
            error_summary=str(ent.error)[:200],
            error_log=str(ent.error)[:2000],
            cache_hit=True,
        )

    t0 = time.perf_counter()
    try:
        kernel = backend.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
            ensure_row_contiguous=ensure_row_contiguous,
            atomic_outputs=atomic_outputs,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        cache.put_ok(cache_key, kernel)
        return kernel, BuildResult(ok=True, ms=ms, cache_hit=False)
    except Exception as exc:
        ms = (time.perf_counter() - t0) * 1000.0
        cache.put_err(cache_key, exc)
        return None, BuildResult(
            ok=False,
            ms=ms,
            error_type="compile_error",
            error_summary=str(exc).split("\n", 1)[0][:200],
            error_log=str(exc)[:2000],
            cache_hit=False,
        )
