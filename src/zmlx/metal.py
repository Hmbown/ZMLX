from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._compat import import_mx
from .cache import GLOBAL_KERNEL_CACHE, KernelCacheKey


@dataclass
class KernelStats:
    """Runtime statistics for a single :class:`MetalKernel` instance.

    Attributes:
        compile_time_ms: Time spent constructing the ``mx.fast.metal_kernel``
            callable (Python-side, not Metal compilation).
        run_count: Number of times ``__call__`` has been invoked.
        total_run_time_ms: Cumulative wall-clock time of launches made with
            ``verbose=True``.  Zero when verbose mode is not used.
    """

    compile_time_ms: float = 0.0
    run_count: int = 0
    total_run_time_ms: float = 0.0


def _prod(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _default_threadgroup_x(n_threads: int) -> int:
    # Prefer a warp-ish multiple; Metal SIMD-group sizes vary, but 32 is a safe default.
    # We choose the largest candidate <= n_threads.
    for c in (512, 256, 128, 64, 32, 16, 8, 4, 2, 1):
        if c <= max(1, n_threads):
            return c
    return 1


@dataclass(frozen=True)
class MetalKernelSpec:
    name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    source: str
    header: str = ""
    ensure_row_contiguous: bool = True
    atomic_outputs: bool = False


class MetalKernel:
    """A small wrapper around `mx.fast.metal_kernel`.

    Provides:
    - defaults for grid/threadgroup for elementwise-style kernels
    - a friendlier `__call__` signature
    - optional in-process caching at construction time (via `metal.kernel()` factory)
    """

    def __init__(self, spec: MetalKernelSpec):
        self.spec = spec
        self.stats = KernelStats()
        mx = import_mx()
        
        t0 = time.perf_counter_ns()
        self._kernel = mx.fast.metal_kernel(
            name=spec.name,
            input_names=list(spec.input_names),
            output_names=list(spec.output_names),
            source=spec.source,
            header=spec.header or "",
            ensure_row_contiguous=spec.ensure_row_contiguous,
            atomic_outputs=spec.atomic_outputs,
        )
        self.stats.compile_time_ms = (time.perf_counter_ns() - t0) / 1e6

    def __call__(
        self,
        *inputs: Any,
        template: list[tuple[str, Any]] | None = None,
        grid: tuple[int, int, int] | None = None,
        threadgroup: tuple[int, int, int] | None = None,
        output_shapes: list[Sequence[int]] | None = None,
        output_dtypes: list[Any] | None = None,
        init_value: Any | None = None,
        verbose: bool = False,
    ) -> list[Any]:
        """Launch the kernel.

        Args:
            *inputs: Input arrays for the kernel.
            template: Metal template specializations (e.g. ``[("T", mx.float32)]``).
            grid: Metal grid dimensions ``(x, y, z)``. Defaults to elementwise sizing.
            threadgroup: Metal threadgroup dimensions ``(x, y, z)``.
            output_shapes: Shape for each output buffer. Defaults to ``inputs[0].shape``.
            output_dtypes: Dtype for each output buffer. Defaults to ``inputs[0].dtype``.
            init_value: Optional initialization value for outputs (required for atomics).
            verbose: If True, times the launch and accumulates ``KernelStats``.

        Returns:
            A list of MLX arrays (one per output).

        Notes:
            - By default, assumes an elementwise pattern with one output matching
              ``inputs[0]``.
            - ``grid`` follows Metal's ``dispatchThreads`` convention.
        """
        mx = import_mx()

        if len(inputs) != len(self.spec.input_names):
            raise ValueError(
                f"{self.spec.name}: expected {len(self.spec.input_names)} inputs, got {len(inputs)}"
            )

        if output_shapes is None:
            # Default: each output matches the first input
            if len(inputs) == 0:
                raise ValueError(f"{self.spec.name}: cannot infer output shape with zero inputs")
            shape0 = tuple(int(d) for d in inputs[0].shape)
            output_shapes = [shape0 for _ in self.spec.output_names]

        if output_dtypes is None:
            if len(inputs) == 0:
                raise ValueError(f"{self.spec.name}: cannot infer output dtype with zero inputs")
            dt0 = inputs[0].dtype
            output_dtypes = [dt0 for _ in self.spec.output_names]

        if grid is None:
            # Elementwise default: 1 thread per output element (assumes outputs[0] is representative).
            n = _prod(output_shapes[0])
            grid = (n, 1, 1)

        if threadgroup is None:
            tgx = min(_default_threadgroup_x(grid[0]), grid[0]) if grid[0] > 0 else 1
            threadgroup = (tgx, 1, 1)

        kwargs: dict[str, Any] = {
            "inputs": list(inputs),
            "template": template or [],
            "grid": grid,
            "threadgroup": threadgroup,
            "output_shapes": [tuple(int(d) for d in s) for s in output_shapes],
            "output_dtypes": output_dtypes,
        }
        if init_value is not None:
            kwargs["init_value"] = init_value
        if verbose:
            kwargs["verbose"] = True

        self.stats.run_count += 1
        
        if verbose:
            t0 = time.perf_counter_ns()
            outputs = self._kernel(**kwargs)
            mx.eval(*outputs)
            # Try to sync if possible for accurate timing
            sync = getattr(mx, "synchronize", None)
            if callable(sync):
                sync()
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            self.stats.total_run_time_ms += elapsed_ms
        else:
            outputs = self._kernel(**kwargs)
            
        return list(outputs)


def kernel(
    *,
    name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    source: str,
    header: str = "",
    ensure_row_contiguous: bool = True,
    atomic_outputs: bool = False,
    cache: bool = True,
) -> MetalKernel:
    """Build (or retrieve) a cached :class:`MetalKernel`.

    Args:
        name: Kernel name used for caching and debugging.
        input_names: Ordered input buffer names.
        output_names: Ordered output buffer names.
        source: Metal source string.
        header: Optional Metal header (e.g. helper functions).
        ensure_row_contiguous: If True, inputs are forced row-contiguous.
        atomic_outputs: If True, outputs are allocated as atomics.
        cache: If True, reuse an existing kernel from the global cache.

    Returns:
        A :class:`MetalKernel` instance.
    """
    spec = MetalKernelSpec(
        name=name,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        source=source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        atomic_outputs=atomic_outputs,
    )

    if not cache:
        return MetalKernel(spec)

    key = KernelCacheKey.from_parts(
        name=name,
        input_names=input_names,
        output_names=output_names,
        source=source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        atomic_outputs=atomic_outputs,
    )
    cached = GLOBAL_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    return GLOBAL_KERNEL_CACHE.put(key, MetalKernel(spec))  # type: ignore[no-any-return]
