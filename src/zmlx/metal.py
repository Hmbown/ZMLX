from __future__ import annotations

import ctypes
import fnmatch
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._compat import import_mx
from .cache import GLOBAL_KERNEL_CACHE, KernelCacheKey

_ENV_DISABLE_ROW_CONTIGUOUS_KERNELS = "ZMLX_METAL_DISABLE_ROW_CONTIGUOUS_KERNELS"
_ENV_CONTIGUITY_TELEMETRY = "ZMLX_METAL_CONTIGUITY_TELEMETRY"
_ENV_CONTIGUITY_TELEMETRY_KERNELS = "ZMLX_METAL_CONTIGUITY_TELEMETRY_KERNELS"


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


try:
    _PY_CAPSULE_GET_POINTER = ctypes.pythonapi.PyCapsule_GetPointer
    _PY_CAPSULE_GET_POINTER.argtypes = [ctypes.py_object, ctypes.c_char_p]
    _PY_CAPSULE_GET_POINTER.restype = ctypes.c_void_p
except Exception:  # pragma: no cover - CPython capsule API should exist
    _PY_CAPSULE_GET_POINTER = None  # type: ignore[assignment]

try:
    _PY_CAPSULE_IS_VALID = ctypes.pythonapi.PyCapsule_IsValid
    _PY_CAPSULE_IS_VALID.argtypes = [ctypes.py_object, ctypes.c_char_p]
    _PY_CAPSULE_IS_VALID.restype = ctypes.c_int
except Exception:  # pragma: no cover - CPython capsule API should exist
    _PY_CAPSULE_IS_VALID = None  # type: ignore[assignment]


@dataclass
class KernelStats:
    """Runtime statistics for a single :class:`MetalKernel` instance.

    Attributes:
        compile_time_ms: Time spent constructing the ``mx.fast.metal_kernel``
            callable (Python-side, not Metal compilation).
        run_count: Number of times ``__call__`` has been invoked.
        total_run_time_ms: Cumulative wall-clock time of launches made with
            ``verbose=True``.  Zero when verbose mode is not used.
        contiguity_launches: Number of launches where input contiguity telemetry
            was sampled.
        contiguity_checks: Number of inputs for which row-contiguity could be
            determined.
        contiguity_unknown_inputs: Number of inputs where contiguity could not
            be determined from exposed metadata.
        non_row_contiguous_inputs: Number of sampled inputs that were detected
            as non-row-contiguous.
        launches_with_non_row_contiguous: Launch count where at least one input
            was detected as non-row-contiguous.
        copy_risk_launches: Launch count where non-row-contiguous inputs were
            seen while ``ensure_row_contiguous=True`` (these launches may pay
            implicit row-contiguity copies in MLX).
        copy_risk_inputs: Total number of non-row-contiguous inputs seen during
            ``copy_risk_launches``.
    """

    compile_time_ms: float = 0.0
    run_count: int = 0
    total_run_time_ms: float = 0.0
    contiguity_launches: int = 0
    contiguity_checks: int = 0
    contiguity_unknown_inputs: int = 0
    non_row_contiguous_inputs: int = 0
    launches_with_non_row_contiguous: int = 0
    copy_risk_launches: int = 0
    copy_risk_inputs: int = 0


def _bool_env(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _patterns_env(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _resolve_ensure_row_contiguous(name: str, requested: bool) -> bool:
    if not requested:
        return False
    disabled = _patterns_env(_ENV_DISABLE_ROW_CONTIGUOUS_KERNELS)
    if disabled and _matches_any(name, disabled):
        return False
    return True


def _contiguity_telemetry_enabled(name: str) -> bool:
    if not _bool_env(_ENV_CONTIGUITY_TELEMETRY):
        return False
    only = _patterns_env(_ENV_CONTIGUITY_TELEMETRY_KERNELS)
    if not only:
        return True
    return _matches_any(name, only)


def _is_row_contiguous(shape: Sequence[int], strides: Sequence[int] | None) -> bool:
    if strides is None:
        return True
    if len(shape) != len(strides):
        return False
    expected = 1
    for dim, stride in zip(reversed(shape), reversed(strides), strict=True):
        d = int(dim)
        s = int(stride)
        if d == 0:
            return True
        if d > 1:
            if s != expected:
                return False
            expected *= d
    return True


def _dlpack_layout(x: Any) -> tuple[tuple[int, ...], tuple[int, ...] | None] | None:
    if _PY_CAPSULE_GET_POINTER is None or _PY_CAPSULE_IS_VALID is None:
        return None
    dlpack = getattr(x, "__dlpack__", None)
    if not callable(dlpack):
        return None
    try:
        capsule = dlpack()
        if _PY_CAPSULE_IS_VALID(capsule, b"dltensor") != 1:
            return None
        ptr = _PY_CAPSULE_GET_POINTER(capsule, b"dltensor")
        if not ptr:
            return None
        managed = ctypes.cast(ptr, ctypes.POINTER(_DLManagedTensor)).contents
        ndim = int(managed.dl_tensor.ndim)
        shape = tuple(int(managed.dl_tensor.shape[i]) for i in range(ndim))
        strides_ptr = managed.dl_tensor.strides
        if bool(strides_ptr):
            strides = tuple(int(strides_ptr[i]) for i in range(ndim))
        else:
            strides = None
        return shape, strides
    except Exception:
        return None


def _attr_layout(x: Any) -> tuple[tuple[int, ...], tuple[int, ...] | None] | None:
    shape = getattr(x, "shape", None)
    strides = getattr(x, "strides", None)
    if shape is None or strides is None:
        return None
    try:
        shape_t = tuple(int(d) for d in shape)
        itemsize = int(getattr(x, "itemsize", 0))
        if itemsize <= 0 and hasattr(x, "dtype"):
            itemsize = int(getattr(x.dtype, "itemsize", 0))
        if itemsize > 0:
            strides_t = tuple(int(s) // itemsize for s in strides)
        else:
            strides_t = tuple(int(s) for s in strides)
        return shape_t, strides_t
    except Exception:
        return None


def _is_input_row_contiguous(x: Any) -> bool | None:
    layout = _dlpack_layout(x)
    if layout is None:
        layout = _attr_layout(x)
    if layout is None:
        return None
    shape, strides = layout
    return _is_row_contiguous(shape, strides)


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
    requested_ensure_row_contiguous: bool = True
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
        self._contiguity_telemetry_enabled = _contiguity_telemetry_enabled(spec.name)
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

        if self._contiguity_telemetry_enabled:
            self.stats.contiguity_launches += 1
            non_row_contiguous = 0
            for inp in inputs:
                status = _is_input_row_contiguous(inp)
                if status is None:
                    self.stats.contiguity_unknown_inputs += 1
                    continue
                self.stats.contiguity_checks += 1
                if status is False:
                    self.stats.non_row_contiguous_inputs += 1
                    non_row_contiguous += 1
            if non_row_contiguous > 0:
                self.stats.launches_with_non_row_contiguous += 1
                if self.spec.ensure_row_contiguous:
                    self.stats.copy_risk_launches += 1
                    self.stats.copy_risk_inputs += non_row_contiguous

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
    effective_ensure_row_contiguous = _resolve_ensure_row_contiguous(
        name=name,
        requested=ensure_row_contiguous,
    )
    spec = MetalKernelSpec(
        name=name,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        source=source,
        header=header,
        requested_ensure_row_contiguous=ensure_row_contiguous,
        ensure_row_contiguous=effective_ensure_row_contiguous,
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
        ensure_row_contiguous=effective_ensure_row_contiguous,
        atomic_outputs=atomic_outputs,
    )
    cached = GLOBAL_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    return GLOBAL_KERNEL_CACHE.put(key, MetalKernel(spec))  # type: ignore[no-any-return]
