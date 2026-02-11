"""MLX and Mock backends for the evaluation harness.

Adapted from DataFoundry's harness/backend.py.  The ``maybe_quantize``
helper is inlined here (instead of importing from datafoundry.ops.common)
to keep the foundry self-contained.

Both backends satisfy the ``Backend`` protocol in ``taxonomy.py``.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Inlined maybe_quantize (was: from datafoundry.ops.common import maybe_quantize)
# ---------------------------------------------------------------------------


def _quantize_to_bfloat16(x: np.ndarray) -> np.ndarray:
    """Simulate bfloat16 by truncating the lower 16 bits of float32 mantissa."""
    x32 = x.astype(np.float32, copy=False)
    u = x32.view(np.uint32)
    u_trunc = u & np.uint32(0xFFFF0000)
    return u_trunc.view(np.float32)


def maybe_quantize(x: np.ndarray, dtype: str) -> np.ndarray:
    """Cast *x* to the precision level of *dtype*.

    * ``float16`` / ``float32`` -- straightforward ``astype``.
    * ``bfloat16`` -- numpy has no native bf16 so we truncate float32 mantissa
      bits to simulate the reduced precision.
    """
    if dtype == "float16":
        return x.astype(np.float16)
    if dtype == "float32":
        return x.astype(np.float32)
    if dtype == "bfloat16":
        return _quantize_to_bfloat16(x.astype(np.float32))
    raise ValueError(f"unsupported dtype: {dtype}")


# ---------------------------------------------------------------------------
# MLXBackend -- real Metal GPU execution
# ---------------------------------------------------------------------------


class MLXBackend:
    """Backend that delegates to ``mlx.core`` for real GPU execution."""

    name = "mlx"

    def __init__(self) -> None:
        try:
            import mlx.core as mx  # noqa: F811
            self._mx = mx
        except Exception as exc:
            self._mx = None
            self._import_error = exc

    def is_available(self) -> bool:
        return self._mx is not None

    def mlx_version(self) -> str | None:
        if self._mx is None:
            return None
        try:
            import mlx
            return getattr(mlx, "__version__", None)
        except Exception:
            return None

    def device_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {}
        if self._mx is None:
            return info
        try:
            from mlx.core import metal
            di = metal.device_info()
            info.update(di)
        except Exception:
            pass
        return info

    def array(self, np_array: Any, dtype: str) -> Any:
        assert self._mx is not None, "MLX is not available"
        mx = self._mx
        if dtype == "float16":
            return mx.array(np_array.astype(np.float16))
        if dtype == "float32":
            return mx.array(np_array.astype(np.float32))
        if dtype == "bfloat16":
            return mx.array(np_array.astype(np.float32)).astype(mx.bfloat16)
        raise ValueError(f"unsupported dtype: {dtype}")

    def to_numpy(self, arr: Any) -> Any:
        assert self._mx is not None, "MLX is not available"
        return np.array(arr)

    def eval(self, arr: Any) -> None:
        assert self._mx is not None, "MLX is not available"
        self._mx.eval(arr)

    def synchronize(self) -> None:
        assert self._mx is not None, "MLX is not available"
        try:
            self._mx.synchronize()
        except Exception:
            # Older MLX versions may lack synchronize(); eval acts as sync.
            pass

    def metal_kernel(
        self,
        name: str,
        input_names: list[str],
        output_names: list[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
        atomic_outputs: bool = False,
    ) -> Any:
        assert self._mx is not None, "MLX is not available"
        mx = self._mx
        return mx.fast.metal_kernel(
            name=name,
            input_names=input_names,
            output_names=output_names,
            source=source,
            header=header,
            ensure_row_contiguous=ensure_row_contiguous,
            atomic_outputs=atomic_outputs,
        )


# ---------------------------------------------------------------------------
# MockBackend -- no GPU required; for CI and testing
# ---------------------------------------------------------------------------


@dataclass
class MockKernel:
    """Placeholder kernel object returned by ``MockBackend.metal_kernel``.

    Calling it raises ``RuntimeError`` -- the harness is expected to use
    the reference implementation instead of executing mock kernels.  The
    one exception is compile-error simulation: if the source contains
    ``THIS_WILL_NOT_COMPILE``, ``metal_kernel`` raises at creation time.
    """

    name: str
    source: str
    header: str

    def __call__(
        self,
        *,
        inputs: list[Any],
        template: list[Any],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: list[tuple[int, ...]],
        output_dtypes: list[Any],
        verbose: bool = False,
    ) -> Any:
        if "THIS_WILL_NOT_COMPILE" in self.source or "THIS_WILL_NOT_COMPILE" in self.header:
            raise RuntimeError(
                "Mock compile error: injected syntax error (THIS_WILL_NOT_COMPILE)."
            )
        raise RuntimeError("MockKernel should not be executed directly.")


class MockBackend:
    """In-process mock backend for environments without Apple Silicon / MLX."""

    name = "mock"

    def is_available(self) -> bool:
        return True

    def mlx_version(self) -> str | None:
        return None

    def device_info(self) -> dict[str, Any]:
        return {
            "chip": platform.machine(),
            "gpu": "mock",
            "mem_gb": None,
        }

    def array(self, np_array: Any, dtype: str) -> Any:
        return maybe_quantize(np_array, dtype)

    def to_numpy(self, arr: Any) -> Any:
        return np.array(arr)

    def eval(self, arr: Any) -> None:
        return

    def synchronize(self) -> None:
        return

    def metal_kernel(
        self,
        name: str,
        input_names: list[str],
        output_names: list[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
        atomic_outputs: bool = False,
    ) -> Any:
        if "THIS_WILL_NOT_COMPILE" in source or "THIS_WILL_NOT_COMPILE" in header:
            raise RuntimeError(
                "Mock compile error: injected syntax error (THIS_WILL_NOT_COMPILE)."
            )
        return MockKernel(name=name, source=source, header=header)
