"""Foundation types for the ZMLX foundry module.

Adapted from DataFoundry's types.py.  All types are plain dataclasses
(no framework dependency) so they serialize cleanly to NDJSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Kernel class taxonomy (9 classes for LLM inference on-device)
# ---------------------------------------------------------------------------


class KernelClass(str, Enum):
    ELEMENTWISE = "A"       # Elementwise / broadcast
    REDUCTION = "B"         # Reductions (sum/max/norm)
    FUSED_POINTWISE = "C"   # Fused pointwise (bias+act, gated MLP)
    GEMM = "D"              # Matmul / GEMM-like (incl. grouped/batched)
    ATTENTION = "E"         # Attention primitives (softmax, RoPE)
    QUANT = "F"             # Quantization primitives (dequant/quant)
    DATA_MOVEMENT = "G"     # Data movement / re-layout (gather/scatter)
    MOE = "H"               # MoE routing + dispatch
    KV_CACHE = "I"          # KV cache management


# ---------------------------------------------------------------------------
# Op specification
# ---------------------------------------------------------------------------


@dataclass
class OpSpec:
    """Compact, serializable spec for an op (used in datasets & reports)."""

    name: str
    kernel_class: KernelClass
    summary: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    op_params_schema: dict[str, Any] = field(default_factory=dict)
    shape_hints: dict[str, Any] = field(default_factory=dict)
    dtype_hints: list[str] = field(default_factory=list)
    has_reference: bool = True
    templates: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Shape = dict[str, int]
Layout = dict[str, Any]

# ---------------------------------------------------------------------------
# Template metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelTemplate:
    """Pointer to a .metal template on disk."""

    op: str
    template_id: str
    path: str  # relative path under templates/<op>/


# ---------------------------------------------------------------------------
# Candidate (the unit of work for the harness)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelCandidate:
    """Fully-specified kernel attempt: op + dtype + shape + template + knobs."""

    op: str
    dtype: str
    shape: Shape
    layout: Layout
    template_id: str
    knobs: dict[str, Any]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    ok: bool
    ms: float
    error_type: str | None = None  # 'compile_error' | 'api_error' | None
    error_summary: str = ""
    error_log: str = ""
    cache_hit: bool = False


@dataclass
class CorrectnessResult:
    ok: bool
    max_abs_err: float | None
    max_rel_err: float | None
    ulp: float | None
    n_tests: int
    seed: int
    fail_reason: str | None = None


@dataclass
class BenchResult:
    ok: bool
    warmup: int
    repeats: int
    latency_ms: dict[str, float | None]  # p50/p90/min/mean
    throughput_gbs: float | None
    throughput_gflops: float | None
    timeout: bool = False


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerances:
    max_abs: float
    max_rel: float


def tolerances(dtype: str) -> Tolerances:
    """Conservative defaults for correctness gating."""
    if dtype == "float32":
        return Tolerances(max_abs=5e-5, max_rel=5e-5)
    if dtype == "float16":
        return Tolerances(max_abs=5e-3, max_rel=5e-3)
    if dtype == "bfloat16":
        return Tolerances(max_abs=1e-2, max_rel=1e-2)
    raise ValueError(f"unknown dtype: {dtype}")


# ---------------------------------------------------------------------------
# Backend protocol -- structural typing so MLXBackend / MockBackend both match
# ---------------------------------------------------------------------------


class Backend(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def device_info(self) -> dict[str, Any]: ...

    def mlx_version(self) -> str | None: ...

    def array(self, np_array: Any, dtype: str) -> Any: ...

    def to_numpy(self, arr: Any) -> Any: ...

    def eval(self, arr: Any) -> None: ...

    def synchronize(self) -> None: ...

    def metal_kernel(
        self,
        name: str,
        input_names: list[str],
        output_names: list[str],
        source: str,
        header: str = "",
        ensure_row_contiguous: bool = True,
        atomic_outputs: bool = False,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# KernelOp protocol -- the GENERIC abstraction for op-specific logic
# ---------------------------------------------------------------------------


class KernelOp(Protocol):
    """Protocol that every op module must satisfy.

    Instead of DataFoundry's ``if op == "rmsnorm" ... elif op == "swiglu"``
    switch, the harness accepts a KernelOp instance and delegates all
    op-specific behaviour to it.
    """

    @property
    def name(self) -> str:
        """Canonical op name, e.g. ``"rmsnorm"``, ``"swiglu"``."""
        ...

    @property
    def input_names(self) -> list[str]:
        """Metal kernel input buffer names, e.g. ``["x", "w", "eps"]``."""
        ...

    @property
    def output_names(self) -> list[str]:
        """Metal kernel output buffer names, e.g. ``["y"]``."""
        ...

    def canonical_math(self) -> str:
        """Human-readable math specification string."""
        ...

    def correctness_constraints(self, dtype: str) -> dict[str, Any]:
        """Tolerance metadata dict for this dtype."""
        ...

    def bytes_and_flops(self, shape: Shape, dtype: str) -> tuple[int, int]:
        """Estimated total bytes transferred and FLOPs for throughput calc."""
        ...

    def generate_inputs_numpy(
        self,
        rng: Any,
        shape: Shape,
        dtype: str,
        layout: Layout,
    ) -> dict[str, Any]:
        """Create numpy input arrays for a single test case."""
        ...

    def reference_numpy(
        self,
        inputs: dict[str, Any],
        dtype: str,
    ) -> Any:
        """Gold-standard numpy reference implementation."""
        ...

    def reference_mlx(
        self,
        inputs: dict[str, Any],
        dtype: str,
    ) -> Any:
        """MLX reference implementation (for on-device comparison)."""
        ...

    def output_shapes(
        self,
        inputs: dict[str, Any],
        shape: Shape,
    ) -> list[tuple[int, ...]]:
        """Expected output shapes for the kernel call."""
        ...

    def tg_mem_bytes(self, template_id: str, tg_size: int) -> int:
        """Threadgroup memory in bytes for a given template + tg_size."""
        ...
