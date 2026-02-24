from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..taxonomy import OpSpec, Tolerances

# ---------------------------------------------------------------------------
# Shape ladders (canonical sizes for LLM inference workloads)
# ---------------------------------------------------------------------------

TOKENS_LADDER: list[int] = [16, 32, 64, 128, 256, 512, 1024, 2048]
HIDDEN_LADDER: list[int] = [512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
N_EXPERTS_LADDER: list[int] = [8, 16, 32, 64]
MOE_K_LADDER: list[int] = [2, 4]


# ---------------------------------------------------------------------------
# Quantization helpers (from datafoundry common.py)
# ---------------------------------------------------------------------------

DEFAULT_TOLERANCES: dict[str, Tolerances] = {
    "float32": Tolerances(max_abs=5e-5, max_rel=5e-5),
    "float16": Tolerances(max_abs=5e-3, max_rel=5e-3),
    "bfloat16": Tolerances(max_abs=1e-2, max_rel=1e-2),
}


def np_dtype(dtype: str):
    if dtype == "float16":
        return np.float16
    if dtype == "float32":
        return np.float32
    if dtype == "bfloat16":
        # numpy lacks native bfloat16; store as float32 and truncate when needed
        return np.float32
    raise ValueError(f"unknown dtype {dtype}")


def quantize_to_bfloat16(x: np.ndarray) -> np.ndarray:
    """Simulate bf16 by truncating lower 16 bits of float32 mantissa."""
    x32 = x.astype(np.float32, copy=False)
    u = x32.view(np.uint32)
    u_trunc = (u & np.uint32(0xFFFF0000))
    return u_trunc.view(np.float32)


def maybe_quantize(x: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float16":
        return x.astype(np.float16)
    if dtype == "float32":
        return x.astype(np.float32)
    if dtype == "bfloat16":
        return quantize_to_bfloat16(x.astype(np.float32))
    raise ValueError(dtype)


# ---------------------------------------------------------------------------
# Numpy I/O helpers
# ---------------------------------------------------------------------------

def randn_np(shape: tuple[int, ...], *, dtype: str, seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    x = rng.standard_normal(size=shape).astype(np.float32) * scale
    return maybe_quantize(x, dtype)


def randint_np(low: int, high: int, shape: tuple[int, ...], *, dtype: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    x = rng.integers(low, high, size=shape)
    if dtype == "int8":
        return x.astype(np.int8)
    if dtype == "int32":
        return x.astype(np.int32)
    return x


def maybe_to_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def max_abs_err(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# ---------------------------------------------------------------------------
# KernelOp ABC
# ---------------------------------------------------------------------------

class KernelOp(ABC):
    """Standard op interface for the unified foundry.

    Every op provides:
      - spec(): static metadata
      - supported_dtypes(): what dtypes this op handles
      - templates(): list of Metal template IDs (or ["ref"] for reference-only)
      - knob_space(): dict describing tunable knob ranges per template
      - sample_shape(): random shape sampling for data generation
      - generate_inputs_numpy(): produce NumPy input tensors
      - reference_numpy(): golden reference implementation in NumPy
      - validate_knobs(): check if a knob configuration is valid
      - bytes_and_flops(): throughput estimation
    """

    name: str = ""

    @abstractmethod
    def spec(self) -> OpSpec:
        ...

    def supported_dtypes(self) -> list[str]:
        return ["float16", "float32"]

    def templates(self) -> list[str]:
        return ["ref"]

    def knob_space(self, template_id: str) -> dict[str, Any]:
        return {}

    @abstractmethod
    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        ...

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        return {}

    @abstractmethod
    def generate_inputs_numpy(
        self,
        shape: dict[str, int],
        dtype: str,
        op_params: dict[str, Any],
        seed: int,
    ) -> dict[str, np.ndarray]:
        ...

    @abstractmethod
    def reference_numpy(
        self,
        inputs: dict[str, np.ndarray],
        op_params: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        ...

    def validate_knobs(
        self, template_id: str, knobs: dict[str, Any], shape: dict[str, int], dtype: str
    ) -> tuple[bool, str]:
        return True, ""

    def bytes_and_flops(self, shape: dict[str, int], dtype: str) -> tuple[int, int]:
        return 0, 0

    # ------------------------------------------------------------------
    # Bridge methods: satisfy the taxonomy.KernelOp Protocol interface
    # so that harness/evaluate.py can use ops directly.
    # ------------------------------------------------------------------

    @property
    def input_names(self) -> list[str]:
        """Buffer names from spec().inputs, e.g. 'x[tokens,hidden]' -> 'x'."""
        return [s.split("[")[0] for s in self.spec().inputs]

    @property
    def output_names(self) -> list[str]:
        """Buffer names from spec().outputs, e.g. 'y[tokens,hidden]' -> 'y'."""
        return [s.split("[")[0] for s in self.spec().outputs]

    def canonical_math(self) -> str:
        return self.spec().summary

    def correctness_constraints(self, dtype: str) -> dict[str, Any]:
        tol = DEFAULT_TOLERANCES.get(dtype, Tolerances(max_abs=1e-2, max_rel=1e-2))
        return {"max_abs": tol.max_abs, "max_rel": tol.max_rel}

    def generate_inputs_numpy_bridge(
        self,
        rng: Any,
        shape: dict[str, int],
        dtype: str,
        layout: dict[str, Any],
    ) -> dict[str, Any]:
        """Adapter for taxonomy Protocol's generate_inputs_numpy signature.

        Translates sampler shape format (batch/seq/hidden) into op-specific
        format (e.g. tokens/hidden for norms), then delegates to the
        concrete generate_inputs_numpy.
        """
        # Translate batch+seq -> tokens if the op expects tokens
        adapted_shape = dict(shape)
        if "tokens" not in adapted_shape and "batch" in adapted_shape:
            adapted_shape["tokens"] = (
                adapted_shape.get("batch", 1) * adapted_shape.get("seq", 1)
            )

        seed = int(rng.integers(0, 2**31)) if hasattr(rng, "integers") else 42
        gen_rng = np.random.default_rng(seed)
        op_params = self.sample_op_params(adapted_shape, gen_rng)
        return self.generate_inputs_numpy(adapted_shape, dtype, op_params, seed)

    def reference_numpy_bridge(
        self,
        inputs: dict[str, Any],
        dtype: str,
    ) -> Any:
        """Adapter for taxonomy Protocol's reference_numpy signature."""
        result = self.reference_numpy(inputs, {})
        # If result is a dict, return the first output array
        if isinstance(result, dict):
            out_names = self.output_names
            if out_names and out_names[0] in result:
                return result[out_names[0]]
            # Fallback: return first value
            return next(iter(result.values()))
        return result

    def reference_mlx(
        self,
        inputs: dict[str, Any],
        dtype: str,
    ) -> Any:
        """Default: no MLX reference. The harness falls back to numpy."""
        raise NotImplementedError(f"{self.name} has no MLX reference implementation")

    def output_shapes(
        self,
        inputs: dict[str, Any],
        shape: dict[str, int],
    ) -> list[tuple[int, ...]]:
        """Derive output shapes from the reference numpy computation."""
        try:
            result = self.reference_numpy(inputs, {})
            if isinstance(result, dict):
                return [np.asarray(v).shape for v in result.values()]
            return [np.asarray(result).shape]
        except Exception:
            # Fallback: assume output matches first input shape
            for v in inputs.values():
                if hasattr(v, "shape"):
                    return [v.shape]
            return [(1,)]

    def tg_mem_bytes(self, template_id: str, tg_size: int) -> int:
        """Threadgroup memory estimate. Default: 0 (no shared memory)."""
        return 0
