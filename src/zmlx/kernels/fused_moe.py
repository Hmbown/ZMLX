"""Fused MoE kernels — wrappers around MLX C++ fused ops.

This module provides ZMLX-style wrappers for fused MoE operations that
are implemented as C++ primitives in MLX (not as custom Metal kernels).

These are thin wrappers that:
- Check for op availability at import time
- Validate constraints (transpose=True, affine mode, alignment)
- Provide the ZMLX-standard calling convention
"""

from __future__ import annotations

from typing import Any

from .._compat import import_mx

# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

_HAS_GATHER_QMM_SWIGLU: bool | None = None


def has_gather_qmm_swiglu() -> bool:
    """Return True if mx.gather_qmm_swiglu is available."""
    global _HAS_GATHER_QMM_SWIGLU
    if _HAS_GATHER_QMM_SWIGLU is None:
        mx = import_mx()
        _HAS_GATHER_QMM_SWIGLU = hasattr(mx, "gather_qmm_swiglu")
    return _HAS_GATHER_QMM_SWIGLU


# ---------------------------------------------------------------------------
# Fused gather + QMM + SwiGLU
# ---------------------------------------------------------------------------


def gather_qmm_swiglu(
    x: Any,
    gate_w: Any,
    gate_scales: Any,
    gate_biases: Any,
    up_w: Any,
    up_scales: Any,
    up_biases: Any,
    *,
    lhs_indices: Any | None = None,
    rhs_indices: Any | None = None,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 4,
) -> Any:
    """Fused gather + quantized matmul + SwiGLU.

    Computes ``silu(gather_qmm(x, gate_w, ...)) * gather_qmm(x, up_w, ...)``
    in a single kernel launch, reading ``x`` only once from memory.

    This halves the memory bandwidth for the input tensor compared to the
    naive two-pass approach.

    Args:
        x: Input tensor, shape ``(..., M, K)``.
        gate_w: Quantized gate projection weights, shape ``(E, N, K_packed)``.
        gate_scales: Gate scales per group, shape ``(E, N, K // group_size)``.
        gate_biases: Gate biases per group, shape ``(E, N, K // group_size)``.
        up_w: Quantized up projection weights (same shape as gate_w).
        up_scales: Up scales per group (same shape as gate_scales).
        up_biases: Up biases per group (same shape as gate_biases).
        lhs_indices: Optional integer indices for ``x`` batch dims.
        rhs_indices: Optional integer indices for weight batch dims.
        transpose: Must be True (only supported mode). Default: True.
        group_size: Quantization group size (32, 64, or 128). Default: 64.
        bits: Quantization bits (4 or 8). Default: 4.

    Returns:
        Fused SwiGLU output with shape determined by gather indices and N.

    Raises:
        RuntimeError: If mx.gather_qmm_swiglu is not available.

    Constraints:
        - Only ``transpose=True`` and ``mode='affine'`` are supported.
        - ``N`` must be divisible by 8.
        - ``K`` must be divisible by 512.
        - Gate and up weights must have identical shapes/strides.
        - Forward-only (no gradient support).
    """
    mx = import_mx()

    if not has_gather_qmm_swiglu():
        raise RuntimeError(
            "mx.gather_qmm_swiglu is not available. "
            "Requires MLX >= 0.30.4.dev with fused QMM+SwiGLU support."
        )

    return mx.gather_qmm_swiglu(
        x,
        gate_w, gate_scales, gate_biases,
        up_w, up_scales, up_biases,
        lhs_indices=lhs_indices,
        rhs_indices=rhs_indices,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
    )


__all__ = [
    "gather_qmm_swiglu",
    "has_gather_qmm_swiglu",
]
