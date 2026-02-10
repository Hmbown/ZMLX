"""Auto-generated kernel from ZMLX Discover.

Target: glm_moe_combine
Speedup: 1.64x
Device: Apple M4
Session: 4a82a9ba06d1485b
"""

from __future__ import annotations

from functools import cache
from typing import Any

from zmlx.metal import kernel as metal_kernel
from zmlx.msl import DEFAULT_HEADER


@cache
def _discovered_kernel() -> Any:
    """Build the discovered kernel."""
    source = """
#pragma clang fp contract(off)
constexpr uint D = 2048;
constexpr uint K = 4;
uint token_idx = thread_position_in_grid.y;
uint d_idx = thread_position_in_grid.x;

if (d_idx < D) {
    // Multiply then accumulate in native T — contract(off) prevents FMA
    // contraction so products are rounded to T before addition, matching
    // MLX's (expert_outputs * weights[..., None]).sum(axis=-2) semantics.
    // Explicit (T) casts needed because bfloat16 multiply promotes to float.
    T p0 = (T)(weights[token_idx * K + 0] * expert_outputs[(token_idx * K + 0) * D + d_idx]);
    T p1 = (T)(weights[token_idx * K + 1] * expert_outputs[(token_idx * K + 1) * D + d_idx]);
    T p2 = (T)(weights[token_idx * K + 2] * expert_outputs[(token_idx * K + 2) * D + d_idx]);
    T p3 = (T)(weights[token_idx * K + 3] * expert_outputs[(token_idx * K + 3) * D + d_idx]);
    // Left-to-right accumulation matching MLX's .sum(axis=-2)
    T acc = (T)(p0 + p1);
    acc = (T)(acc + p2);
    acc = (T)(acc + p3);
    out[token_idx * D + d_idx] = acc;
}"""

    return metal_kernel(
        name="kk_discovered_glm_moe_combine",
        input_names=['expert_outputs', 'weights'],
        output_names=['out'],
        source=source,
        header=DEFAULT_HEADER,
        cache=True,
    )


# Specialization constants
_D = 2048
_K = 4


@cache
def _promoted_dtype(lhs_dtype: Any, rhs_dtype: Any) -> Any:
    """Infer MLX promotion result for binary multiply."""
    import mlx.core as mx

    return (mx.array([0], dtype=lhs_dtype) * mx.array([0], dtype=rhs_dtype)).dtype


def glm_moe_combine(expert_outputs: Any, weights: Any) -> Any | None:
    """Drop-in for ``moe.moe_combine`` when D=2048, K=4 (GLM-4.7-Flash).

    Returns ``None`` when dimensions don't match so the caller can fall back.

    When expert_outputs and weights have different dtypes (e.g. bfloat16
    experts + float32 weights), promotes both to the wider dtype before
    the kernel and casts the result back — matching MLX's broadcast-multiply
    promotion semantics.
    """
    K = weights.shape[-1]
    D = expert_outputs.shape[-1]
    if D != _D or K != _K:
        return None

    # Match MLX's dtype promotion: when dtypes differ, promote to the wider
    # type for the compute, then cast back to the original expert dtype.
    out_dtype = expert_outputs.dtype
    if expert_outputs.dtype != weights.dtype:
        compute_dtype = _promoted_dtype(expert_outputs.dtype, weights.dtype)
        expert_outputs = expert_outputs.astype(compute_dtype)
        weights = weights.astype(compute_dtype)
    else:
        compute_dtype = expert_outputs.dtype

    original_shape = weights.shape[:-1]
    expert_outputs_flat = expert_outputs.reshape(-1, K, D)
    weights_flat = weights.reshape(-1, K)
    B = weights_flat.shape[0]

    k = _discovered_kernel()
    out = k(
        expert_outputs_flat, weights_flat,
        template=[("T", compute_dtype)],
        grid=(D, B, 1),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, D)],
        output_dtypes=[compute_dtype],
    )[0]
    out = out.reshape((*original_shape, D))
    if compute_dtype != out_dtype:
        out = out.astype(out_dtype)
    return out
