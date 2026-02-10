"""Auto-generated kernel from ZMLX Discover.

Target: glm_rmsnorm
Speedup: 1.16x
Device: Apple M4
Session: 9fc64556386a4503
"""

from __future__ import annotations

from functools import cache
from typing import Any

from zmlx.metal import kernel as metal_kernel
from zmlx.msl import DEFAULT_HEADER


@cache
def _discovered_kernel(eps: float) -> Any:
    """Build the discovered kernel specialized for epsilon."""
    eps_f = float(eps)
    source = """constexpr uint D = 2048;
constexpr uint TG = 256;
constexpr float EPS = __EPS__f;
constexpr uint ELEMS_PER_THREAD = D / TG;  // 8

uint gid = thread_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint row = gid / TG;
uint base = row * D;

threadgroup float final_sum[1];
float local_vals[ELEMS_PER_THREAD];
float local_weights[ELEMS_PER_THREAD];

// Level 1: Register accumulation (8 elements per thread)
float sumsq = 0.0f;
uint offset = tid * ELEMS_PER_THREAD;
for (uint k = 0; k < ELEMS_PER_THREAD; k++) {
    local_vals[k] = (float)inp[base + offset + k];
    local_weights[k] = (float)weight[offset + k];
    sumsq += local_vals[k] * local_vals[k];
}

// Level 2: SIMD reduction (32 lanes)
for (uint s = 16; s > 0; s >>= 1) {
    sumsq += simd_shuffle_down(sumsq, s);
}

// Level 3: Threadgroup reduction (8 SIMD groups)
threadgroup float simd_results[TG / 32];
uint lane = tid & 31;
uint simd_idx = tid >> 5;

if (lane == 0) {
    simd_results[simd_idx] = sumsq;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

if (tid == 0) {
    float total = 0.0f;
    for (uint i = 0; i < TG / 32; i++) {
        total += simd_results[i];
    }
    final_sum[0] = total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float inv = metal::rsqrt(final_sum[0] / (float)D + EPS);

// Direct register output
for (uint k = 0; k < ELEMS_PER_THREAD; k++) {
    out[base + offset + k] = (T)(local_vals[k] * inv * local_weights[k]);
}"""
    source = source.replace("__EPS__", f"{eps_f}")

    return metal_kernel(
        name="kk_discovered_glm_rmsnorm",
        input_names=['inp', 'weight'],
        output_names=['out'],
        source=source,
        header=DEFAULT_HEADER,
        cache=True,
    )


# Specialization constants
_D = 2048
_TG = 256


def glm_rmsnorm(x: Any, weight: Any, eps: float = 1e-6) -> Any | None:
    """Drop-in for ``norms.rmsnorm`` when D=2048 (GLM-4.7-Flash).

    Returns ``None`` when dimensions don't match so the caller can fall back.
    """

    import mlx.core as mx

    D = x.shape[-1]
    if D != _D:
        return None
    # Fidelity guard: fp16/bf16 accumulation order is not bit-identical to
    # the current rmsnorm reference. Use discovered kernel only for float32.
    if x.dtype != mx.float32 or weight.dtype != mx.float32:
        return None

    original_shape = x.shape
    rows = x.size // D
    k = _discovered_kernel(float(eps))
    out = k(
        x.reshape(-1), weight,
        template=[("T", x.dtype)],
        grid=(rows * _TG, 1, 1),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(rows * D,)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(original_shape)
