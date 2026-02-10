"""Auto-generated kernel from ZMLX Discover.

Target: glm_topk_gating
Speedup: 5.79x
Device: Apple M4
Session: 9ab4ff964c9c4955
"""

from __future__ import annotations

from functools import cache
from typing import Any

from zmlx.metal import kernel as metal_kernel
from zmlx.msl import DEFAULT_HEADER


@cache
def _discovered_kernel() -> Any:
    """Build the discovered kernel."""
    source = """constexpr uint D = 64;
constexpr uint K = 4;
constexpr uint SG = 32;

uint gid = thread_position_in_grid.x;
uint tid = thread_position_in_threadgroup.x;
uint row = gid / SG;
uint base = row * D;

float v0 = (float)inp[base + tid];
float v1 = (float)inp[base + tid + 32];
float v = metal::max(v0, v1);
uint local_idx = (v0 >= v1) ? tid : (tid + 32);

thread float topk_vals[K];
thread uint topk_idx[K];

float cur = v;
uint cur_idx = local_idx;

for (uint i = 0; i < K; ++i) {
    float cur_max = simd_max(cur);
    uint candidate_idx = (cur == cur_max) ? cur_idx : 0;
    uint winner_idx = simd_max(candidate_idx);
    
    topk_vals[i] = simd_broadcast(cur_max, 0);
    topk_idx[i] = simd_broadcast(winner_idx, 0);
    
    if (cur_idx == winner_idx) {
        cur = -INFINITY;
    }
}

// Parallel softmax: each thread handles one K value
float m = topk_vals[0];
float local_exp = (tid < K) ? metal::exp(topk_vals[tid] - m) : 0.0f;
float sum_exp = simd_sum(local_exp);
float inv = 1.0f / sum_exp;

// Thread 0-3 write results
if (tid < K) {
    uint out_base = row * K;
    weights[out_base + tid] = (T)(local_exp * inv);
    indices[out_base + tid] = topk_idx[tid];
}"""

    return metal_kernel(
        name="kk_discovered_glm_topk_gating",
        input_names=['inp'],
        output_names=['weights', 'indices'],
        source=source,
        header=DEFAULT_HEADER,
        cache=True,
    )


# Specialization constants
_D = 64
_K = 4
_SG = 32


def glm_topk_gating(logits: Any, k: int = 4) -> tuple[Any, Any] | None:
    """Drop-in for top-K gating when D=64, K=4 (GLM-4.7-Flash).

    Returns ``(indices, weights)`` or ``None`` when dimensions don't match.
    """
    import mlx.core as mx

    D = logits.shape[-1]
    if D != _D or k != _K:
        return None

    original_shape = logits.shape[:-1]
    logits_flat = logits.reshape(-1, D)
    rows = logits_flat.shape[0]

    kern = _discovered_kernel()
    weights, indices = kern(
        logits_flat,
        template=[("T", logits.dtype)],
        grid=(rows * _SG, 1, 1),
        threadgroup=(_SG, 1, 1),
        output_shapes=[(rows, _K), (rows, _K)],
        output_dtypes=[logits.dtype, mx.uint32],
    )
    return (
        indices.reshape((*original_shape, _K)),
        weights.reshape((*original_shape, _K)),
    )
