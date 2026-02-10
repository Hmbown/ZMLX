"""Auto-generated kernel from ZMLX Discover.

Target: glm_fused_swiglu
Speedup: 1.20x
Device: Apple M4
Session: 0cf2e9c745ef4bf4
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from zmlx.metal import kernel as metal_kernel
from zmlx.msl import DEFAULT_HEADER


@lru_cache(maxsize=8)
def _discovered_kernel(n: int) -> Any:
    """Build the discovered kernel for *n* total elements.

    The kernel source is parameterized by N (total element count, not just D).
    MLX caches compiled Metal programs by source-string hash, so repeated calls
    with the same *n* reuse the compiled binary.
    """
    source = f"""constexpr uint N = {n};
uint base_idx = thread_position_in_grid.x * 16;

// Prefetch first batch
T g0 = (base_idx < N) ? gate[base_idx] : T(0);
T u0 = (base_idx < N) ? up[base_idx] : T(0);

#pragma unroll
for (uint i = 0; i < 16; i++) {{
    uint idx = base_idx + i;
    if (idx >= N) return;

    // Prefetch next iteration
    uint next_idx = idx + 1;
    T g_next = (next_idx < N && i < 15) ? gate[next_idx] : T(0);
    T u_next = (next_idx < N && i < 15) ? up[next_idx] : T(0);

    // Compute: silu(gate) * up — matches MLX's native SwitchGLU precision
    out[idx] = kk_silu(g0) * u0;

    // Swap buffers
    g0 = g_next;
    u0 = u_next;
}}"""

    return metal_kernel(
        name="kk_discovered_glm_fused_swiglu",
        input_names=['gate', 'up'],
        output_names=['out'],
        source=source,
        header=DEFAULT_HEADER,
        cache=True,
    )


# Specialization constant
_D = 1536


def glm_fused_swiglu(gate: Any, up: Any) -> Any | None:
    """Drop-in for ``transformer.swiglu2`` when D=1536 (GLM-4.7-Flash).

    Returns ``None`` when dimensions don't match so the caller can fall back.
    """

    D = gate.shape[-1]
    if D != _D:
        return None

    n = gate.size
    k = _discovered_kernel(n)
    # Kernel processes 16 elements per thread
    grid_x = (n + 15) // 16
    out = k(
        gate.reshape(-1), up.reshape(-1),
        template=[("T", gate.dtype)],
        grid=(grid_x, 1, 1),
        threadgroup=(min(grid_x, 256), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[gate.dtype],
    )[0]
    return out.reshape(gate.shape)
