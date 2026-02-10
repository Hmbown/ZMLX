"""Auto-generated kernel from ZMLX Discover.

Target: glm_fused_swiglu
Speedup: 1.37x
Device: Apple M4
Session: 36ad2a16e0914ca9
"""

from __future__ import annotations

from functools import cache
from typing import Any

from zmlx.metal import kernel as metal_kernel
from zmlx.msl import DEFAULT_HEADER


@cache
def _discovered_kernel() -> Any:
    """Build the discovered kernel."""
    source = """constexpr uint N = 1536;
uint idx = thread_position_in_grid.x * 2;
if (idx >= N) return;

// Load both elements
float g0 = (float)gate[idx];
float g1 = (float)gate[idx + 1];
float u0 = (float)up[idx];
float u1 = (float)up[idx + 1];

// Interleaved computation for ILP
float neg_g0 = -g0;
float neg_g1 = -g1;
float exp_g0 = metal::exp(neg_g0);
float exp_g1 = metal::exp(neg_g1);
float sig0 = 1.0f / (1.0f + exp_g0);
float sig1 = 1.0f / (1.0f + exp_g1);
float r0 = g0 * sig0 * u0;
float r1 = g1 * sig1 * u1;

// Store
out[idx] = (T)r0;
out[idx + 1] = (T)r1;"""

    return metal_kernel(
        name="kk_discovered_glm_fused_swiglu",
        input_names=['gate', 'up'],
        output_names=['out'],
        source=source,
        header=DEFAULT_HEADER,
        cache=True,
    )
