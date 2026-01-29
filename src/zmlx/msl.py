"""Metal helper snippets shared across kernels.

These are *strings* intended to be passed to ``mx.fast.metal_kernel(..., header=...)``.

Design goals:
- Keep kernels small and readable
- Avoid repeating common activation formulas
- Make it easy to expand the library of fused kernels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

DEFAULT_HEADER: Final[str] = r"""
#include <metal_stdlib>
using namespace metal;

template <typename T>
inline T kk_sigmoid(T x) {
    return T(1) / (T(1) + metal::exp(-x));
}

template <typename T>
inline T kk_silu(T x) {
    return x * kk_sigmoid(x);
}

// tanh approximation form used in many transformer implementations
template <typename T>
inline T kk_gelu_tanh(T x) {
    const T k0 = T(0.7978845608028654);   // sqrt(2/pi)
    const T k1 = T(0.044715);
    T x3 = x * x * x;
    return T(0.5) * x * (T(1) + metal::tanh(k0 * (x + k1 * x3)));
}

// erf-based GELU is also common; Metal doesn't expose erf() in all profiles,
// so we default to tanh GELU. If you need erf GELU, implement a poly approximation.
"""


@dataclass(frozen=True)
class Launch2D:
    """Convenience for 2D launch configuration.

    MLX `metal_kernel` launches with `grid=(gx, gy, gz)` and `threadgroup=(tx, ty, tz)`.
    Most kernels in this repo are 1D, but 2D launches can be helpful for (row, col) indexing.
    """
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
