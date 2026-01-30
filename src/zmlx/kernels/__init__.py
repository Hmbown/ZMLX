"""Kernel catalog for ZMLX.

These are higher-level, ready-to-use ops built on top of:
- `mx.fast.metal_kernel` (JIT Metal kernels)
- `mx.custom_function` (custom VJP/JVP hooks)

Most functions here compile a specialized kernel on first use for a given shape
(e.g. last-dimension `D`) and threadgroup choice.
"""

from . import (
    activations,
    attention,
    bits,
    fused,
    image,
    indexing,
    linear,
    loss,
    moe,
    norms,
    optimizers,
    quant,
    reductions,
    rope,
    scan,
    softmax,
    transformer,
    vlsp,
)

__all__ = [
    "activations",
    "fused",
    "norms",
    "rope",
    "softmax",
    "transformer",
    "reductions",
    "attention",
    "scan",
    "quant",
    "linear",
    "moe",
    "optimizers",
    "bits",
    "image",
    "indexing",
    "loss",
    "vlsp",
]
