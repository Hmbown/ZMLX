"""LayerNorm pattern: replace nn.LayerNorm with ZMLX fused kernel."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .._registry import register
from .._types import PatchConfig


class _LayerNormPattern:
    @property
    def name(self) -> str:
        return "layernorm"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return isinstance(module, nn.LayerNorm)

    def apply(self, module: Any, config: PatchConfig) -> Any:
        from .._modules import ZMLXLayerNorm

        cd = getattr(mx, config.compute_dtype) if isinstance(config.compute_dtype, str) else config.compute_dtype
        dims = int(module.weight.shape[0])
        eps = getattr(module, "eps", 1e-5)
        affine = hasattr(module, "weight") and hasattr(module, "bias")

        replacement = ZMLXLayerNorm(
            dims=dims,
            eps=eps,
            affine=affine,
            threadgroup=config.threadgroup,
            compute_dtype=cd,
        )
        if affine:
            replacement.weight = module.weight
            replacement.bias = module.bias
        return replacement


register(_LayerNormPattern())
