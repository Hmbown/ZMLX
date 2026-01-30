"""RMSNorm pattern: replace nn.RMSNorm with ZMLX fused kernel."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .._registry import register
from .._types import PatchConfig


class _RMSNormPattern:
    @property
    def name(self) -> str:
        return "rmsnorm"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        return isinstance(module, nn.RMSNorm)

    def apply(self, module: Any, config: PatchConfig) -> Any:
        from .._modules import ZMLXRMSNorm

        cd = getattr(mx, config.compute_dtype) if isinstance(config.compute_dtype, str) else config.compute_dtype
        dims = int(module.weight.shape[0])
        eps = getattr(module, "eps", 1e-6)

        replacement = ZMLXRMSNorm(
            dims=dims,
            eps=eps,
            threadgroup=config.threadgroup,
            compute_dtype=cd,
        )
        replacement.weight = module.weight
        return replacement


register(_RMSNormPattern())
