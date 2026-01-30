"""Softmax pattern: replace manual softmax calls with ZMLX fused kernel.

Targets attention modules that expose a callable ``softmax`` attribute and
appear to implement manual attention (not MX SDPA). This is a conservative
heuristic to avoid touching non-attention softmax usage (e.g., MoE gating).
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .._registry import register
from .._types import PatchConfig


class _SoftmaxPattern:
    @property
    def name(self) -> str:
        return "softmax"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False

        # Look for attention-like modules with explicit softmax attribute.
        has_scale = hasattr(module, "scale")
        has_qkv = any(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "wq", "wk", "wv"))
        has_softmax = hasattr(module, "softmax") and callable(module.softmax)

        return bool(has_scale and has_qkv and has_softmax)

    def apply(self, module: Any, config: PatchConfig) -> Any:
        softmax_fn = getattr(module, "softmax", None)
        if softmax_fn is None or not callable(softmax_fn):
            return module

        from ...kernels import softmax as zmlx_softmax

        def _zmlx_softmax(x: Any, axis: int = -1, precise: bool = False) -> Any:
            # Only replace last-dim softmax when precision requirements allow.
            if axis not in (-1, x.ndim - 1) or precise:
                return mx.softmax(x, axis=axis, precise=precise)
            tg = config.threadgroup if isinstance(config.threadgroup, int) else 256
            return zmlx_softmax.softmax_lastdim(
                x,
                threadgroup=tg,
            )

        module._zmlx_original_softmax = softmax_fn  # type: ignore[attr-defined]
        module.softmax = _zmlx_softmax  # type: ignore[assignment]
        return module


register(_SoftmaxPattern())
