"""Residual + RMSNorm fusion at TransformerBlock level.

Matches Llama-style blocks that compute ``h = x + self_attn(norm(x))``
followed by ``mlp(norm(h))``. Rewrites the block to use
``zmlx.kernels.transformer.rmsnorm_residual`` for the post-attention norm.
"""

from __future__ import annotations

from typing import Any

import mlx.nn as nn

from .._registry import register
from .._types import PatchConfig


class _ResidualNormPattern:
    @property
    def name(self) -> str:
        return "residual_norm"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False

        attn_norm = getattr(module, "input_layernorm", None) or getattr(module, "norm1", None)
        post_norm = getattr(module, "post_attention_layernorm", None) or getattr(module, "norm2", None)
        attn = getattr(module, "self_attn", None) or getattr(module, "attention", None)
        mlp = (
            getattr(module, "mlp", None)
            or getattr(module, "feed_forward", None)
            or getattr(module, "ffn", None)
        )

        if attn_norm is None or post_norm is None or attn is None or mlp is None:
            return False

        # Only fuse for RMSNorm-based blocks.
        from .._modules import ZMLXRMSNorm

        return isinstance(post_norm, (nn.RMSNorm, ZMLXRMSNorm))

    def apply(self, module: Any, config: PatchConfig) -> Any:
        from ...kernels import transformer

        attn_norm_name = "input_layernorm" if hasattr(module, "input_layernorm") else "norm1"
        post_norm_name = (
            "post_attention_layernorm"
            if hasattr(module, "post_attention_layernorm")
            else "norm2"
        )
        attn_name = "self_attn" if hasattr(module, "self_attn") else "attention"
        if hasattr(module, "mlp"):
            mlp_name = "mlp"
        elif hasattr(module, "feed_forward"):
            mlp_name = "feed_forward"
        else:
            mlp_name = "ffn"

        original_call = module.__call__.__func__ if hasattr(module.__call__, "__func__") else None

        def _call_attn(attn_mod: Any, x_in: Any, mask: Any | None, cache: Any | None) -> Any:
            if mask is None and cache is None:
                return attn_mod(x_in)
            try:
                return attn_mod(x_in, mask=mask, cache=cache)
            except TypeError:
                if cache is not None:
                    try:
                        return attn_mod(x_in, mask, cache)
                    except TypeError:
                        return attn_mod(x_in, mask)
                if mask is not None:
                    try:
                        return attn_mod(x_in, mask)
                    except TypeError:
                        return attn_mod(x_in)
                return attn_mod(x_in)

        def patched_call(self_mod: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
            mask = kwargs.get("mask")
            cache = kwargs.get("cache")
            if len(args) > 0:
                mask = args[0]
            if len(args) > 1:
                cache = args[1]

            attn_norm = getattr(self_mod, attn_norm_name)
            post_norm = getattr(self_mod, post_norm_name)
            attn_mod = getattr(self_mod, attn_name)
            mlp_mod = getattr(self_mod, mlp_name)

            attn_in = attn_norm(x)
            attn_out = _call_attn(attn_mod, attn_in, mask, cache)

            tg = config.threadgroup if isinstance(config.threadgroup, int) else 256
            normed, updated_res = transformer.rmsnorm_residual(
                attn_out,
                x,
                post_norm.weight,
                eps=getattr(post_norm, "eps", 1e-6),
                threadgroup=tg,
            )

            mlp_out = mlp_mod(normed)
            return updated_res + mlp_out

        module._zmlx_original_call = original_call  # type: ignore[attr-defined]
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_ResidualNormPattern())
