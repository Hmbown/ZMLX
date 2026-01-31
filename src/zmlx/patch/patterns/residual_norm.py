"""Residual + RMSNorm fusion at TransformerBlock level.

Matches transformer blocks that compute ``h = x + self_attn(norm(x))``
followed by ``mlp(norm(h))``. Rewrites the block to use
``zmlx.kernels.norms.add_rms_norm`` for the post-attention norm,
fusing the residual add and RMSNorm into a single kernel with
improved float16 precision.

Supports all major mlx-lm model architectures (Llama, Qwen, Mistral,
Gemma, Phi, DeepSeek, LFM, GLM, etc.) via broad attribute name matching.
"""

from __future__ import annotations

from typing import Any

import mlx.nn as nn

from .._registry import register
from .._types import PatchConfig

# ---------------------------------------------------------------------------
# Attribute name candidates — covers all 60+ mlx-lm architectures
# ---------------------------------------------------------------------------
_ATTN_NORM_NAMES = (
    "input_layernorm",
    "norm1",
    "operator_norm",
    "attention_norm",
    "attention_layernorm",
)
_POST_NORM_NAMES = (
    "post_attention_layernorm",
    "norm2",
    "ffn_norm",
    "pre_ff_layernorm",
    "feedforward_layernorm",
)
_ATTN_NAMES = ("self_attn", "attention", "conv")
_MLP_NAMES = ("mlp", "feed_forward", "ffn", "block_sparse_moe")


def _first_match(module: Any, names: tuple[str, ...]) -> tuple[str | None, Any]:
    """Return ``(attr_name, attr)`` for the first present attribute, or ``(None, None)``."""
    for name in names:
        attr = getattr(module, name, None)
        if attr is not None:
            return name, attr
    return None, None


class _ResidualNormPattern:
    @property
    def name(self) -> str:
        return "residual_norm"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False

        _, attn_norm = _first_match(module, _ATTN_NORM_NAMES)
        _, post_norm = _first_match(module, _POST_NORM_NAMES)
        _, attn = _first_match(module, _ATTN_NAMES)
        _, mlp = _first_match(module, _MLP_NAMES)

        if attn_norm is None or post_norm is None or attn is None or mlp is None:
            return False

        # Only fuse for RMSNorm-based blocks.
        from .._modules import ZMLXRMSNorm

        return isinstance(post_norm, (nn.RMSNorm, ZMLXRMSNorm))

    def apply(self, module: Any, config: PatchConfig) -> Any:
        attn_norm_name, _ = _first_match(module, _ATTN_NORM_NAMES)
        post_norm_name, _ = _first_match(module, _POST_NORM_NAMES)
        attn_name, _ = _first_match(module, _ATTN_NAMES)
        mlp_name, _ = _first_match(module, _MLP_NAMES)
        assert attn_norm_name and post_norm_name and attn_name and mlp_name  # ensured by matches()

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
            from ...kernels.norms import add_rms_norm

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
            normed, updated_res = add_rms_norm(
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
