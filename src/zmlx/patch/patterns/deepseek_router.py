"""DeepSeek/Kimi router pattern: fuse sigmoid top-k routing into a Metal kernel.

This is intentionally **opt-in** (not part of default patch presets).

Targets MLX-LM DeepSeek v3/v3.2 style gates where routing is:
  1) affinity = sigmoid(logits)
  2) selection_score = affinity + bias
  3) top-k experts selected by selection_score
  4) weights = normalize(affinity[topk])

Current fused kernel support:
  - n_routed_experts (Nr): 256 or 384
  - top_k (K): 8
  - group selection when ``n_group`` divides ``Nr`` (e.g. DeepSeek-V3.2 uses
    ``n_group=8, topk_group=4``)
  - norm_topk_prob == True
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...kernels.moe import deepseek_router_topk_sigmoid
from .._registry import register
from .._types import PatchConfig

_SUPPORTED_NR = {256, 384}
_SUPPORTED_K = 8


def _is_deepseek_gate_module(module: Any) -> bool:
    cls = module.__class__
    name = getattr(cls, "__name__", "") or ""
    mod_path = (getattr(cls, "__module__", "") or "").lower()
    if name != "MoEGate":
        return False
    return "mlx_lm.models.deepseek_v3" in mod_path or "mlx_lm.models.deepseek_v32" in mod_path


class _DeepSeekRouterPattern:
    @property
    def name(self) -> str:
        return "deepseek_router"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False
        if not _is_deepseek_gate_module(module):
            return False
        # Conservative: require the expected attribute surface.
        for attr in (
            "weight",
            "e_score_correction_bias",
            "top_k",
            "norm_topk_prob",
            "n_group",
            "topk_group",
            "routed_scaling_factor",
        ):
            if not hasattr(module, attr):
                return False
        return True

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = module.__call__.__func__ if hasattr(module.__call__, "__func__") else None
        if original_call is None:
            return module

        def patched_call(self_mod: Any, x: Any) -> Any:
            # Only fuse the exact DeepSeek/Kimi router convention we support.
            try:
                top_k = int(self_mod.top_k)
                n_group = int(self_mod.n_group)
                topk_group = int(self_mod.topk_group)
                norm_topk_prob = bool(self_mod.norm_topk_prob)
                scaling = float(self_mod.routed_scaling_factor)
            except Exception:
                return original_call(self_mod, x)

            if top_k != _SUPPORTED_K or not norm_topk_prob:
                return original_call(self_mod, x)
            if n_group <= 0 or topk_group <= 0 or topk_group > n_group:
                return original_call(self_mod, x)

            weight = getattr(self_mod, "weight", None)
            bias = getattr(self_mod, "e_score_correction_bias", None)
            if weight is None or bias is None:
                return original_call(self_mod, x)

            try:
                if int(getattr(weight, "ndim", 0)) != 2:
                    return original_call(self_mod, x)
                nr = int(weight.shape[0])
            except Exception:
                return original_call(self_mod, x)

            if nr not in _SUPPORTED_NR:
                return original_call(self_mod, x)

            tg = config.threadgroup if isinstance(config.threadgroup, int) else 256
            logits = x @ weight.T
            w, idx = deepseek_router_topk_sigmoid(
                logits,
                bias,
                k=top_k,
                n_group=n_group,
                topk_group=topk_group,
                threadgroup=tg,
                compute_dtype=mx.float32,  # Match MLX-LM reference (sigmoid in float32)
            )
            if scaling != 1.0:
                w = w * scaling
            return idx.astype(mx.int32), w

        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_DeepSeekRouterPattern())
