"""MoE MLP pattern: fuse the expert combine step with a Metal kernel.

Preserves each model's original gating logic (softmax ordering, expert bias,
renormalization) exactly — only the final weighted-sum of expert outputs is
replaced with a fused ``moe_combine`` Metal kernel.

Targets several MoE styles:
- **Qwen3** — ``gate`` returns raw logits, ``switch_mlp`` handles experts.
- **GPT-OSS** — ``router`` returns raw logits, ``experts`` is a SwitchGLU.
- **LFM2** — ``gate`` returns raw logits with ``expert_bias`` post-softmax.
- **GLM-4 / DeepSeek-V3** — ``gate`` returns ``(indices, scores)`` already
  computed (sigmoid + group selection).
- **Mixtral** — ``gate`` returns logits, ``experts`` is a list of modules.

Also handles ``shared_experts`` (additive dense MLP) when present.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...kernels import moe
from .._registry import register
from .._types import PatchConfig


def _gating(self_mod: Any, x: Any, gate_attr: str, k: int) -> tuple[Any, Any]:
    """Run the model's gating logic faithfully, returning (indices, weights).

    Handles several conventions:
    - Gate that returns a tuple ``(indices, scores)`` directly (GLM-4 / DeepSeek-V3).
    - Gate that returns raw logits, with optional ``expert_bias`` and
      ``norm_topk_prob`` (Qwen3, LFM2, Mixtral, GPT-OSS).
    """
    gate_fn = getattr(self_mod, gate_attr)
    gate_out = gate_fn(x)

    if isinstance(gate_out, tuple):
        # Gate already computed indices + scores (GLM-4, DeepSeek-V3 style).
        indices, weights = gate_out
        return indices, weights

    # Raw logits path — replicate the standard gating sequence exactly.
    gates = gate_out.astype(mx.float32)
    gates = mx.softmax(gates, axis=-1)

    # Expert bias (LFM2-style): applied after softmax, before selection.
    expert_bias = getattr(self_mod, "expert_bias", None)
    if expert_bias is not None:
        gates = gates + expert_bias

    # Top-k selection.
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)

    # Optional renormalization so selected weights sum to 1.
    if getattr(self_mod, "norm_topk_prob", False):
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)

    scores = scores.astype(x.dtype)
    return inds, scores


class _MoEMLPPattern:
    @property
    def name(self) -> str:
        return "moe_mlp"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False
        # Match Qwen3MoeSparseMoeBlock (gate), GPT-OSS MLPBlock (router), etc.
        has_gate = hasattr(module, "gate") or hasattr(module, "router")
        # Check for experts list or a single expert-handling module like switch_mlp
        has_experts = hasattr(module, "experts") or hasattr(module, "switch_mlp")
        return bool(has_gate and has_experts)

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = module.__call__

        # Detect the number of experts activated per token from the module.
        num_experts_per_tok = (
            getattr(module, "num_experts_per_tok", None)
            or getattr(module, "top_k", None)
            or getattr(module, "num_selected_experts", None)
            or 2  # conservative fallback
        )

        # Resolve which attribute holds the gating linear layer.
        _gate_attr = "gate" if hasattr(module, "gate") else "router"

        def patched_call(self_mod: Any, x: Any) -> Any:
            # 1. Gating — preserve the model's original logic exactly.
            indices, weights = _gating(self_mod, x, _gate_attr, num_experts_per_tok)

            # 2. Expert Execution
            if hasattr(self_mod, "switch_mlp"):
                # Qwen3/GLM/LFM2 style: vectorized experts (SwitchGLU)
                expert_outputs = self_mod.switch_mlp(x, indices)
            else:
                # Mixtral/DeepSeek style: list of expert modules
                B = indices.shape[0]
                K = indices.shape[-1]
                D = x.shape[-1]
                expert_outputs = mx.zeros((B, K, D), dtype=x.dtype)

                for i, expert in enumerate(self_mod.experts):
                    for k in range(K):
                        mask = indices[:, k] == i
                        if mask.any():
                            expert_outputs[mask, k] = expert(x[mask])

            # 3. Fused Combine: weighted sum of expert outputs in one kernel
            y = moe.moe_combine(expert_outputs, weights)

            # 4. Shared experts (GLM-4, DeepSeek-V3): additive dense path
            shared = getattr(self_mod, "shared_experts", None)
            if shared is not None:
                y = y + shared(x)

            return y

        # Store original for unpatch
        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_MoEMLPPattern())
