"""MoE MLP pattern: fuse MoE dispatch + combine logic with expert selection.

Targets several MoE styles:
- **Qwen3** — ``gate`` returns raw logits, ``switch_mlp`` handles experts.
  We fuse gating (softmax + top-2 + renorm) and combining into Metal kernels.
- **GLM-4 / DeepSeek-V3** — ``gate`` returns ``(indices, scores)`` already
  computed (sigmoid + group selection).  We preserve the original gating and
  only fuse the combine step.
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


class _MoEMLPPattern:
    @property
    def name(self) -> str:
        return "moe_mlp"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False
        # Match Qwen3MoeSparseMoeBlock and similar (Mixtral/DeepSeek)
        has_gate = hasattr(module, "gate")
        # Check for experts list or a single expert-handling module like switch_mlp
        has_experts = hasattr(module, "experts") or hasattr(module, "switch_mlp")
        return bool(has_gate and has_experts)

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = module.__call__

        def patched_call(self_mod: Any, x: Any) -> Any:
            # 1. Gating — detect whether gate returns logits or (inds, scores)
            gate_out = self_mod.gate(x)

            if isinstance(gate_out, tuple):
                # Gate already computed indices + scores (GLM-4, DeepSeek-V3 style)
                # Preserve the original gating logic exactly — only fuse combine.
                indices, weights = gate_out
            else:
                # Gate returned raw logits (Qwen3, Mixtral style)
                # Fuse softmax + top-2 + renorm into a single Metal kernel.
                weights, indices = moe.top2_gating_softmax(gate_out)

            # 2. Expert Execution
            if hasattr(self_mod, "switch_mlp"):
                # Qwen3/GLM style: vectorized experts (SwitchGLU)
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
