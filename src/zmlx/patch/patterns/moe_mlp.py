"""MoE MLP pattern: fuse Moe dispatch + combine logic with expert selection.

Targets Mixtral/DeepSeek style MoE layers with a `gate` and `experts` attribute,
and Qwen3 style MoE layers with `gate` and `switch_mlp`.
Fuses the top-k selection, routing, and combining passes.
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
            # 1. Fused Gating: Softmax + Top-2 + Re-normalization in one kernel
            logits = self_mod.gate(x)
            # weights: (B, 2), indices: (B, 2)
            weights, indices = moe.top2_gating_softmax(logits)
            
            # 2. Expert Execution
            if hasattr(self_mod, "switch_mlp"):
                # Qwen3 style: vectorized experts (SwitchGLU)
                # SwitchGLU handles the internal routing/experts but expects indices.
                expert_outputs = self_mod.switch_mlp(x, indices)
            else:
                # Mixtral/DeepSeek style: list of expert modules
                # Group by expert to minimize module calls
                B, K = indices.shape
                D = x.shape[-1]
                expert_outputs = mx.zeros((B, K, D), dtype=x.dtype)
                
                for i, expert in enumerate(self_mod.experts):
                    for k in range(K):
                        mask = (indices[:, k] == i)
                        if mask.any():
                            expert_outputs[mask, k] = expert(x[mask])

            # 3. Fused Combine: Weighted sum of expert outputs
            # expert_outputs shape: (B, K, D), weights shape: (B, K)
            return moe.moe_combine(expert_outputs, weights)

        # Store original for unpatch
        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module

register(_MoEMLPPattern())
