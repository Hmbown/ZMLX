"""SwiGLU MLP pattern: fuse silu(gate) * up into a single ZMLX kernel.

Structural match: module has `gate_proj` + `up_proj` + `down_proj` attributes
(Llama/Mistral/Qwen MLP pattern).  Also handles the fused `gate_up_proj` + `down_proj`
variant (Phi-3).

Only the activation is fused; linear layers are left untouched (quantized-safe).
"""

from __future__ import annotations

from typing import Any

import mlx.nn as nn

from ...kernels import transformer
from .._registry import register
from .._types import PatchConfig


def _is_silu_activation(module: Any) -> bool:
    """Heuristic: check if the module uses SiLU-based gating."""
    # Check for explicit hidden_act attribute or config
    if hasattr(module, "hidden_act"):
        return module.hidden_act in ("silu", "swish")
    # Check for a SiLU activation attribute
    for attr_name in ("act", "act_fn", "activation_fn", "activation"):
        act = getattr(module, attr_name, None)
        if act is not None:
            act_type = type(act).__name__
            if act_type in ("SiLU", "silu"):
                return True
    return True  # Default assumption for gate+up pattern


class _SwiGLUMLPPattern:
    @property
    def name(self) -> str:
        return "swiglu_mlp"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False
        # Skip MoE switch/dispatch modules — they take (x, inds), not just (x)
        if "switch" in name.lower() or "dispatch" in name.lower():
            return False
        if parent is not None and (hasattr(parent, "router") or hasattr(parent, "gate")):
            return False
        # Pattern 1: gate_proj + up_proj + down_proj (Llama/Mistral/Qwen)
        has_gate_up = hasattr(module, "gate_proj") and hasattr(module, "up_proj")
        has_down = hasattr(module, "down_proj")
        if has_gate_up and has_down and _is_silu_activation(module):
            return True
        return False

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = module.__call__.__func__ if hasattr(module.__call__, "__func__") else None

        def patched_call(self_mod: Any, x: Any) -> Any:
            gate = self_mod.gate_proj(x)
            up = self_mod.up_proj(x)
            activated = transformer.swiglu2(gate, up)
            return self_mod.down_proj(activated)

        # Store original for unpatch
        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_SwiGLUMLPPattern())
