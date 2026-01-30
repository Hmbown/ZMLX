"""GeGLU MLP pattern: fuse gelu(gate) * up into a single ZMLX kernel.

Structural match: module has `gate_proj` + `up_proj` + `down_proj` with GeLU activation
(Gemma pattern).
"""

from __future__ import annotations

from typing import Any

import mlx.nn as nn

from ...kernels import transformer
from .._registry import register
from .._types import PatchConfig


def _is_gelu_activation(module: Any) -> bool:
    """Check if the module uses GeLU-based gating."""
    if hasattr(module, "hidden_act"):
        return module.hidden_act in ("gelu", "gelu_new", "gelu_tanh", "gelu_fast")
    for attr_name in ("act", "act_fn", "activation_fn", "activation"):
        act = getattr(module, attr_name, None)
        if act is not None:
            act_type = type(act).__name__
            if "gelu" in act_type.lower() or "GELU" in act_type:
                return True
    return False


class _GeGLUMLPPattern:
    @property
    def name(self) -> str:
        return "geglu_mlp"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False
        has_gate_up = hasattr(module, "gate_proj") and hasattr(module, "up_proj")
        has_down = hasattr(module, "down_proj")
        if has_gate_up and has_down and _is_gelu_activation(module):
            return True
        return False

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = module.__call__.__func__ if hasattr(module.__call__, "__func__") else None

        def patched_call(self_mod: Any, x: Any) -> Any:
            gate = self_mod.gate_proj(x)
            up = self_mod.up_proj(x)
            activated = transformer.geglu2(gate, up)
            return self_mod.down_proj(activated)

        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_GeGLUMLPPattern())
