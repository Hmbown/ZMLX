"""SwiGLU MLP pattern: fuse silu(gate) * up into a single ZMLX kernel.

Structural match: module has `gate_proj` + `up_proj` + `down_proj` attributes
(Llama/Mistral/Qwen MLP pattern).  Also handles the fused `gate_up_proj` + `down_proj`
variant (Phi-3).

Only the activation is fused; linear layers are left untouched (quantized-safe).
"""

from __future__ import annotations

from typing import Any
import os

import mlx.nn as nn

from ...kernels import transformer, quant
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


_QSWIGLU_ENV = "ZMLX_FUSED_QSWIGLU"
_QSWIGLU_MAX_TOKENS_ENV = "ZMLX_FUSED_QSWIGLU_MAX_TOKENS"


def _is_quantized_linear(module: Any) -> bool:
    return isinstance(module, nn.QuantizedLinear)


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}; expected integer.") from exc


def _flatten_x(x: Any) -> tuple[Any, tuple[int, ...], int]:
    if x.ndim == 1:
        return x.reshape(1, x.shape[0]), (), 1
    if x.ndim == 2:
        return x, (int(x.shape[0]),), 2
    batch = 1
    for d in x.shape[:-1]:
        batch *= int(d)
    return x.reshape(batch, x.shape[-1]), tuple(int(d) for d in x.shape[:-1]), x.ndim


def _total_tokens(prefix: tuple[int, ...]) -> int:
    total = 1
    for d in prefix:
        total *= int(d)
    return total


def _can_fuse_quant_swiglu(
    gate_proj: Any,
    up_proj: Any,
    *,
    max_tokens: int,
    x: Any,
) -> bool:
    if not (_is_quantized_linear(gate_proj) and _is_quantized_linear(up_proj)):
        return False
    if gate_proj.mode != "affine" or up_proj.mode != "affine":
        return False
    if gate_proj.group_size != up_proj.group_size or gate_proj.bits != up_proj.bits:
        return False
    if gate_proj.bits not in (4, 8):
        return False
    if gate_proj.biases is None or up_proj.biases is None:
        return False

    x_flat, prefix, _ = _flatten_x(x)
    tokens = _total_tokens(prefix) if prefix else int(x_flat.shape[0])
    if tokens > max_tokens:
        return False
    if x_flat.ndim != 2:
        return False
    return True


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
            use_quant_fused = os.environ.get(_QSWIGLU_ENV, "").strip() not in ("", "0", "false", "False")
            if use_quant_fused:
                max_tokens = _parse_int_env(_QSWIGLU_MAX_TOKENS_ENV, 1)
                gate_proj = self_mod.gate_proj
                up_proj = self_mod.up_proj
                if _can_fuse_quant_swiglu(gate_proj, up_proj, max_tokens=max_tokens, x=x):
                    x_flat, prefix, orig_ndim = _flatten_x(x)
                    fused = quant.fused_quantized_swiglu_gemv(
                        x_flat,
                        gate_proj.weight,
                        gate_proj.scales,
                        gate_proj.biases,
                        up_proj.weight,
                        up_proj.scales,
                        up_proj.biases,
                        group_size=gate_proj.group_size,
                        bits=gate_proj.bits,
                    )
                    if orig_ndim == 1:
                        activated = fused.reshape(fused.shape[-1])
                    elif prefix:
                        activated = fused.reshape(*prefix, fused.shape[-1])
                    else:
                        activated = fused
                else:
                    gate = gate_proj(x)
                    up = up_proj(x)
                    activated = transformer.swiglu2(gate, up)
            else:
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
