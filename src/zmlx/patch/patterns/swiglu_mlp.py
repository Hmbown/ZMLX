"""SwiGLU MLP pattern: fuse silu(gate) * up into a single ZMLX kernel.

Structural match: module has `gate_proj` + `up_proj` + `down_proj` attributes
(Llama/Mistral/Qwen MLP pattern), or `w1` + `w3` + `w2` (LFM/Llama reference).
Also handles the fused `gate_up_proj` + `down_proj` variant (Phi-3).

Only the activation is fused; linear layers are left untouched (quantized-safe).
"""

from __future__ import annotations

import os
from typing import Any

import mlx.nn as nn

from ...kernels import quant, transformer
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
_QSWIGLU_MODE_ENV = "ZMLX_FUSED_QSWIGLU_MODE"
_QSWIGLU_PROGRESSIVE_ENV = "ZMLX_FUSED_QSWIGLU_PROGRESSIVE"
_QSWIGLU_EPS_ENV = "ZMLX_FUSED_QSWIGLU_EPS"
_QSWIGLU_TG_ENV = "ZMLX_FUSED_QSWIGLU_TG"
_QSWIGLU_PER_GROUP_ENV = "ZMLX_FUSED_QSWIGLU_PER_GROUP"
_QSWIGLU_MAX_TOKENS_ENV = "ZMLX_FUSED_QSWIGLU_MAX_TOKENS"
_QSWIGLU_MAX_OUT_ENV = "ZMLX_FUSED_QSWIGLU_MAX_OUT"
_QSWIGLU_MAX_IN_ENV = "ZMLX_FUSED_QSWIGLU_MAX_IN"
_QSWIGLU_TG_MIN_EPS_ENV = "ZMLX_FUSED_QSWIGLU_TG_MIN_EPS"
_QSWIGLU_TG_ALLOWLIST_ENV = "ZMLX_FUSED_QSWIGLU_TG_ALLOWLIST"
_QSWIGLU_TG_DENY_FAMILY_ENV = "ZMLX_FUSED_QSWIGLU_TG_DENY_FAMILY"


def _is_quantized_linear(module: Any) -> bool:
    return isinstance(module, nn.QuantizedLinear)


def _get_swiglu_projections(module: Any) -> tuple[Any, Any, Any] | None:
    """Return (gate, up, down) projections if the module matches a known layout."""
    if hasattr(module, "gate_proj") and hasattr(module, "up_proj") and hasattr(module, "down_proj"):
        return (module.gate_proj, module.up_proj, module.down_proj)
    if hasattr(module, "w1") and hasattr(module, "w3") and hasattr(module, "w2"):
        return (module.w1, module.w3, module.w2)
    return None


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}; expected integer.") from exc


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}; expected float.") from exc


def _parse_shape_allowlist(raw: str | None, *, default: str) -> set[tuple[int, int]] | None:
    """Parse a comma-separated KxN allowlist string.

    Returns None to allow all shapes. Entries are "KxN" pairs.
    """
    if raw is None or raw.strip() == "":
        raw = default
    raw = raw.strip()
    if raw in {"*", "all", "any"}:
        return None
    allow: set[tuple[int, int]] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Invalid {_QSWIGLU_TG_ALLOWLIST_ENV} entry {part!r}; expected KxN.")
        k_str, n_str = part.split("x", 1)
        try:
            allow.add((int(k_str), int(n_str)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid {_QSWIGLU_TG_ALLOWLIST_ENV} entry {part!r}; expected integer KxN."
            ) from exc
    return allow


def _tg_progressive_allowed(tokens: int, n_in: int, n_out: int, eps: float) -> bool:
    """Return True if TG progressive is allowed for the shape."""
    if tokens != 1:
        return False
    min_eps = _parse_float_env(_QSWIGLU_TG_MIN_EPS_ENV, 10.0)
    if eps < min_eps:
        return False
    allowlist = _parse_shape_allowlist(
        os.environ.get(_QSWIGLU_TG_ALLOWLIST_ENV),
        default="2048x7168",
    )
    if allowlist is None:
        return True
    return (int(n_in), int(n_out)) in allowlist


def _tg_family_allowed(module: Any) -> bool:
    raw = os.environ.get(_QSWIGLU_TG_DENY_FAMILY_ENV, "lfm")
    tokens = [tok.strip().lower() for tok in raw.split(",") if tok.strip()]
    if not tokens:
        return True
    mod_name = (module.__class__.__module__ or "").lower()
    cls_name = module.__class__.__name__.lower()
    for tok in tokens:
        if tok in mod_name or tok in cls_name:
            return False
    return True


def _progressive_enabled() -> bool:
    raw = os.environ.get(_QSWIGLU_PROGRESSIVE_ENV, "")
    return raw.strip() not in ("", "0", "false", "False")


def _per_group_enabled() -> bool:
    raw = os.environ.get(_QSWIGLU_PER_GROUP_ENV, "")
    return raw.strip() not in ("", "0", "false", "False")


def _get_qswiglu_mode() -> str:
    raw = os.environ.get(_QSWIGLU_ENV)
    if raw is not None:
        if raw.strip() in ("", "0", "false", "False"):
            return "off"
        return "force"
    mode = os.environ.get(_QSWIGLU_MODE_ENV, "off").strip().lower()
    if mode not in {"off", "auto", "force"}:
        raise ValueError(f"Invalid {_QSWIGLU_MODE_ENV}={mode!r}; expected off|auto|force.")
    return mode


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
    mode: str,
    max_out: int,
    max_in: int,
    max_tokens: int,
    x: Any,
) -> bool:
    if mode == "off":
        return False
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
    if x_flat.ndim != 2:
        return False
    tokens = _total_tokens(prefix) if prefix else int(x_flat.shape[0])
    if tokens > max_tokens:
        return False
    if mode == "auto":
        n_out = int(gate_proj.weight.shape[0])
        n_in = int(x_flat.shape[1])
        if n_out > max_out or n_in > max_in:
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
            # Allow dense shared experts MLPs inside MoE blocks (e.g. GLM-4),
            # but avoid patching routed expert dispatch modules.
            if name.lower() != "shared_experts":
                return False
        projections = _get_swiglu_projections(module)
        if projections is not None and _is_silu_activation(module):
            return True
        return False

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = module.__call__.__func__ if hasattr(module.__call__, "__func__") else None

        def patched_call(self_mod: Any, x: Any) -> Any:
            mode = _get_qswiglu_mode()
            projections = _get_swiglu_projections(self_mod)
            if projections is None:
                return original_call(self_mod, x) if original_call is not None else self_mod.__call__(x)
            gate_proj, up_proj, down_proj = projections
            if mode != "off":
                max_tokens = _parse_int_env(_QSWIGLU_MAX_TOKENS_ENV, 1)
                max_out = _parse_int_env(_QSWIGLU_MAX_OUT_ENV, 2048)
                max_in = _parse_int_env(_QSWIGLU_MAX_IN_ENV, 2048)
                eps = _parse_float_env(_QSWIGLU_EPS_ENV, 0.0)
                tg = _parse_int_env(_QSWIGLU_TG_ENV, 0)
                per_group = _per_group_enabled()
                if _can_fuse_quant_swiglu(
                    gate_proj,
                    up_proj,
                    mode=mode,
                    max_out=max_out,
                    max_in=max_in,
                    max_tokens=max_tokens,
                    x=x,
                ):
                    x_flat, prefix, orig_ndim = _flatten_x(x)
                    if _progressive_enabled() and tg > 0:
                        if not _tg_family_allowed(self_mod):
                            gate = gate_proj(x)
                            up = up_proj(x)
                            activated = transformer.swiglu2(gate, up)
                            return down_proj(activated)
                        tokens = _total_tokens(prefix) if prefix else int(x_flat.shape[0])
                        n_out = int(gate_proj.weight.shape[0])
                        n_in = int(x_flat.shape[1])
                        if not _tg_progressive_allowed(tokens, n_in, n_out, eps):
                            gate = gate_proj(x)
                            up = up_proj(x)
                            activated = transformer.swiglu2(gate, up)
                            return down_proj(activated)
                    if _progressive_enabled():
                        if per_group and tg > 0:
                            tg = 0
                        fused = quant.fused_quantized_swiglu_gemv_progressive(
                            x_flat,
                            gate_proj.weight,
                            gate_proj.scales,
                            gate_proj.biases,
                            up_proj.weight,
                            up_proj.scales,
                            up_proj.biases,
                            group_size=gate_proj.group_size,
                            bits=gate_proj.bits,
                            epsilon=eps,
                            threadgroup=(tg if tg > 0 else None),
                            per_group=per_group,
                        )
                    else:
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
                gate = gate_proj(x)
                up = up_proj(x)
                activated = transformer.swiglu2(gate, up)
            return down_proj(activated)

        # Store original for unpatch
        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_SwiGLUMLPPattern())
