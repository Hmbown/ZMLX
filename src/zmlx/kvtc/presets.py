"""Model presets for KVTC compression.

Each preset captures the KV cache geometry and RoPE configuration for a
supported model family, so that calibration and codec setup can be done with
a single ``model_preset("lfm2")`` call instead of manually specifying every
parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .rope import RotaryConfig


@dataclass(frozen=True)
class KVTCPreset:
    """Complete KVTC configuration for a model family.

    Attributes:
        name: Canonical short name (e.g. ``"lfm2"``).
        description: Human-readable model description.
        layers: Number of transformer layers.
        kv_heads: Number of KV heads per layer.
        head_dim: Dimension of each KV head.
        rope: RoPE configuration (dim, base, traditional, offset).
        mode: ``"dual_stream"`` for separate K and V caches, or
            ``"single_stream"`` for MLA-style keys-only caches (GLM).
        v_dim: Value head dimension. 0 for single_stream (MLA) where values
            are derived from the latent portion at attention time.
        aliases: Alternative names for fuzzy matching.
    """

    name: str
    description: str
    layers: int
    kv_heads: int
    head_dim: int
    rope: RotaryConfig
    mode: Literal["dual_stream", "single_stream"]
    v_dim: int
    aliases: tuple[str, ...] = ()


_PRESETS: dict[str, KVTCPreset] = {}


def _register(p: KVTCPreset) -> KVTCPreset:
    _PRESETS[p.name] = p
    for alias in p.aliases:
        _PRESETS[alias] = p
    return p


# ── LFM2-8B-A1B ──────────────────────────────────────────────────────────
_register(KVTCPreset(
    name="lfm2",
    description="LFM2-8B-A1B (24 layers, 8 KV heads, head_dim=64)",
    layers=24,
    kv_heads=8,
    head_dim=64,
    rope=RotaryConfig(dim=64, base=1_000_000.0, traditional=False, offset=0),
    mode="dual_stream",
    v_dim=64,
    aliases=("lfm2-8b", "lfm2-8b-a1b"),
))

# ── Qwen3-30B-A3B ────────────────────────────────────────────────────────
_register(KVTCPreset(
    name="qwen3",
    description="Qwen3-30B-A3B (48 layers, 4 KV heads, head_dim=128)",
    layers=48,
    kv_heads=4,
    head_dim=128,
    rope=RotaryConfig(dim=128, base=10_000_000.0, traditional=False, offset=0),
    mode="dual_stream",
    v_dim=128,
    aliases=("qwen3-30b", "qwen3-30b-a3b"),
))

# ── GLM-4.7-Flash (MLA) ──────────────────────────────────────────────────
# Cache stores [kv_latent(512) | k_pe(64)] in keys slot. Values slot is empty.
# RoPE applies only to the last 64 dims (offset=512), traditional=True.
_register(KVTCPreset(
    name="glm",
    description="GLM-4.7-Flash MLA (47 layers, 1 KV head, head_dim=576, keys-only)",
    layers=47,
    kv_heads=1,
    head_dim=576,
    rope=RotaryConfig(dim=64, base=1_000_000.0, traditional=True, offset=512),
    mode="single_stream",
    v_dim=0,
    aliases=("glm4", "glm-4.7", "glm-4.7-flash"),
))


def model_preset(name: str) -> KVTCPreset:
    """Look up a preset by name with fuzzy matching.

    Tries exact match first, then case-insensitive, then substring match.

    Raises:
        KeyError: If no preset matches.
    """
    # Exact match
    if name in _PRESETS:
        return _PRESETS[name]

    # Case-insensitive match
    lower = name.lower()
    for key, preset in _PRESETS.items():
        if key.lower() == lower:
            return preset

    # Substring match (check if name is contained in any key or vice versa)
    for key, preset in _PRESETS.items():
        if lower in key.lower() or key.lower() in lower:
            return preset

    available = sorted({p.name for p in _PRESETS.values()})
    raise KeyError(f"Unknown preset {name!r}. Available: {available}")


def list_presets() -> list[KVTCPreset]:
    """Return all unique presets (no duplicates from aliases)."""
    seen: set[str] = set()
    result: list[KVTCPreset] = []
    for preset in _PRESETS.values():
        if preset.name not in seen:
            seen.add(preset.name)
            result.append(preset)
    return result
