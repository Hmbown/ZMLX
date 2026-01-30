from __future__ import annotations

from ._types import PatchPattern

_PATTERNS: dict[str, PatchPattern] = {}


def register(pattern: PatchPattern) -> PatchPattern:
    """Register a patch pattern by name."""
    _PATTERNS[pattern.name] = pattern
    return pattern


def get_pattern(name: str) -> PatchPattern:
    """Get a registered pattern by name."""
    _ensure_loaded()
    if name not in _PATTERNS:
        raise KeyError(f"Unknown patch pattern: {name!r}. Available: {list_patterns()}")
    return _PATTERNS[name]


def list_patterns() -> list[str]:
    """List all registered pattern names."""
    _ensure_loaded()
    return sorted(_PATTERNS.keys())


def get_all_patterns() -> dict[str, PatchPattern]:
    """Return all registered patterns."""
    _ensure_loaded()
    return dict(_PATTERNS)


def _ensure_loaded() -> None:
    """Import all built-in patterns to populate the registry."""
    if _PATTERNS:
        return
    from .patterns import (  # noqa: F401
        geglu_mlp,
        layernorm,
        residual_norm,
        rmsnorm,
        softmax,
        swiglu_mlp,
    )
