from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

from .._compat import import_mx
from ..patch._types import (
    FusionCacheKey,
    FusionConfig,
    FusionLegacyCacheKey,
    FusionRuntimeState,
    FusionScopedCacheKey,
    FusionSignature,
    FusionSignatureItem,
)
from .jit import jit


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _signature_for(args: tuple[Any, ...], kwargs: dict[str, Any]) -> FusionSignature | None:
    if kwargs:
        return None
    signature: list[FusionSignatureItem] = []
    for arg in args:
        if _is_tensor_like(arg):
            signature.append(("tensor", tuple(int(dim) for dim in arg.shape), str(arg.dtype)))
            continue
        if _is_scalar(arg):
            signature.append(("scalar", type(arg).__name__, repr(arg)))
            continue
        return None
    return tuple(signature)


def _outputs_close(expected: Any, actual: Any, *, rtol: float, atol: float) -> bool:
    mx = import_mx()
    if _is_tensor_like(expected) and _is_tensor_like(actual):
        mx.eval(expected, actual)
        return bool(mx.allclose(expected, actual, rtol=rtol, atol=atol).item())
    if isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        if len(expected) != len(actual):
            return False
        return all(
            _outputs_close(exp_item, act_item, rtol=rtol, atol=atol)
            for exp_item, act_item in zip(expected, actual, strict=True)
        )
    return expected == actual


def _fusion_stats(fn: Callable[..., Any]) -> dict[str, Any] | None:
    stats_fn = getattr(fn, "fusion_stats", None)
    if not callable(stats_fn):
        return None
    try:
        stats = stats_fn()
    except Exception:
        return None
    if not isinstance(stats, dict):
        return None
    return stats


def _decision_key_for(
    pattern_name: str,
    signature: FusionSignature,
    module_instance_key: str | None,
) -> FusionCacheKey:
    if module_instance_key is None:
        return (pattern_name, signature)
    return (pattern_name, module_instance_key, signature)


def _legacy_decision_key_for(
    pattern_name: str,
    signature: FusionSignature,
) -> FusionLegacyCacheKey:
    return (pattern_name, signature)


def _normalize_legacy_key(
    legacy_key: FusionLegacyCacheKey,
    module_instance_key: str,
) -> FusionScopedCacheKey:
    """Convert a legacy (non-scoped) key to scoped format."""
    pattern_name, signature = legacy_key
    return (pattern_name, module_instance_key, signature)


def _has_key_with_migration(
    entries: set[FusionCacheKey],
    key: FusionCacheKey,
    legacy_key: FusionLegacyCacheKey,
    module_instance_key: str | None,
    normalize: bool,
) -> bool:
    """
    Check if key exists in entries, with optional legacy key migration.

    When normalize=True and a legacy key is found, it is removed from entries
    and the scoped key is added instead (migrating to the new format).

    Returns True if either key format is found.
    """
    if key in entries:
        return True
    if legacy_key in entries:
        if (
            normalize
            and module_instance_key is not None
            and isinstance(key, tuple)
            and len(key) == 3
        ):
            entries.remove(legacy_key)
            entries.add(key)
        return True
    return False


def wrap_callable_with_fusion_validation(
    fn: Callable[..., Any],
    *,
    pattern_name: str,
    fusion: FusionConfig,
    state: FusionRuntimeState,
    module_instance_key: str | None = None,
) -> Callable[..., Any]:
    """Wrap callable with opt-in fusion and per-signature validation/blacklist."""

    fused_candidate = jit(fn)

    @wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        signature = _signature_for(args, kwargs)
        if signature is None:
            return fn(*args, **kwargs)

        key = _decision_key_for(pattern_name, signature, module_instance_key)
        legacy_key = _legacy_decision_key_for(pattern_name, signature)
        if _has_key_with_migration(
            state.blacklist, key, legacy_key, module_instance_key, fusion.normalize_legacy_keys
        ):
            state.blacklist_hits += 1
            return fn(*args, **kwargs)

        if fusion.validate and not _has_key_with_migration(
            state.validated, key, legacy_key, module_instance_key, fusion.normalize_legacy_keys
        ):
            stats_before = _fusion_stats(fused_candidate)
            try:
                expected = fn(*args, **kwargs)
                actual = fused_candidate(*args, **kwargs)
            except Exception:
                state.compile_failures += 1
                state.blacklist.add(key)
                return fn(*args, **kwargs)
            stats_after = _fusion_stats(fused_candidate)
            if (
                stats_before is not None
                and stats_after is not None
                and int(stats_after.get("fallbacks", 0)) > int(stats_before.get("fallbacks", 0))
                and not bool(stats_after.get("fusable", False))
            ):
                state.compile_failures += 1
                state.blacklist.add(key)
                return expected

            if _outputs_close(expected, actual, rtol=fusion.rtol, atol=fusion.atol):
                state.validated.add(key)
                return actual

            state.validation_mismatches += 1
            state.blacklist.add(key)
            return expected

        stats_before = _fusion_stats(fused_candidate)
        try:
            out = fused_candidate(*args, **kwargs)
        except Exception:
            state.runtime_failures += 1
            state.blacklist.add(key)
            return fn(*args, **kwargs)
        stats_after = _fusion_stats(fused_candidate)
        if (
            stats_before is not None
            and stats_after is not None
            and int(stats_after.get("fallbacks", 0)) > int(stats_before.get("fallbacks", 0))
        ):
            state.runtime_failures += 1
            state.blacklist.add(key)
            return fn(*args, **kwargs)
        return out

    _wrapped._zmlx_fusion_wrapped = True  # type: ignore[attr-defined]
    if hasattr(fused_candidate, "fusion_stats"):
        _wrapped.fusion_stats = fused_candidate.fusion_stats  # type: ignore[attr-defined]
    return _wrapped


__all__ = ["wrap_callable_with_fusion_validation"]
