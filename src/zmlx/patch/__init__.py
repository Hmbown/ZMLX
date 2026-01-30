"""zmlx.patch — Module-level patching for MLX models.

Usage::

    import zmlx
    model = zmlx.patch.patch(model)

Or selectively::

    model = zmlx.patch.patch(model, patterns=["rmsnorm", "swiglu_mlp"])
"""

from __future__ import annotations

from typing import Any

import mlx.nn as nn

from ._registry import _ensure_loaded, get_all_patterns, get_pattern, list_patterns
from ._traversal import apply_patterns
from ._types import PatchConfig, PatchResult


def patch(
    model: nn.Module,
    *,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
    compute_dtype: str = "float32",
    threadgroup: int = 256,
    verbose: bool = False,
) -> nn.Module:
    """Patch an MLX model to use fused ZMLX Metal kernels.

    Walks the module tree and replaces matching submodules with
    ZMLX-backed drop-in replacements.

    Args:
        model: The nn.Module to patch (modified in place and returned).
        patterns: List of pattern names to apply. ``None`` means all.
        exclude: Pattern names to skip.
        compute_dtype: Compute dtype name (e.g. "float32", "float16").
        threadgroup: Default threadgroup size for fused kernels.
        verbose: Print each replacement as it happens.

    Returns:
        The same model, modified in place, with a ``_zmlx_patch_result``
        attribute containing a :class:`PatchResult`.
    """
    _ensure_loaded()

    config = PatchConfig(
        compute_dtype=compute_dtype,
        threadgroup=threadgroup,
        verbose=verbose,
    )

    # Resolve patterns
    if patterns is not None:
        selected = [get_pattern(name) for name in patterns]
    else:
        selected = list(get_all_patterns().values())

    if exclude:
        exclude_set = set(exclude)
        selected = [p for p in selected if p.name not in exclude_set]

    if verbose:
        print(f"[zmlx.patch] Applying {len(selected)} patterns: {[p.name for p in selected]}")

    result = apply_patterns(model, selected, config)

    if verbose:
        print(result.summary())

    model._zmlx_patch_result = result  # type: ignore[attr-defined]
    return model


def unpatch(model: nn.Module) -> nn.Module:
    """Remove ZMLX patches from a model.

    Note: Only modules that stored their original call can be unpatched.
    Module replacements (like RMSNorm -> ZMLXRMSNorm) cannot be automatically
    reversed — reload the model instead.
    """
    children: dict[str, Any] = {}
    if hasattr(model, "children") and callable(model.children):
        children = dict(model.children())

    for _name, child in children.items():
        if hasattr(child, "_zmlx_original_call") and child._zmlx_original_call is not None:
            child.__call__ = child._zmlx_original_call
            del child._zmlx_original_call
        if hasattr(child, "_zmlx_original_softmax"):
            child.softmax = child._zmlx_original_softmax
            del child._zmlx_original_softmax
        if isinstance(child, nn.Module):
            unpatch(child)

    if hasattr(model, "_zmlx_patch_result"):
        del model._zmlx_patch_result
    return model


__all__ = [
    "patch",
    "unpatch",
    "list_patterns",
    "PatchConfig",
    "PatchResult",
]
