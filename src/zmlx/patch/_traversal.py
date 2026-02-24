from __future__ import annotations

from typing import Any

import mlx.nn as nn

from ..fusion.integration import wrap_callable_with_fusion_validation
from ._types import PatchConfig, PatchPattern, PatchResult


def _module_fingerprint(module: Any) -> str:
    """Generate a deterministic fingerprint for a module instance.

    Uses structural properties (class name, parameter count, child count)
    rather than id() for reproducibility across runs while maintaining
    reasonable cross-instance isolation.
    """
    parts = [type(module).__name__]
    params_fn = getattr(module, "parameters", None)
    if callable(params_fn):
        try:
            parts.append(f"p{sum(1 for _ in params_fn())}")
        except Exception:
            parts.append("p?")
    children_fn = getattr(module, "children", None)
    if callable(children_fn):
        try:
            children_dict = children_fn()
            if isinstance(children_dict, dict):
                parts.append(f"c{len(children_dict)}")
            else:
                parts.append("c?")
        except Exception:
            parts.append("c?")
    return ":".join(parts)


def _wrap_module_call_path(
    module: Any,
    wrapped_call: Any,
) -> None:
    """Install a per-instance class-level __call__ wrapper for module(...)."""

    original_class = module.__class__

    def _patched_call(_self: Any, *args: Any, **kwargs: Any) -> Any:
        return wrapped_call(*args, **kwargs)

    _patched_call._zmlx_fusion_wrapped = True  # type: ignore[attr-defined]
    if hasattr(wrapped_call, "fusion_stats"):
        _patched_call.fusion_stats = wrapped_call.fusion_stats  # type: ignore[attr-defined]

    module.__class__ = type(
        original_class.__name__,
        (original_class,),
        {"__call__": _patched_call},
    )
    module._zmlx_fusion_original_class = original_class  # type: ignore[attr-defined]
    module._zmlx_fusion_wrapped = True  # type: ignore[attr-defined]


def _maybe_wrap_fusion(
    module: Any,
    *,
    pattern_name: str,
    path: str,
    config: PatchConfig,
    result: PatchResult,
) -> None:
    fusion = config.fusion
    if not fusion.allows_pattern(pattern_name):
        return
    if not callable(module):
        result.fusion_skipped.append(f"{path}: missing __call__")
        return
    if getattr(module, "_zmlx_fusion_wrapped", False):
        return
    wrapped = module.__call__
    if getattr(wrapped, "_zmlx_fusion_wrapped", False):
        module._zmlx_fusion_wrapped = True  # type: ignore[attr-defined]
        return
    try:
        wrapped_call = wrap_callable_with_fusion_validation(
            wrapped,
            pattern_name=pattern_name,
            fusion=fusion,
            state=config.fusion_state,
            module_instance_key=f"{path}#{_module_fingerprint(module)}",
        )
        _wrap_module_call_path(module, wrapped_call)
    except Exception as exc:
        result.fusion_skipped.append(f"{path}: {exc}")
        return
    result.fusion_wrapped_count += 1


def _walk_modules(
    module: Any,
    patterns: list[PatchPattern],
    config: PatchConfig,
    result: PatchResult,
    path: str = "",
) -> None:
    """Recursively walk an nn.Module tree and apply matching patterns."""
    children: dict[str, Any] = {}
    if hasattr(module, "children") and callable(module.children):
        children = dict(module.children())

    # Recurse into children first (depth-first), handling lists of modules
    for child_name, child in children.items():
        child_path = f"{path}.{child_name}" if path else child_name
        if isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Module):
                    item_path = f"{child_path}[{i}]"
                    _walk_modules(item, patterns, config, result, item_path)
        elif isinstance(child, nn.Module):
            _walk_modules(child, patterns, config, result, child_path)

    # Now try to match patterns on children (for replacement)
    for child_name, child in children.items():
        child_path = f"{path}.{child_name}" if path else child_name
        if isinstance(child, list):
            # Handle lists of modules (e.g., self.layers = [...])
            for i, item in enumerate(child):
                if not isinstance(item, nn.Module):
                    continue
                item_path = f"{child_path}[{i}]"
                for pattern in patterns:
                    if pattern.matches(item, child_name, parent=module):
                        try:
                            replacement = pattern.apply(item, config)
                            child[i] = replacement
                            result.patched_count += 1
                            result.pattern_counts[pattern.name] = (
                                result.pattern_counts.get(pattern.name, 0) + 1
                            )
                            _maybe_wrap_fusion(
                                replacement,
                                pattern_name=pattern.name,
                                path=item_path,
                                config=config,
                                result=result,
                            )
                            if config.verbose:
                                print(f"  [zmlx.patch] {item_path}: {pattern.name}")
                        except Exception as e:
                            result.skipped.append(f"{item_path}: {e}")
                        break
        elif isinstance(child, nn.Module):
            for pattern in patterns:
                if pattern.matches(child, child_name, parent=module):
                    try:
                        replacement = pattern.apply(child, config)
                        setattr(module, child_name, replacement)
                        result.patched_count += 1
                        result.pattern_counts[pattern.name] = (
                            result.pattern_counts.get(pattern.name, 0) + 1
                        )
                        _maybe_wrap_fusion(
                            replacement,
                            pattern_name=pattern.name,
                            path=child_path,
                            config=config,
                            result=result,
                        )
                        if config.verbose:
                            print(f"  [zmlx.patch] {child_path}: {pattern.name}")
                    except Exception as e:
                        result.skipped.append(f"{child_path}: {e}")
                    break


def apply_patterns(
    model: Any,
    patterns: list[PatchPattern],
    config: PatchConfig,
) -> PatchResult:
    """Walk the model tree and apply all matching patterns."""
    result = PatchResult()
    _walk_modules(model, patterns, config, result)
    return result
