from __future__ import annotations

from typing import Any

import mlx.nn as nn

from ._types import PatchConfig, PatchPattern, PatchResult


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
