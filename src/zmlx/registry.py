"""Kernel registry with rich introspection.

Backward-compatible: existing ``list_kernels()``, ``clear_cache()``,
``cache_size()`` keep working exactly as before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .cache import GLOBAL_KERNEL_CACHE

# ---------------------------------------------------------------------------
# Legacy API (unchanged)
# ---------------------------------------------------------------------------


def list_kernels() -> list[str]:
    """List all currently cached Metal kernels."""
    return [key.name for key in GLOBAL_KERNEL_CACHE.keys()]


def clear_cache() -> None:
    """Clear the global kernel cache."""
    GLOBAL_KERNEL_CACHE.clear()


def cache_size() -> int:
    """Return the number of kernels in the cache."""
    return GLOBAL_KERNEL_CACHE.size()


# ---------------------------------------------------------------------------
# Enhanced registry
# ---------------------------------------------------------------------------


@dataclass
class KernelEntry:
    """Rich metadata for a registered kernel."""

    name: str
    ir_node: Any | None = None  # KernelIR from ir.py (optional)
    pattern: str | None = None  # PatternKind value string
    has_vjp: bool = False
    run_count: int = 0
    last_used: float = 0.0
    tags: frozenset[str] = field(default_factory=frozenset)

    def touch(self) -> None:
        """Update usage tracking."""
        self.run_count += 1
        self.last_used = time.time()


class KernelRegistry:
    """Rich kernel registry with discovery and introspection.

    This supplements the compile-cache (``GLOBAL_KERNEL_CACHE``) with
    optional metadata. Kernels may exist in the compile cache without
    being in this registry, and vice versa.
    """

    def __init__(self) -> None:
        self._entries: dict[str, KernelEntry] = {}

    def register(self, entry: KernelEntry) -> None:
        """Register or update a kernel entry."""
        self._entries[entry.name] = entry

    def get(self, name: str) -> KernelEntry | None:
        """Look up a kernel by name."""
        return self._entries.get(name)

    def list_kernels(self, *, tag: str | None = None) -> list[KernelEntry]:
        """List kernels, optionally filtered by tag."""
        entries = list(self._entries.values())
        if tag is not None:
            entries = [e for e in entries if tag in e.tags]
        return sorted(entries, key=lambda e: e.name)

    def by_pattern(self, kind: str) -> list[KernelEntry]:
        """List kernels by pattern kind value string."""
        return [e for e in self._entries.values() if e.pattern == kind]

    def with_vjp(self) -> list[KernelEntry]:
        """List kernels that have backward (VJP) support."""
        return [e for e in self._entries.values() if e.has_vjp]

    def hottest(self, n: int = 10) -> list[KernelEntry]:
        """Return the N most-used kernels."""
        entries = sorted(self._entries.values(), key=lambda e: e.run_count, reverse=True)
        return entries[:n]

    def size(self) -> int:
        """Number of registered kernels."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all registered entries."""
        self._entries.clear()

    def summary(self) -> str:
        """Human-readable summary of the registry."""
        total = self.size()
        with_vjp = len(self.with_vjp())
        patterns: dict[str, int] = {}
        for e in self._entries.values():
            p = e.pattern or "unknown"
            patterns[p] = patterns.get(p, 0) + 1

        lines = [
            f"KernelRegistry: {total} kernels ({with_vjp} with VJP)",
        ]
        for p, count in sorted(patterns.items()):
            lines.append(f"  {p}: {count}")
        return "\n".join(lines)


# Global enhanced registry instance
GLOBAL_REGISTRY = KernelRegistry()
