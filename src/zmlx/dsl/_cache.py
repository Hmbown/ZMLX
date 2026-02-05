"""Family-aware caching layer.

Thin wrapper — for now, just re-exports ``functools.cache`` since the
underlying kernel builders already use ``@cache`` and MLX's source-string
cache. This module exists as a hook for future cache introspection.
"""

from __future__ import annotations

from functools import cache

__all__ = ["cache"]
