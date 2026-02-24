"""In-process compile cache for Metal kernels.

A simple dict-based cache that stores compiled kernel objects (or the
error that prevented compilation).  The cache is keyed on a stable
hash of (op, dtype, template_id, knobs, normalized_source).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheEntry:
    ok: bool
    kernel: Any = None
    error: Exception | None = None


class CompileCache:
    """Thread-local (single-process) compile cache.

    Not designed for cross-process sharing -- each worker gets its own.
    """

    def __init__(self) -> None:
        self._cache: dict[str, CacheEntry] = {}

    def get(self, key: str) -> CacheEntry | None:
        return self._cache.get(key)

    def put_ok(self, key: str, kernel: Any) -> None:
        self._cache[key] = CacheEntry(ok=True, kernel=kernel)

    def put_err(self, key: str, error: Exception) -> None:
        self._cache[key] = CacheEntry(ok=False, error=error)

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
