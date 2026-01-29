from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

@dataclass(frozen=True)
class KernelCacheKey:
    """Hashable key used by :class:`KernelCache` to deduplicate kernels.

    The ``source_hash`` and ``header_hash`` fields are SHA-256 digests so that
    cache lookups are fast regardless of source length.

    Attributes:
        name: Kernel name.
        input_names: Input buffer names in order.
        output_names: Output buffer names in order.
        source_hash: SHA-256 hash of the Metal source.
        header_hash: SHA-256 hash of the Metal header.
        ensure_row_contiguous: Whether inputs are forced row-contiguous.
        atomic_outputs: Whether outputs are allocated as atomics.
    """

    name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    source_hash: str
    header_hash: str
    ensure_row_contiguous: bool
    atomic_outputs: bool

    @classmethod
    def from_parts(
        cls,
        *,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
        header: str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> KernelCacheKey:
        """Create a cache key from kernel metadata.

        Args:
            name: Kernel name.
            input_names: Input buffer names in order.
            output_names: Output buffer names in order.
            source: Metal source string.
            header: Metal header string.
            ensure_row_contiguous: Whether inputs are forced row-contiguous.
            atomic_outputs: Whether outputs are allocated as atomics.

        Returns:
            A :class:`KernelCacheKey` with hashed source and header.
        """
        return cls(
            name=name,
            input_names=tuple(input_names),
            output_names=tuple(output_names),
            source_hash=_sha256(source),
            header_hash=_sha256(header or ""),
            ensure_row_contiguous=ensure_row_contiguous,
            atomic_outputs=atomic_outputs,
        )

class KernelCache:
    """In-memory cache of compiled kernel callables.

    Note: MLX itself may cache Metal compilation artifacts internally. This cache is about
    avoiding repeated construction of the Python callable objects in a single process.
    """

    def __init__(self) -> None:
        self._cache: dict[KernelCacheKey, Any] = {}

    def get(self, key: KernelCacheKey) -> Any | None:
        """Return a cached kernel callable if present.

        Args:
            key: Cache key for the kernel.

        Returns:
            The cached value, or ``None`` if not found.
        """
        return self._cache.get(key)

    def put(self, key: KernelCacheKey, value: Any) -> Any:
        """Insert a kernel callable into the cache.

        Args:
            key: Cache key for the kernel.
            value: Kernel callable (typically a :class:`MetalKernel`).

        Returns:
            The stored value.
        """
        self._cache[key] = value
        return value

    def clear(self) -> None:
        """Remove all cached kernels."""
        self._cache.clear()

    def keys(self) -> list[KernelCacheKey]:
        """Return a list of cache keys currently stored."""
        return list(self._cache.keys())

    def size(self) -> int:
        """Return the number of cached kernels."""
        return len(self._cache)

GLOBAL_KERNEL_CACHE = KernelCache()
