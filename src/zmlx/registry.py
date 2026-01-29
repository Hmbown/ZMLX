from __future__ import annotations

from .cache import GLOBAL_KERNEL_CACHE


def list_kernels() -> list[str]:
    """List all currently cached Metal kernels."""
    return [key.name for key in GLOBAL_KERNEL_CACHE.keys()]

def clear_cache():
    """Clear the global kernel cache."""
    GLOBAL_KERNEL_CACHE.clear()

def cache_size() -> int:
    """Return the number of kernels in the cache."""
    return GLOBAL_KERNEL_CACHE.size()
