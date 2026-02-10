"""Adapters for extracting K/V arrays from various cache representations.

These bridge between mlx-lm's ``KVCache`` objects and the codec's
list-of-numpy-arrays interface.
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "KVTC requires numpy. Install with: pip install zmlx[kvtc]"
    ) from e

from .utils import to_numpy


class MLXLMCacheAdapter:
    """Extract K/V numpy arrays from an mlx-lm style KVCache list.

    mlx-lm caches are typically a list of objects with ``.keys`` and
    ``.values`` attributes, each an ``mx.array`` shaped
    ``(batch, heads, seq, head_dim)``.
    """

    def extract_kv(
        self, cache: list[Any]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (k_layers, v_layers) as lists of numpy arrays."""
        k_layers = []
        v_layers = []
        for layer_cache in cache:
            k = getattr(layer_cache, "keys", None)
            v = getattr(layer_cache, "values", None)
            if k is None or v is None:
                raise ValueError(
                    f"Cache layer {type(layer_cache).__name__} missing .keys or .values"
                )
            k_layers.append(to_numpy(k))
            v_layers.append(to_numpy(v))
        return k_layers, v_layers

    def inject_kv(
        self, cache: list[Any],
        k_layers: list[np.ndarray], v_layers: list[np.ndarray],
    ) -> None:
        """Write K/V numpy arrays back into the cache objects.

        Attempts to use ``mx.array`` conversion if MLX is available,
        otherwise stores raw numpy arrays.
        """
        from .utils import maybe_to_mlx

        for i, layer_cache in enumerate(cache):
            layer_cache.keys = maybe_to_mlx(k_layers[i], dtype="float16")
            layer_cache.values = maybe_to_mlx(v_layers[i], dtype="float16")


class GLMMlaCacheAdapter:
    """Extract keys-only cache for GLM MLA models.

    GLM stores ``[kv_latent(512) | k_pe(64)]`` in the keys slot.
    The values slot is empty (dim=0) — values are derived at attention
    time from the latent portion.

    Returns (k_layers, v_layers) where v_layers are zero-dim arrays
    matching the expected shape for single_stream mode.
    """

    def extract_kv(
        self, cache: list[Any]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (k_layers, v_layers) where v_layers have head_dim=0."""
        k_layers = []
        v_layers = []
        for layer_cache in cache:
            k = getattr(layer_cache, "keys", None)
            if k is None:
                raise ValueError(
                    f"Cache layer {type(layer_cache).__name__} missing .keys"
                )
            k_np = to_numpy(k)
            k_layers.append(k_np)
            # Empty values: same shape as keys but head_dim=0
            shape = list(k_np.shape)
            shape[-1] = 0
            v_layers.append(np.zeros(shape, dtype=np.float16))
        return k_layers, v_layers

    def inject_kv(
        self, cache: list[Any],
        k_layers: list[np.ndarray], v_layers: list[np.ndarray],
    ) -> None:
        """Write keys back into the cache objects (values are ignored)."""
        from .utils import maybe_to_mlx

        for i, layer_cache in enumerate(cache):
            layer_cache.keys = maybe_to_mlx(k_layers[i], dtype="float16")
