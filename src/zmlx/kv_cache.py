"""KV cache quantization helpers (opt-in).

These helpers surface mlx-lm's quantized KV cache support behind simple
environment variables or explicit kwargs so ZMLX tools can opt into
bandwidth-saving decode paths without changing defaults.
"""

from __future__ import annotations

from typing import Any
import os


def _parse_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}; expected integer.") from exc


def kv_cache_kwargs(
    *,
    kv_bits: int | None = None,
    kv_group_size: int | None = None,
    quantized_kv_start: int | None = None,
) -> dict[str, Any]:
    """Return mlx-lm generate kwargs for quantized KV cache.

    If ``kv_bits`` is None, will read from env vars:
      - ZMLX_KV_BITS
      - ZMLX_KV_GROUP_SIZE (default 64 when kv_bits is set)
      - ZMLX_QUANTIZED_KV_START (default 0 when kv_bits is set)
    """
    bits = kv_bits
    if bits is None:
        bits = _parse_int_env("ZMLX_KV_BITS")

    if bits is None:
        return {}

    if bits <= 0:
        raise ValueError(f"kv_bits must be > 0 (got {bits}).")

    group = kv_group_size
    if group is None:
        group = _parse_int_env("ZMLX_KV_GROUP_SIZE")
    if group is None:
        group = 64

    start = quantized_kv_start
    if start is None:
        start = _parse_int_env("ZMLX_QUANTIZED_KV_START")
    if start is None:
        start = 0

    return {
        "kv_bits": int(bits),
        "kv_group_size": int(group),
        "quantized_kv_start": int(start),
    }

