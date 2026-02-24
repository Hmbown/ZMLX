from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any


def _shape_key(shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(d) for d in shape)


def _dtype_key(dtype: Any) -> str:
    # MLX dtype objects stringify deterministically (e.g. "float16").
    return str(dtype)


@dataclass(frozen=True)
class FusionCompileKey:
    """Key for fused elementwise compile cache entries."""

    op_signature: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]

    @classmethod
    def from_parts(
        cls,
        *,
        op_signature: str,
        input_shapes: Sequence[Sequence[int]],
        input_dtypes: Sequence[Any],
    ) -> FusionCompileKey:
        return cls(
            op_signature=str(op_signature),
            input_shapes=tuple(_shape_key(shape) for shape in input_shapes),
            input_dtypes=tuple(_dtype_key(dtype) for dtype in input_dtypes),
        )


class FusionCompileCache:
    """Thread-safe in-process LRU cache for compiled fused kernels."""

    def __init__(self, max_entries: int = 1000) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[FusionCompileKey, Any] = OrderedDict()
        self._lock = RLock()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def set_max_entries(self, max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        with self._lock:
            self._max_entries = int(max_entries)
            self._evict_if_needed()

    def get(self, key: FusionCompileKey) -> Any | None:
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                return None
            self._entries.move_to_end(key)
            return value

    def put(self, key: FusionCompileKey, value: Any) -> Any:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            self._evict_if_needed()
            return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def keys(self) -> list[FusionCompileKey]:
        with self._lock:
            return list(self._entries.keys())

    def __len__(self) -> int:
        return self.size()

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


GLOBAL_FUSION_COMPILE_CACHE = FusionCompileCache(max_entries=1000)


__all__ = [
    "FusionCompileKey",
    "FusionCompileCache",
    "GLOBAL_FUSION_COMPILE_CACHE",
]
