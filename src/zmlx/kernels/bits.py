from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def _pack_bits_kernel() -> Any:
    source = """
        uint gid = thread_position_in_grid.x;
        uint base = gid * 8;
        uint8_t res = 0;
        for (uint i = 0; i < 8; ++i) {
            if (inp[base + i]) {
                res |= (1 << i);
            }
        }
        out[gid] = res;
    """
    return metal_kernel(
        name="kk_pack_bits",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

@cache
def _unpack_bits_kernel() -> Any:
    source = """
        uint gid = thread_position_in_grid.x;
        uint byte_idx = gid / 8;
        uint bit_idx = gid % 8;
        uint8_t val = inp[byte_idx];
        out[gid] = (val & (1 << bit_idx)) ? 1 : 0;
    """
    return metal_kernel(
        name="kk_unpack_bits",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def pack_bits(x: Any) -> Any:
    """Pack a boolean (or uint8) array into uint8 bits.

    Args:
        x: Input array with size divisible by 8.

    Returns:
        A uint8 array with size ``x.size // 8``.
    """
    if x.size % 8 != 0:
        raise ValueError("pack_bits: size must be a multiple of 8")
    
    k = _pack_bits_kernel()
    return k(
        x,
        template=[("T", mx.uint8)],
        grid=(x.size // 8, 1, 1),
        output_shapes=[(x.size // 8,)],
        output_dtypes=[mx.uint8],
    )[0]

def unpack_bits(x: Any, original_shape: tuple[int, ...] | None = None) -> Any:
    """Unpack uint8 bits into a byte array.

    Args:
        x: Packed uint8 array.
        original_shape: Optional output shape to reshape to.

    Returns:
        An unpacked uint8 array with size ``x.size * 8`` or reshaped to
        ``original_shape``.
    """
    k = _unpack_bits_kernel()
    out_size = x.size * 8
    res = k(
        x,
        template=[("T", mx.uint8)],
        grid=(out_size, 1, 1),
        output_shapes=[(out_size,)],
        output_dtypes=[mx.uint8],
    )[0]
    if original_shape:
        return res.reshape(original_shape)
    return res

__all__ = [
    "pack_bits",
    "unpack_bits",
]
