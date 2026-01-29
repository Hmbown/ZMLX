from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def _dequant_int8_kernel(act_expr: str = "val") -> Any:
    source = f"""
        uint gid = thread_position_in_grid.x;
        float val = (float)inp[gid] * scale[0];
        out[gid] = (T)({act_expr});
    """
    return metal_kernel(
        name=f"kk_dequant_int8_{hash(act_expr) % 10000}",
        input_names=["inp", "scale"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def dequantize_int8(x: Any, scale: Any, *, compute_dtype: Any | None = None) -> Any:
    """Simple int8 dequantization: y = x * scale.
    
    x: int8 array
    scale: float32 scalar (or array of size 1)
    """
    cd = compute_dtype or mx.float32
    k = _dequant_int8_kernel()
    return k(
        x, scale,
        template=[("T", cd)],
        grid=(x.size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[cd],
    )[0]

def dequantize_silu_int8(x: Any, scale: Any, *, compute_dtype: Any | None = None) -> Any:
    """Fused dequantize + SiLU."""
    cd = compute_dtype or mx.float32
    k = _dequant_int8_kernel(act_expr="kk_silu(val)")
    return k(
        x, scale,
        template=[("T", cd)],
        grid=(x.size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[cd],
    )[0]


@cache
def _dequant_int4_kernel() -> Any:
    source = """
        uint gid = thread_position_in_grid.x;
        uint byte_idx = gid / 2;
        uint nibble_idx = gid % 2;
        
        uint8_t packed = inp[byte_idx];
        int8_t val;
        if (nibble_idx == 0) {
            val = (int8_t)(packed & 0x0F);
        } else {
            val = (int8_t)(packed >> 4);
        }
        // Signed 4-bit range is -8 to 7. 
        // If we want simple unsigned 0-15:
        // val = (nibble_idx == 0) ? (packed & 0x0F) : (packed >> 4);
        
        out[gid] = (T)((float)val * scale[0]);
    """
    return metal_kernel(
        name="kk_dequant_int4",
        input_names=["inp", "scale"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def dequantize_int4(x: Any, scale: Any, *, compute_dtype: Any | None = None) -> Any:
    """Dequantize 4-bit data packed into uint8.
    
    x: uint8 array (size is half of output size)
    scale: float32 scalar
    """
    cd = compute_dtype or mx.float32
    k = _dequant_int4_kernel()
    out_size = x.size * 2
    return k(
        x, scale,
        template=[("T", cd)],
        grid=(out_size, 1, 1),
        output_shapes=[(out_size,)],
        output_dtypes=[cd],
    )[0]

__all__ = [
    "dequantize_int8",
    "dequantize_silu_int8",
    "dequantize_int4",
]
