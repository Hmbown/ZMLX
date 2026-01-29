from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def _rope_kernel(d: int, seq_len: int) -> Any:
    D = int(d)
    S = int(seq_len)
    if D % 2 != 0:
        raise ValueError("RoPE requires an even last dimension")
    half = D // 2

    src = f"""
        constexpr uint D = {D};
        constexpr uint HALF = {half};
        constexpr uint S = {S};

        uint elem = thread_position_in_grid.x;
        uint row = elem / D;
        uint col = elem - row * D; // elem % D
        uint pos = row % S;

        uint base = row * D;

        if (col < HALF) {{
            uint j = col;
            float a = (float)inp[base + j];
            float b = (float)inp[base + j + HALF];
            float c = (float)cos[pos * HALF + j];
            float s = (float)sin[pos * HALF + j];
            out[base + j] = (T)(a * c - b * s);
        }} else {{
            uint j = col - HALF;
            float a = (float)inp[base + j];
            float b = (float)inp[base + j + HALF];
            float c = (float)cos[pos * HALF + j];
            float s = (float)sin[pos * HALF + j];
            out[base + col] = (T)(a * s + b * c);
        }}
    """

    return metal_kernel(
        name=f"kk_rope_D{D}_S{S}",
        input_names=["inp", "cos", "sin"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def apply_rope(
    x: Any,
    cos: Any,
    sin: Any,
    *,
    compute_dtype: Any | None = None,
) -> Any:
    """Apply rotary positional embedding over the last dimension.

    Expected shapes:
      - x: (..., S, D)
      - cos: (S, D/2)
      - sin: (S, D/2)

    We assume the second-to-last dimension of `x` is the sequence length `S`.
    """
    if x.ndim < 2:
        raise ValueError("apply_rope: x must have rank >= 2 and include a sequence dimension")
    S = int(x.shape[-2])
    D = int(x.shape[-1])
    if D % 2 != 0:
        raise ValueError("apply_rope: D must be even")
    if int(cos.ndim) != 2 or int(sin.ndim) != 2:
        raise ValueError("apply_rope: cos and sin must be 2D (S, D/2)")
    if int(cos.shape[0]) != S or int(sin.shape[0]) != S:
        raise ValueError(f"apply_rope: cos/sin must have first dim S={S}")
    if int(cos.shape[1]) != D // 2 or int(sin.shape[1]) != D // 2:
        raise ValueError(f"apply_rope: cos/sin must have second dim D/2={D//2}")

    rows = 1
    for s in x.shape[:-1]:
        rows *= int(s)

    cd = compute_dtype or mx.float32
    k = _rope_kernel(D, S)
    out = k(
        x,
        cos,
        sin,
        template=[("T", cd)],
        grid=(rows * D, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
    return out


@cache
def _rope_interleaved_kernel(d: int, seq_len: int) -> Any:
    D = int(d)
    S = int(seq_len)
    half = D // 2

    src = f"""
        constexpr uint D = {D};
        constexpr uint HALF = {half};
        constexpr uint S = {S};

        uint elem = thread_position_in_grid.x;
        uint row = elem / D;
        uint col = elem % D;
        uint pos = row % S;
        uint base = row * D;

        uint pair_idx = col / 2;
        float c = (float)cos[pos * HALF + pair_idx];
        float s = (float)sin[pos * HALF + pair_idx];

        if (col % 2 == 0) {{
            float a = (float)inp[base + col];
            float b = (float)inp[base + col + 1];
            out[base + col] = (T)(a * c - b * s);
        }} else {{
            float a = (float)inp[base + col - 1];
            float b = (float)inp[base + col];
            out[base + col] = (T)(a * s + b * c);
        }}
    """

    return metal_kernel(
        name=f"kk_rope_inter_D{D}_S{S}",
        input_names=["inp", "cos", "sin"],
        output_names=["out"],
        source=src,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def apply_rope_interleaved(
    x: Any,
    cos: Any,
    sin: Any,
    *,
    compute_dtype: Any | None = None,
) -> Any:
    """Apply rotary positional embedding with interleaved layout.
    
    x: (..., S, D)
    cos, sin: (S, D/2)
    
    y[..., 2i] = x[..., 2i] * cos[i] - x[..., 2i+1] * sin[i]
    y[..., 2i+1] = x[..., 2i] * sin[i] + x[..., 2i+1] * cos[i]
    """
    if x.ndim < 2:
        raise ValueError("apply_rope_interleaved: x must have rank >= 2")
    S = int(x.shape[-2])
    D = int(x.shape[-1])
    if D % 2 != 0:
        raise ValueError("apply_rope_interleaved: D must be even")
    
    rows = x.size // D
    cd = compute_dtype or mx.float32
    k = _rope_interleaved_kernel(D, S)
    return k(
        x, cos, sin,
        template=[("T", cd)],
        grid=(rows * D, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _gqa_rope_kernel(d: int, s: int, n_heads: int, n_kv_heads: int) -> Any:
    D = int(d)
    S = int(s)
    H = int(n_heads)
    HKV = int(n_kv_heads)
    G = H // HKV # group size
    half = D // 2

    source = f"""
        constexpr uint D = {D};
        constexpr uint HALF = {half};
        constexpr uint S = {S};
        constexpr uint G = {G};

        uint gid = thread_position_in_grid.x;
        uint col = gid % D;
        uint head = (gid / D) % {H};
        uint pos = (gid / D / {H}) % S;
        uint batch = gid / D / {H} / S;

        uint kv_head = head / G;
        uint base = (((batch * S + pos) * {H}) + head) * D;

        if (col < HALF) {{
            uint j = col;
            float a = (float)inp[base + j];
            float b = (float)inp[base + j + HALF];
            float c = (float)cos[pos * HALF + j];
            float s = (float)sin[pos * HALF + j];
            out[base + j] = (T)(a * c - b * s);
        }} else {{
            uint j = col - HALF;
            float a = (float)inp[base + j];
            float b = (float)inp[base + j + HALF];
            float c = (float)cos[pos * HALF + j];
            float s = (float)sin[pos * HALF + j];
            out[base + col] = (T)(a * s + b * c);
        }}
    """
    return metal_kernel(
        name=f"kk_gqa_rope_D{D}_S{S}_H{H}_HKV{HKV}",
        input_names=["inp", "cos", "sin"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def apply_gqa_rope(
    x: Any,
    cos: Any,
    sin: Any,
    *,
    n_kv_heads: int,
    compute_dtype: Any | None = None,
) -> Any:
    """Apply RoPE for Grouped Query Attention.
    
    x: (B, S, H, D)
    cos, sin: (S, D/2)
    n_kv_heads: number of KV heads (H must be a multiple)
    """
    B, S, H, D = x.shape
    if H % n_kv_heads != 0:
        raise ValueError("H must be a multiple of n_kv_heads")
    
    cd = compute_dtype or mx.float32
    k = _gqa_rope_kernel(D, S, H, n_kv_heads)
    
    return k(
        x, cos, sin,
        template=[("T", cd)],
        grid=(x.size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]

__all__ = [
    "apply_rope",
    "apply_rope_interleaved",
    "apply_gqa_rope",
]
