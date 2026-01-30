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

@cache
def _rope_cache_update_kernel(d: int, n_heads: int, n_kv_heads: int) -> Any:
    D = int(d)
    H = int(n_heads)
    HKV = int(n_kv_heads)
    G = H // HKV
    half = D // 2

    source = f"""
        constexpr uint D = {D};
        constexpr uint HALF = {half};
        constexpr uint H = {H};
        constexpr uint HKV = {HKV};
        constexpr uint G = {G};

        uint col = thread_position_in_grid.x;
        uint head = thread_position_in_grid.y;
        uint batch = thread_position_in_grid.z;

        uint pos = (uint)offset[batch];
        
        // RoPE for Q
        if (head < H) {{
            uint q_idx = ((batch * H + head) * D) + col;
            if (col < HALF) {{
                float a = (float)q[q_idx];
                float b = (float)q[q_idx + HALF];
                float c = (float)cos[pos * HALF + col];
                float s = (float)sin[pos * HALF + col];
                q_out[q_idx] = (T)(a * c - b * s);
                q_out[q_idx + HALF] = (T)(a * s + b * c);
            }}
        }}

        // RoPE for K and write to cache
        if (head < HKV) {{
            uint k_idx = ((batch * HKV + head) * D) + col;
            if (col < HALF) {{
                float a = (float)k[k_idx];
                float b = (float)k[k_idx + HALF];
                float c = (float)cos[pos * HALF + col];
                float s = (float)sin[pos * HALF + col];
                float k_rope_a = a * c - b * s;
                float k_rope_b = a * s + b * c;
                
                // Write to cache (contiguous for now)
                // Cache shape: (B, MAX_SEQ, HKV, D)
                uint cache_idx_a = (((batch * max_seq[0] + pos) * HKV + head) * D) + col;
                k_cache[cache_idx_a] = (T)k_rope_a;
                k_cache[cache_idx_a + HALF] = (T)k_rope_b;
                
                // V cache
                uint v_idx = ((batch * HKV + head) * D) + col;
                float v_val_a = (float)v[v_idx];
                float v_val_b = (float)v[v_idx + HALF];
                uint v_cache_idx_a = (((batch * max_seq[0] + pos) * HKV + head) * D) + col;
                v_cache[v_cache_idx_a] = (T)v_val_a;
                v_cache[v_cache_idx_a + HALF] = (T)v_val_b;
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_rope_cache_update_D{D}_H{H}_HKV{HKV}",
        input_names=["q", "k", "v", "cos", "sin", "k_cache", "v_cache", "offset", "max_seq"],
        output_names=["q_out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def rope_and_cache_update(
    q: Any,
    k: Any,
    v: Any,
    cos: Any,
    sin: Any,
    k_cache: Any,
    v_cache: Any,
    offset: Any,
) -> Any:
    """Fused RoPE + KV cache update.
    
    q: (B, H, D)
    k, v: (B, HKV, D)
    cos, sin: (MAX_SEQ, D/2)
    k_cache, v_cache: (B, MAX_SEQ, HKV, D)
    offset: (B,) - current position for each batch
    
    Returns new_q (RoPE applied).
    k_cache and v_cache are updated in-place (if MLX allows) 
    or we return them if we had used output_names.
    Wait, MLX doesn't support true in-place in custom kernels easily unless
    the input is also an output.
    """
    B, H, D = q.shape
    _, HKV, _ = k.shape
    MAX_SEQ = k_cache.shape[1]
    
    # We need k_cache and v_cache to be outputs to actually update them
    # OR we use atomic/unsafe writes if we really want in-place.
    # But for ZMLX, we should probably follow the pattern of returning them.
    
    k_op = _rope_cache_update_kernel_v2(D, H, HKV)
    
    max_seq_arr = mx.array([MAX_SEQ], dtype=mx.int32)
    
    res = k_op(
        q, k, v, cos, sin, k_cache, v_cache, offset, max_seq_arr,
        template=[("T", q.dtype)],
        grid=(D // 2, max(H, HKV), B),
        threadgroup=(min(D // 2, 256), 1, 1),
        output_shapes=[q.shape, k_cache.shape, v_cache.shape],
        output_dtypes=[q.dtype, k_cache.dtype, v_cache.dtype],
    )
    return res[0], res[1], res[2]


@cache
def _rope_cache_update_kernel_v2(d: int, n_heads: int, n_kv_heads: int) -> Any:
    D = int(d)
    H = int(n_heads)
    HKV = int(n_kv_heads)
    half = D // 2

    source = f"""
        constexpr uint D = {D};
        constexpr uint HALF = {half};
        constexpr uint H = {H};
        constexpr uint HKV = {HKV};

        uint col = thread_position_in_grid.x;
        uint head = thread_position_in_grid.y;
        uint batch = thread_position_in_grid.z;

        uint pos = (uint)offset[batch];
        uint MS = (uint)max_seq[0];
        
        // RoPE for Q
        if (head < H) {{
            uint q_idx = ((batch * H + head) * D) + col;
            float a = (float)q[q_idx];
            float b = (float)q[q_idx + HALF];
            float c = (float)cos[pos * HALF + col];
            float s = (float)sin[pos * HALF + col];
            q_out[q_idx] = (T)(a * c - b * s);
            q_out[q_idx + HALF] = (T)(a * s + b * c);
        }}

        // RoPE for K and write to cache
        if (head < HKV) {{
            uint k_idx = ((batch * HKV + head) * D) + col;
            float a = (float)k[k_idx];
            float b = (float)k[k_idx + HALF];
            float c = (float)cos[pos * HALF + col];
            float s = (float)sin[pos * HALF + col];
            float k_rope_a = a * c - b * s;
            float k_rope_b = a * s + b * c;
            
            uint cache_off = ((batch * MS + pos) * HKV + head) * D + col;
            k_cache_out[cache_off] = (T)k_rope_a;
            k_cache_out[cache_off + HALF] = (T)k_rope_b;
            
            uint v_idx = ((batch * HKV + head) * D) + col;
            v_cache_out[cache_off] = v[v_idx];
            v_cache_out[cache_off + HALF] = v[v_idx + HALF];
        }}
    """
    return metal_kernel(
        name=f"kk_rope_cache_update_v2_D{D}_H{H}_HKV{HKV}",
        input_names=["q", "k", "v", "cos", "sin", "k_cache", "v_cache", "offset", "max_seq"],
        output_names=["q_out", "k_cache_out", "v_cache_out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

@cache
def _paged_rope_cache_update_kernel(d: int, n_heads: int, n_kv_heads: int, block_size: int, max_blocks: int) -> Any:
    D = int(d)
    H = int(n_heads)
    HKV = int(n_kv_heads)
    BS = int(block_size)
    MB = int(max_blocks)
    half = D // 2

    source = f"""
        constexpr uint D = {D};
        constexpr uint HALF = {half};
        constexpr uint H = {H};
        constexpr uint HKV = {HKV};
        constexpr uint BS = {BS};
        constexpr uint MB = {MB};

        uint col = thread_position_in_grid.x;
        uint head = thread_position_in_grid.y;
        uint batch = thread_position_in_grid.z;

        uint pos = (uint)offset[batch];
        
        // RoPE for Q
        if (head < H) {{
            uint q_idx = ((batch * H + head) * D) + col;
            float a = (float)q[q_idx];
            float b = (float)q[q_idx + HALF];
            float c = (float)cos[pos * HALF + col];
            float s = (float)sin[pos * HALF + col];
            q_out[q_idx] = (T)(a * c - b * s);
            q_out[q_idx + HALF] = (T)(a * s + b * c);
        }}

        // RoPE for K and write to paged cache
        if (head < HKV) {{
            uint block_logical_idx = pos / BS;
            uint token_block_idx = pos % BS;
            uint physical_block = (uint)block_table[batch * MB + block_logical_idx];
            
            uint k_idx = ((batch * HKV + head) * D) + col;
            float a = (float)k[k_idx];
            float b = (float)k[k_idx + HALF];
            float c = (float)cos[pos * HALF + col];
            float s = (float)sin[pos * HALF + col];
            float k_rope_a = a * c - b * s;
            float k_rope_b = a * s + b * c;
            
            uint cache_off = ((physical_block * BS + token_block_idx) * HKV + head) * D + col;
            k_cache_out[cache_off] = (T)k_rope_a;
            k_cache_out[cache_off + HALF] = (T)k_rope_b;
            
            uint v_idx = ((batch * HKV + head) * D) + col;
            v_cache_out[cache_off] = v[v_idx];
            v_cache_out[cache_off + HALF] = v[v_idx + HALF];
        }}
    """
    return metal_kernel(
        name=f"kk_paged_rope_cache_update_D{D}_H{H}_HKV{HKV}_BS{BS}",
        input_names=["q", "k", "v", "cos", "sin", "k_cache", "v_cache", "offset", "block_table"],
        output_names=["q_out", "k_cache_out", "v_cache_out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def paged_rope_and_cache_update(
    q: Any,
    k: Any,
    v: Any,
    cos: Any,
    sin: Any,
    k_cache: Any,
    v_cache: Any,
    offset: Any,
    block_table: Any,
) -> Any:
    """Paged version of Fused RoPE + KV cache update."""
    B, H, D = q.shape
    _, HKV, _ = k.shape
    _, BS, _, _ = k_cache.shape
    MB = block_table.shape[1]
    
    k_op = _paged_rope_cache_update_kernel(D, H, HKV, BS, MB)
    
    res = k_op(
        q, k, v, cos, sin, k_cache, v_cache, offset, block_table,
        template=[("T", q.dtype)],
        grid=(D // 2, max(H, HKV), B),
        threadgroup=(min(D // 2, 256), 1, 1),
        output_shapes=[q.shape, k_cache.shape, v_cache.shape],
        output_dtypes=[q.dtype, k_cache.dtype, v_cache.dtype],
    )
    return res[0], res[1], res[2]


__all__ = [
    "apply_rope",
    "apply_rope_interleaved",
    "apply_gqa_rope",
    "rope_and_cache_update",
    "paged_rope_and_cache_update",
]
