from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@dataclass(frozen=True)
class RoPECosSin:
    cos: mx.array
    sin: mx.array


@cache
def _rope_cos_sin(max_seq: int, dims: int, base: float, scale: float) -> RoPECosSin:
    """Precompute RoPE cos/sin tables for fast custom kernels.

    Matches `mx.fast.rope` exactly by extracting cos/sin from a basis input.

    NOTE: `mx.fast.rope` uses a slightly different sin/cos implementation than
    `mx.sin`/`mx.cos` (likely fast-math approximations). For float32 models
    like GLM-4.7-Flash, using `mx.sin`/`mx.cos` can introduce tiny numerical
    differences that break greedy token fidelity. Deriving the tables from
    `mx.fast.rope` preserves bit-identical behavior.
    """
    D = int(dims)
    if D % 2 != 0:
        raise ValueError("_rope_cos_sin: dims must be even")
    S = int(max_seq)
    if S <= 0:
        raise ValueError("_rope_cos_sin: max_seq must be > 0")

    # Construct a basis where each RoPE pair is (1, 0). Under traditional RoPE,
    # rotation yields (cos, sin) for each pair, so we can read them out.
    basis = mx.zeros((1, 1, S, D), dtype=mx.float32)
    basis[..., ::2] = mx.array(1.0, dtype=mx.float32)
    rotated = mx.fast.rope(
        basis,
        D,
        traditional=True,
        base=float(base),
        scale=float(scale),
        offset=0,
    )
    cos = rotated[0, 0, :, ::2]
    sin = rotated[0, 0, :, 1::2]
    mx.eval(cos, sin)
    return RoPECosSin(cos=cos, sin=sin)


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

    # NOTE: On bfloat16 outputs, casting float -> bfloat16_t is not implicitly
    # supported in Metal; use T=x.dtype to ensure correct compilation.
    # `compute_dtype` is kept for API compatibility but doesn't affect compute
    # (we compute in float internally).
    template_t = x.dtype
    if compute_dtype is not None and x.dtype != mx.bfloat16:
        template_t = compute_dtype
    k = _rope_kernel(D, S)
    out = k(
        x,
        cos,
        sin,
        template=[("T", template_t)],
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
    template_t = x.dtype
    if compute_dtype is not None and x.dtype != mx.bfloat16:
        template_t = compute_dtype
    k = _rope_interleaved_kernel(D, S)
    return k(
        x, cos, sin,
        template=[("T", template_t)],
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

        int p = (int)offset[batch];
        bool neg = (p < 0);
        uint pos = (uint)(neg ? -p : p);
        
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

        int p = (int)offset[batch];
        bool neg = (p < 0);
        uint pos = (uint)(neg ? -p : p);
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


# ---------------------------------------------------------------------------
# GLM-4.7-Flash (glm4_moe_lite): decode-only fused RoPE + concat
# ---------------------------------------------------------------------------

@cache
def _rope_concat_qk_decode_kernel(d_nope: int, d_rope: int, n_q_heads: int) -> Any:
    """Build (queries, keys) with RoPE applied on the rope-slice (decode, T=1).

    Outputs:
      - queries: (B, Hq, 1, D_nope + D_rope)
      - keys:    (B, 1,  1, D_nope + D_rope)

    Assumes interleaved RoPE layout (traditional=True in `mx.fast.rope`).
    """
    D_NOPE = int(d_nope)
    D_ROPE = int(d_rope)
    H_Q = int(n_q_heads)
    if D_NOPE <= 0 or D_ROPE <= 0:
        raise ValueError("_rope_concat_qk_decode_kernel: dims must be > 0")
    if D_ROPE % 2 != 0:
        raise ValueError("_rope_concat_qk_decode_kernel: d_rope must be even")
    if H_Q <= 0:
        raise ValueError("_rope_concat_qk_decode_kernel: n_q_heads must be > 0")

    D_OUT = D_NOPE + D_ROPE
    half = D_ROPE // 2

    source = f"""
        constexpr uint D_NOPE = {D_NOPE};
        constexpr uint D_ROPE = {D_ROPE};
        constexpr uint HALF = {half};
        constexpr uint D_OUT = {D_OUT};
        constexpr uint H_Q = {H_Q};

        // Per-batch element counts (T == 1)
        constexpr uint Q_ELEMS_PER_BATCH = H_Q * D_OUT;
        constexpr uint K_ELEMS_PER_BATCH = D_OUT; // H_K == 1
        constexpr uint ELEMS_PER_BATCH = Q_ELEMS_PER_BATCH + K_ELEMS_PER_BATCH;

        uint gid = thread_position_in_grid.x;
        uint batch = gid / ELEMS_PER_BATCH;
        uint in_batch = gid - batch * ELEMS_PER_BATCH;

        int p = (int)offset[batch];
        bool neg = (p < 0);
        uint pos = (uint)(neg ? -p : p);

        if (in_batch < Q_ELEMS_PER_BATCH) {{
            // queries: (B, H_Q, 1, D_OUT)
            uint head = in_batch / D_OUT;
            uint col = in_batch - head * D_OUT;

            uint q_out_base = (batch * H_Q + head) * D_OUT;

            if (col < D_NOPE) {{
                uint q_nope_base = (batch * H_Q + head) * D_NOPE;
                q_out[q_out_base + col] = q_nope[q_nope_base + col];
                return;
            }}

            // RoPE slice
            uint r = col - D_NOPE;
            uint pair = r / 2;
            float c = (float)cos[pos * HALF + pair];
            float s = (float)sin[pos * HALF + pair];
            if (neg) s = -s;

            uint q_rope_base = (batch * H_Q + head) * D_ROPE;
            if ((r & 1u) == 0u) {{
                float a = (float)q_rope[q_rope_base + r];
                float b = (float)q_rope[q_rope_base + r + 1];
                q_out[q_out_base + col] = (T)(a * c - b * s);
            }} else {{
                float a = (float)q_rope[q_rope_base + r - 1];
                float b = (float)q_rope[q_rope_base + r];
                q_out[q_out_base + col] = (T)(a * s + b * c);
            }}
            return;
        }}

        // keys: (B, 1, 1, D_OUT)
        uint k_col = in_batch - Q_ELEMS_PER_BATCH;
        uint k_out_base = batch * D_OUT;
        if (k_col < D_NOPE) {{
            uint kv_nope_base = batch * D_NOPE;
            k_out[k_out_base + k_col] = kv_nope[kv_nope_base + k_col];
            return;
        }}

        uint kr = k_col - D_NOPE;
        uint kpair = kr / 2;
        float kc = (float)cos[pos * HALF + kpair];
        float ks = (float)sin[pos * HALF + kpair];
        if (neg) ks = -ks;
        uint k_rope_base = batch * D_ROPE;
        if ((kr & 1u) == 0u) {{
            float a = (float)k_rope[k_rope_base + kr];
            float b = (float)k_rope[k_rope_base + kr + 1];
            k_out[k_out_base + k_col] = (T)(a * kc - b * ks);
        }} else {{
            float a = (float)k_rope[k_rope_base + kr - 1];
            float b = (float)k_rope[k_rope_base + kr];
            k_out[k_out_base + k_col] = (T)(a * ks + b * kc);
        }}
    """
    return metal_kernel(
        name=f"kk_rope_concat_qk_decode_dn{D_NOPE}_dr{D_ROPE}_hq{H_Q}",
        input_names=["q_nope", "q_rope", "kv_nope", "k_rope", "cos", "sin", "offset"],
        output_names=["q_out", "k_out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def rope_concat_qk_decode(
    q_nope: Any,
    q_rope: Any,
    kv_nope: Any,
    k_rope: Any,
    cos: Any,
    sin: Any,
    offset: Any,
) -> tuple[Any, Any]:
    """Decode-only fused helper to build Q/K with RoPE applied and concatenated.

    Intended for GLM-4.7-Flash (glm4_moe_lite), where decode uses `T=1` and
    RoPE is applied to a 64-dim slice (`traditional=True`).
    """
    if q_nope.ndim != 4 or q_rope.ndim != 4:
        raise ValueError("rope_concat_qk_decode: q_nope and q_rope must be 4D (B,H,1,D)")
    if kv_nope.ndim != 4 or k_rope.ndim != 4:
        raise ValueError("rope_concat_qk_decode: kv_nope and k_rope must be 4D (B,1,1,D)")
    if int(q_nope.shape[2]) != 1 or int(q_rope.shape[2]) != 1:
        raise ValueError("rope_concat_qk_decode: requires T=1 (decode)")
    if int(kv_nope.shape[2]) != 1 or int(k_rope.shape[2]) != 1:
        raise ValueError("rope_concat_qk_decode: requires T=1 (decode)")

    B = int(q_nope.shape[0])
    Hq = int(q_nope.shape[1])
    if int(q_rope.shape[0]) != B or int(q_rope.shape[1]) != Hq:
        raise ValueError("rope_concat_qk_decode: q_rope must match q_nope batch/head dims")
    if int(kv_nope.shape[0]) != B or int(k_rope.shape[0]) != B:
        raise ValueError("rope_concat_qk_decode: kv_nope/k_rope must match batch size")
    if int(kv_nope.shape[1]) != 1 or int(k_rope.shape[1]) != 1:
        raise ValueError("rope_concat_qk_decode: kv_nope/k_rope must have 1 KV head")

    d_nope = int(q_nope.shape[-1])
    d_rope = int(q_rope.shape[-1])
    if int(kv_nope.shape[-1]) != d_nope or int(k_rope.shape[-1]) != d_rope:
        raise ValueError("rope_concat_qk_decode: K/V dims must match Q dims")
    if d_rope % 2 != 0:
        raise ValueError("rope_concat_qk_decode: d_rope must be even")
    if int(cos.ndim) != 2 or int(sin.ndim) != 2:
        raise ValueError("rope_concat_qk_decode: cos/sin must be 2D (S, d_rope/2)")
    if int(cos.shape[1]) != d_rope // 2 or int(sin.shape[1]) != d_rope // 2:
        raise ValueError("rope_concat_qk_decode: cos/sin second dim must be d_rope/2")

    # Normalize offsets: allow int, scalar array, or (B,) vector.
    if not isinstance(offset, mx.array):
        offs = mx.full((B,), int(offset), dtype=mx.int32)
    else:
        offs = offset
        if int(offs.ndim) == 0:
            offs = mx.broadcast_to(offs, (B,))
        if int(offs.ndim) != 1 or int(offs.shape[0]) != B:
            raise ValueError("rope_concat_qk_decode: offset must be scalar or (B,) array")
        if offs.dtype != mx.int32:
            offs = offs.astype(mx.int32)

    D_OUT = d_nope + d_rope
    k = _rope_concat_qk_decode_kernel(d_nope, d_rope, Hq)
    q_out_shape = (B, Hq, 1, D_OUT)
    k_out_shape = (B, 1, 1, D_OUT)
    total_threads = B * (Hq + 1) * D_OUT

    q_out, k_out = k(
        q_nope,
        q_rope,
        kv_nope,
        k_rope,
        cos,
        sin,
        offs,
        template=[("T", q_nope.dtype)],
        grid=(total_threads, 1, 1),
        output_shapes=[q_out_shape, k_out_shape],
        output_dtypes=[q_nope.dtype, q_nope.dtype],
    )
    return q_out, k_out


@cache
def _rope_concat_qk_decode_pos_kernel(d_nope: int, d_rope: int, n_q_heads: int) -> Any:
    """Same as `_rope_concat_qk_decode_kernel`, but uses a single cos/sin row.

    Inputs:
      - cos, sin: (D_rope/2,) for the current position (already offset-applied)

    This avoids allocating an `(B,)` offset array per decode step.
    """
    D_NOPE = int(d_nope)
    D_ROPE = int(d_rope)
    H_Q = int(n_q_heads)
    if D_NOPE <= 0 or D_ROPE <= 0:
        raise ValueError("_rope_concat_qk_decode_pos_kernel: dims must be > 0")
    if D_ROPE % 2 != 0:
        raise ValueError("_rope_concat_qk_decode_pos_kernel: d_rope must be even")
    if H_Q <= 0:
        raise ValueError("_rope_concat_qk_decode_pos_kernel: n_q_heads must be > 0")

    D_OUT = D_NOPE + D_ROPE
    half = D_ROPE // 2

    source = f"""
        constexpr uint D_NOPE = {D_NOPE};
        constexpr uint D_ROPE = {D_ROPE};
        constexpr uint HALF = {half};
        constexpr uint D_OUT = {D_OUT};
        constexpr uint H_Q = {H_Q};

        // Per-batch element counts (T == 1)
        constexpr uint Q_ELEMS_PER_BATCH = H_Q * D_OUT;
        constexpr uint K_ELEMS_PER_BATCH = D_OUT; // H_K == 1
        constexpr uint ELEMS_PER_BATCH = Q_ELEMS_PER_BATCH + K_ELEMS_PER_BATCH;

        uint gid = thread_position_in_grid.x;
        uint batch = gid / ELEMS_PER_BATCH;
        uint in_batch = gid - batch * ELEMS_PER_BATCH;

        if (in_batch < Q_ELEMS_PER_BATCH) {{
            // queries: (B, H_Q, 1, D_OUT)
            uint head = in_batch / D_OUT;
            uint col = in_batch - head * D_OUT;

            uint q_out_base = (batch * H_Q + head) * D_OUT;

            if (col < D_NOPE) {{
                uint q_nope_base = (batch * H_Q + head) * D_NOPE;
                q_out[q_out_base + col] = q_nope[q_nope_base + col];
                return;
            }}

            // RoPE slice (traditional/interleaved)
            uint r = col - D_NOPE;
            uint pair = r / 2;
            float c = (float)cos[pair];
            float s = (float)sin[pair];

            uint q_rope_base = (batch * H_Q + head) * D_ROPE;
            if ((r & 1u) == 0u) {{
                float a = (float)q_rope[q_rope_base + r];
                float b = (float)q_rope[q_rope_base + r + 1];
                q_out[q_out_base + col] = (T)(a * c - b * s);
            }} else {{
                float a = (float)q_rope[q_rope_base + r - 1];
                float b = (float)q_rope[q_rope_base + r];
                q_out[q_out_base + col] = (T)(a * s + b * c);
            }}
            return;
        }}

        // keys: (B, 1, 1, D_OUT)
        uint k_col = in_batch - Q_ELEMS_PER_BATCH;
        uint k_out_base = batch * D_OUT;
        if (k_col < D_NOPE) {{
            uint kv_nope_base = batch * D_NOPE;
            k_out[k_out_base + k_col] = kv_nope[kv_nope_base + k_col];
            return;
        }}

        uint kr = k_col - D_NOPE;
        uint kpair = kr / 2;
        float kc = (float)cos[kpair];
        float ks = (float)sin[kpair];
        uint k_rope_base = batch * D_ROPE;
        if ((kr & 1u) == 0u) {{
            float a = (float)k_rope[k_rope_base + kr];
            float b = (float)k_rope[k_rope_base + kr + 1];
            k_out[k_out_base + k_col] = (T)(a * kc - b * ks);
        }} else {{
            float a = (float)k_rope[k_rope_base + kr - 1];
            float b = (float)k_rope[k_rope_base + kr];
            k_out[k_out_base + k_col] = (T)(a * ks + b * kc);
        }}
    """
    return metal_kernel(
        name=f"kk_rope_concat_qk_decode_pos_dn{D_NOPE}_dr{D_ROPE}_hq{H_Q}",
        input_names=["q_nope", "q_rope", "kv_nope", "k_rope", "cos", "sin"],
        output_names=["q_out", "k_out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def rope_concat_qk_decode_pos(
    q_nope: Any,
    q_rope: Any,
    kv_nope: Any,
    k_rope: Any,
    cos: Any,
    sin: Any,
) -> tuple[Any, Any]:
    """Decode-only fused helper with a pre-selected (cos, sin) row.

    Shapes:
      - q_nope, q_rope: (B, Hq, 1, D)
      - kv_nope, k_rope: (B, 1, 1, D)
      - cos, sin: (D_rope/2,)
    """
    if q_nope.ndim != 4 or q_rope.ndim != 4:
        raise ValueError(
            "rope_concat_qk_decode_pos: q_nope and q_rope must be 4D (B,H,1,D)"
        )
    if kv_nope.ndim != 4 or k_rope.ndim != 4:
        raise ValueError(
            "rope_concat_qk_decode_pos: kv_nope and k_rope must be 4D (B,1,1,D)"
        )
    if int(q_nope.shape[2]) != 1 or int(q_rope.shape[2]) != 1:
        raise ValueError("rope_concat_qk_decode_pos: requires T=1 (decode)")
    if int(kv_nope.shape[2]) != 1 or int(k_rope.shape[2]) != 1:
        raise ValueError("rope_concat_qk_decode_pos: requires T=1 (decode)")

    B = int(q_nope.shape[0])
    Hq = int(q_nope.shape[1])
    if int(q_rope.shape[0]) != B or int(q_rope.shape[1]) != Hq:
        raise ValueError(
            "rope_concat_qk_decode_pos: q_rope must match q_nope batch/head dims"
        )
    if int(kv_nope.shape[0]) != B or int(k_rope.shape[0]) != B:
        raise ValueError("rope_concat_qk_decode_pos: kv_nope/k_rope must match batch size")
    if int(kv_nope.shape[1]) != 1 or int(k_rope.shape[1]) != 1:
        raise ValueError("rope_concat_qk_decode_pos: kv_nope/k_rope must have 1 KV head")

    d_nope = int(q_nope.shape[-1])
    d_rope = int(q_rope.shape[-1])
    if int(kv_nope.shape[-1]) != d_nope or int(k_rope.shape[-1]) != d_rope:
        raise ValueError("rope_concat_qk_decode_pos: K/V dims must match Q dims")
    if d_rope % 2 != 0:
        raise ValueError("rope_concat_qk_decode_pos: d_rope must be even")
    if int(cos.ndim) != 1 or int(sin.ndim) != 1:
        raise ValueError("rope_concat_qk_decode_pos: cos/sin must be 1D (d_rope/2,)")
    if int(cos.shape[0]) != d_rope // 2 or int(sin.shape[0]) != d_rope // 2:
        raise ValueError("rope_concat_qk_decode_pos: cos/sin length must be d_rope/2")

    D_OUT = d_nope + d_rope
    k = _rope_concat_qk_decode_pos_kernel(d_nope, d_rope, Hq)
    q_out_shape = (B, Hq, 1, D_OUT)
    k_out_shape = (B, 1, 1, D_OUT)
    total_threads = B * (Hq + 1) * D_OUT

    q_out, k_out = k(
        q_nope,
        q_rope,
        kv_nope,
        k_rope,
        cos,
        sin,
        template=[("T", q_nope.dtype)],
        grid=(total_threads, 1, 1),
        output_shapes=[q_out_shape, k_out_shape],
        output_dtypes=[q_nope.dtype, q_nope.dtype],
    )
    return q_out, k_out


__all__ = [
    "apply_rope",
    "apply_rope_interleaved",
    "apply_gqa_rope",
    "RoPECosSin",
    "rope_concat_qk_decode",
    "rope_concat_qk_decode_pos",
    "rope_and_cache_update",
    "paged_rope_and_cache_update",
]
