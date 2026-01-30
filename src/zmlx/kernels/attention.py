from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .softmax import _validate_tg


@cache
def _logsumexp_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)

    # Custom source for logsumexp — a reduction, not per-element write.
    
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // max pass
        float m = -INFINITY;
        for (uint j = tid; j < D; j += TG) {{
            m = metal::max(m, (float)inp[base + j]);
        }}
        buf[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] = metal::max(buf[tid], buf[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float max_val = buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // sum(exp(x - max)) pass
        float sum_exp = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            sum_exp += metal::exp((float)inp[base + j] - max_val);
        }}
        buf[tid] = sum_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        
        if (tid == 0) {{
            out[row] = (T)(max_val + metal::log(buf[0]));
        }}
    """

    return metal_kernel(
        name=f"kk_logsumexp_D{D}_TG{TG}",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def logsumexp_lastdim(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Stable LogSumExp over the last dimension.

    Args:
        x: Input array with shape ``(..., D)``.
        threadgroup: Threadgroup size (must be a power of two).
        compute_dtype: Dtype used for internal computation.

    Returns:
        An array with shape ``x.shape[:-1]`` containing LogSumExp values.
    """
    if x.ndim < 1:
        return x
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _logsumexp_kernel(D, TG)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    return k(
        x,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape[:-1]],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _masked_softmax_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // max pass with mask
        float m = -INFINITY;
        for (uint j = tid; j < D; j += TG) {{
            if (mask[base + j]) {{
                m = metal::max(m, (float)inp[base + j]);
            }}
        }}
        buf[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] = metal::max(buf[tid], buf[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float max_val = buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // sum(exp) pass
        float sum_exp = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            if (mask[base + j]) {{
                sum_exp += metal::exp((float)inp[base + j] - max_val);
            }}
        }}
        buf[tid] = sum_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float total_sum = buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // write
        for (uint j = tid; j < D; j += TG) {{
            if (mask[base + j]) {{
                out[base + j] = (T)(metal::exp((float)inp[base + j] - max_val) / total_sum);
            }} else {{
                out[base + j] = (T)0;
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_masked_softmax_D{D}_TG{TG}",
        input_names=["inp", "mask"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def masked_softmax(x: Any, mask: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> Any:
    """Softmax with a boolean mask over the last dimension.

    Args:
        x: Input array with shape ``(..., D)``.
        mask: Boolean mask with the same shape as ``x``. False entries are zeroed.
        threadgroup: Threadgroup size (must be a power of two).
        compute_dtype: Dtype used for internal computation.

    Returns:
        An array with the same shape as ``x``.
    """
    if x.shape != mask.shape:
        raise ValueError("masked_softmax: x and mask must have same shape")
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _masked_softmax_kernel(D, TG)
    rows = x.size // D
    cd = compute_dtype or mx.float32
    return k(
        x, mask,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _scale_mask_softmax_kernel(d: int, tg: int, scale: float) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    scale_f = float(scale)
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float scale_val = {scale_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // 1. Max pass with scale and mask
        float m = -INFINITY;
        for (uint j = tid; j < D; j += TG) {{
            if (mask[base + j]) {{
                float val = (float)inp[base + j] * scale_val;
                m = metal::max(m, val);
            }}
        }}
        buf[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] = metal::max(buf[tid], buf[tid + s]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float max_val = buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Sum(exp) pass
        float sum_exp = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            if (mask[base + j]) {{
                float val = (float)inp[base + j] * scale_val;
                sum_exp += metal::exp(val - max_val);
            }}
        }}
        buf[tid] = sum_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = TG/2; s > 0; s >>= 1) {{
            if (tid < s) buf[tid] += buf[tid + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float total_sum = buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Write
        for (uint j = tid; j < D; j += TG) {{
            if (mask[base + j]) {{
                float val = (float)inp[base + j] * scale_val;
                out[base + j] = (T)(metal::exp(val - max_val) / total_sum);
            }} else {{
                out[base + j] = (T)0;
            }}
        }}
    """
    scale_str = str(scale_f).replace(".", "_").replace("-", "_")
    return metal_kernel(
        name=f"kk_scale_mask_softmax_D{D}_TG{TG}_S{scale_str}",
        input_names=["inp", "mask"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def scale_mask_softmax(
    x: Any,
    mask: Any,
    scale: float,
    *,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> Any:
    """Fused scaling + masking + softmax over the last dimension.

    Args:
        x: Input array with shape ``(..., D)``.
        mask: Boolean mask with the same shape as ``x``. False entries are zeroed.
        scale: Multiplicative scale applied to ``x`` before softmax.
        threadgroup: Threadgroup size (must be a power of two).
        compute_dtype: Dtype used for internal computation.

    Returns:
        An array with the same shape as ``x``.
    """
    if x.shape != mask.shape:
        raise ValueError("scale_mask_softmax: x and mask must have same shape")
    D = x.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _scale_mask_softmax_kernel(D, TG, float(scale))
    rows = x.size // D
    cd = compute_dtype or mx.float32
    return k(
        x, mask,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


@cache
def _attention_tile_kernel(head_dim: int) -> Any:
    D = int(head_dim)
    # Simple 16x16 attention tile prototype
    # Q: (16, D), K: (16, D)
    source = f"""
        constexpr uint D = {D};
        uint col = thread_position_in_threadgroup.x;
        uint row = thread_position_in_threadgroup.y;
        
        threadgroup float q_tile[16][D];
        threadgroup float k_tile[16][D];
        
        // Load Q and K into shared memory
        // Each thread (row, col) loads elements across D
        for (uint i = col; i < D; i += 16) {{
            q_tile[row][i] = (float)Q[row * D + i];
            k_tile[row][i] = (float)K[row * D + i];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute S[row, col] = sum_i Q[row, i] * K[col, i]
        float score = 0.0f;
        for (uint i = 0; i < D; ++i) {{
            score += q_tile[row][i] * k_tile[col][i];
        }}
        
        out[row * 16 + col] = (T)score;
    """
    return metal_kernel(
        name="kk_attention_tile_proto",
        input_names=["Q", "K"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def attention_tile_proto(q: Any, k: Any) -> Any:
    """Prototype 16x16 attention tile (dot product).

    This is an experimental kernel intended for exploration and benchmarking.

    Args:
        q: Query tile with shape ``(16, D)``.
        k: Key tile with shape ``(16, D)``.

    Returns:
        A ``(16, 16)`` score tile.
    """
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("attention_tile_proto: q and k must be rank-2 arrays")
    if int(q.shape[0]) != 16 or int(k.shape[0]) != 16:
        raise ValueError("attention_tile_proto: q and k must have shape (16, D)")
    if int(q.shape[1]) != int(k.shape[1]):
        raise ValueError("attention_tile_proto: q and k must have the same D")

    D = q.shape[-1]
    kernel = _attention_tile_kernel(D)
    return kernel(
        q, k,
        template=[("T", mx.float32)],
        grid=(16, 16, 1), # One threadgroup
        threadgroup=(16, 16, 1),
        output_shapes=[(16, 16)],
        output_dtypes=[mx.float32],
    )[0]

@cache
def _paged_attention_kernel(
    h: int,
    h_kv: int,
    d: int,
    block_size: int,
    max_blocks: int,
    tg: int,
    max_context: int = 4096,
) -> Any:
    H = int(h)
    HKV = int(h_kv)
    D = int(d)
    BS = int(block_size)
    MB = int(max_blocks)
    TG = _validate_tg(tg)
    G = H // HKV
    MC = int(max_context)

    source = f"""
        constexpr uint H = {H};
        constexpr uint HKV = {HKV};
        constexpr uint D = {D};
        constexpr uint BS = {BS};
        constexpr uint MB = {MB};
        constexpr uint TG = {TG};
        constexpr uint G = {G};
        constexpr uint MC = {MC};

        uint tid = thread_position_in_threadgroup.x;
        uint batch_idx = thread_position_in_grid.y;
        uint head_idx = thread_position_in_grid.z;
        uint kv_head_idx = head_idx / G;

        uint context_len = (uint)context_lens[batch_idx];
        float scale_val = (float)scale[0];

        threadgroup float m_buf[TG];
        threadgroup float s_buf[TG];
        threadgroup float scores[MC]; // Store scores to avoid recomputing

        // 1. Compute scores
        float m = -INFINITY;
        for (uint i = tid; i < context_len; i += TG) {{
            uint block_logical_idx = i / BS;
            uint token_block_idx = i % BS;
            uint physical_block = (uint)block_table[batch_idx * MB + block_logical_idx];
            
            float score = 0.0f;
            uint q_off = (batch_idx * H + head_idx) * D;
            uint k_off = ((physical_block * BS + token_block_idx) * HKV + kv_head_idx) * D;
            
            for (uint d = 0; d < D; ++d) {{
                score += (float)q[q_off + d] * (float)k_cache[k_off + d];
            }}
            score *= scale_val;
            scores[i] = score;
            m = metal::max(m, score);
        }}

        // 2. Reduce m across threadgroup
        m_buf[tid] = m;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = TG / 2; i > 0; i >>= 1) {{
            if (tid < i) m_buf[tid] = metal::max(m_buf[tid], m_buf[tid + i]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float global_m = m_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. Compute sum(exp(score - m)) and update scores to exp(score - m)
        float s = 0.0f;
        for (uint i = tid; i < context_len; i += TG) {{
            float weight = metal::exp(scores[i] - global_m);
            scores[i] = weight;
            s += weight;
        }}
        s_buf[tid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = TG / 2; i > 0; i >>= 1) {{
            if (tid < i) s_buf[tid] += s_buf[tid + i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float global_s = s_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. Compute weighted sum of V
        for (uint d = tid; d < D; d += TG) {{
            float acc = 0.0f;
            for (uint i = 0; i < context_len; ++i) {{
                uint block_logical_idx = i / BS;
                uint token_block_idx = i % BS;
                uint physical_block = (uint)block_table[batch_idx * MB + block_logical_idx];
                uint v_off = ((physical_block * BS + token_block_idx) * HKV + kv_head_idx) * D;
                
                acc += (scores[i] / global_s) * (float)v_cache[v_off + d];
            }}
            out[(batch_idx * H + head_idx) * D + d] = (T)acc;
        }}
    """
    return metal_kernel(
        name=f"kk_paged_attention_H{H}_HKV{HKV}_D{D}_BS{BS}_MC{MC}",
        input_names=["q", "k_cache", "v_cache", "block_table", "context_lens", "scale"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def paged_attention(
    q: Any,
    k_cache: Any,
    v_cache: Any,
    block_table: Any,
    context_lens: Any,
    scale: float | None = None,
    *,
    threadgroup: int = 256,
    max_context: int = 4096,
) -> Any:
    """Paged Attention kernel for decoding.
    
    q: (B, H, D)
    k_cache: (N_BLOCKS, BS, HKV, D)
    v_cache: (N_BLOCKS, BS, HKV, D)
    block_table: (B, MAX_BLOCKS)
    context_lens: (B,)
    """
    B, H, D = q.shape
    _, BS, HKV, _ = k_cache.shape
    MB = block_table.shape[1]
    
    if scale is None:
        scale = 1.0 / (D ** 0.5)
        
    tg = _validate_tg(threadgroup)
    k = _paged_attention_kernel(H, HKV, D, BS, MB, tg, max_context=max_context)
    
    scale_arr = mx.array([scale], dtype=mx.float32)
    
    return k(
        q, k_cache, v_cache, block_table, context_lens, scale_arr,
        template=[("T", q.dtype)],
        grid=(tg, B, H),
        threadgroup=(tg, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


__all__ = [
    "logsumexp_lastdim",
    "masked_softmax",
    "scale_mask_softmax",
    "attention_tile_proto",
    "paged_attention",
]
