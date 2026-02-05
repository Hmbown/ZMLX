from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .softmax import _validate_tg

_MAX_FUSED_TOPK = 8

_TOPK_HEADER = r"""
#define KK_TOPK_INIT(vals, idxs)                          \
    do {                                                  \
        for (uint _i = 0; _i < K; ++_i) {                 \
            (vals)[_i] = -INFINITY;                       \
            (idxs)[_i] = 0;                               \
        }                                                 \
    } while (0)

#define KK_TOPK_INSERT(vals, idxs, v, i)                  \
    do {                                                  \
        if ((v) <= (vals)[K - 1]) {                       \
            break;                                        \
        }                                                 \
        uint _pos = K - 1;                                \
        for (; _pos > 0; --_pos) {                        \
            if ((v) <= (vals)[_pos - 1]) {                \
                break;                                    \
            }                                             \
            (vals)[_pos] = (vals)[_pos - 1];              \
            (idxs)[_pos] = (idxs)[_pos - 1];              \
        }                                                 \
        (vals)[_pos] = (v);                               \
        (idxs)[_pos] = (i);                               \
    } while (0)
"""


@cache
def _topk_gating_simd_kernel(d: int, k: int) -> Any:
    D = int(d)
    K = int(k)
    if D <= 0:
        raise ValueError("topk_gating_softmax: last dimension must be > 0")
    if D > 32:
        raise ValueError("topk_gating_softmax: simd kernel requires D <= 32")
    if K <= 0 or K > _MAX_FUSED_TOPK:
        raise ValueError("topk_gating_softmax: invalid k for simd kernel")

    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        constexpr uint SG = 32;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / SG;
        uint base = row * D;

        float v = -INFINITY;
        if (tid < D) {{
            v = (float)inp[base + tid];
        }}

        thread float topk_vals[K];
        thread uint topk_idx[K];

        float cur = v;
        for (uint i = 0; i < K; ++i) {{
            float cur_max = simd_max(cur);
            uint candidate = (cur == cur_max && tid < D) ? tid : 0;
            uint winner = simd_max(candidate);
            if (tid == 0) {{
                topk_vals[i] = cur_max;
                topk_idx[i] = winner;
            }}
            cur = (tid == winner) ? -INFINITY : cur;
        }}

        if (tid == 0) {{
            float m = topk_vals[0];
            float s = 0.0f;
            for (uint i = 0; i < K; ++i) {{
                s += metal::exp(topk_vals[i] - m);
            }}
            float inv = 1.0f / s;
            uint out_base = row * K;
            for (uint i = 0; i < K; ++i) {{
                weights[out_base + i] = (T)(metal::exp(topk_vals[i] - m) * inv);
                indices[out_base + i] = topk_idx[i];
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_topk_gating_simd_D{D}_K{K}",
        input_names=["inp"],
        output_names=["weights", "indices"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _topk_softmax_simd_kernel(d: int, k: int, renorm: bool) -> Any:
    D = int(d)
    K = int(k)
    if D <= 0:
        raise ValueError("topk_gating_softmax: last dimension must be > 0")
    if D > 32:
        raise ValueError("topk_gating_softmax: simd kernel requires D <= 32")
    if K <= 0 or K > _MAX_FUSED_TOPK:
        raise ValueError("topk_gating_softmax: invalid k for simd kernel")

    renorm_literal = str(bool(renorm)).lower()

    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        constexpr uint SG = 32;
        constexpr bool RENORM = {renorm_literal};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / SG;
        uint base = row * D;

        float v = -INFINITY;
        if (tid < D) {{
            v = (float)inp[base + tid];
        }}

        float row_max = simd_max(v);
        float exp_v = (tid < D) ? metal::exp(v - row_max) : 0.0f;
        float row_sum = simd_sum(exp_v);
        float p = (tid < D) ? (exp_v / row_sum) : -INFINITY;

        thread float topk_vals[K];
        thread uint topk_idx[K];

        float cur = p;
        for (uint i = 0; i < K; ++i) {{
            float cur_max = simd_max(cur);
            uint candidate = (cur == cur_max && tid < D) ? tid : 0;
            uint winner = simd_max(candidate);
            if (tid == 0) {{
                topk_vals[i] = cur_max;
                topk_idx[i] = winner;
            }}
            cur = (tid == winner) ? -INFINITY : cur;
        }}

        if (tid == 0) {{
            float denom = 1.0f;
            if (RENORM) {{
                float sum_k = 0.0f;
                for (uint i = 0; i < K; ++i) {{
                    sum_k += topk_vals[i];
                }}
                denom = sum_k + 1e-20f;
            }}
            uint out_base = row * K;
            for (uint i = 0; i < K; ++i) {{
                float v_out = topk_vals[i];
                weights[out_base + i] = (T)(RENORM ? (v_out / denom) : v_out);
                indices[out_base + i] = topk_idx[i];
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_topk_softmax_simd_D{D}_K{K}_R{int(bool(renorm))}",
        input_names=["inp"],
        output_names=["weights", "indices"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _topk_softmax_bias_simd_kernel(d: int, k: int, renorm: bool) -> Any:
    D = int(d)
    K = int(k)
    if D <= 0:
        raise ValueError("topk_gating_softmax: last dimension must be > 0")
    if D > 32:
        raise ValueError("topk_gating_softmax: simd kernel requires D <= 32")
    if K <= 0 or K > _MAX_FUSED_TOPK:
        raise ValueError("topk_gating_softmax: invalid k for simd kernel")

    renorm_literal = str(bool(renorm)).lower()

    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        constexpr uint SG = 32;
        constexpr bool RENORM = {renorm_literal};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / SG;
        uint base = row * D;

        float v = -INFINITY;
        if (tid < D) {{
            v = (float)inp[base + tid];
        }}

        float row_max = simd_max(v);
        float exp_v = (tid < D) ? metal::exp(v - row_max) : 0.0f;
        float row_sum = simd_sum(exp_v);
        float p = (tid < D) ? (exp_v / row_sum + (float)bias[tid]) : -INFINITY;

        thread float topk_vals[K];
        thread uint topk_idx[K];

        float cur = p;
        for (uint i = 0; i < K; ++i) {{
            float cur_max = simd_max(cur);
            uint candidate = (cur == cur_max && tid < D) ? tid : 0;
            uint winner = simd_max(candidate);
            if (tid == 0) {{
                topk_vals[i] = cur_max;
                topk_idx[i] = winner;
            }}
            cur = (tid == winner) ? -INFINITY : cur;
        }}

        if (tid == 0) {{
            float denom = 1.0f;
            if (RENORM) {{
                float sum_k = 0.0f;
                for (uint i = 0; i < K; ++i) {{
                    sum_k += topk_vals[i];
                }}
                denom = sum_k + 1e-20f;
            }}
            uint out_base = row * K;
            for (uint i = 0; i < K; ++i) {{
                float v_out = topk_vals[i];
                weights[out_base + i] = (T)(RENORM ? (v_out / denom) : v_out);
                indices[out_base + i] = topk_idx[i];
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_topk_softmax_bias_simd_D{D}_K{K}_R{int(bool(renorm))}",
        input_names=["inp", "bias"],
        output_names=["weights", "indices"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

@cache
def _top2_gating_kernel(d: int, tg: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float val1_buf[TG];
        threadgroup float val2_buf[TG];
        threadgroup uint idx1_buf[TG];
        threadgroup uint idx2_buf[TG];

        float top1_v = -INFINITY;
        float top2_v = -INFINITY;
        uint top1_i = 0;
        uint top2_i = 0;

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            if (v > top1_v) {{
                top2_v = top1_v;
                top2_i = top1_i;
                top1_v = v;
                top1_i = j;
            }} else if (v > top2_v) {{
                top2_v = v;
                top2_i = j;
            }}
        }}
        
        val1_buf[tid] = top1_v;
        val2_buf[tid] = top2_v;
        idx1_buf[tid] = top1_i;
        idx2_buf[tid] = top2_i;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint st = TG/2; st > 0; st >>= 1) {{
            if (tid < st) {{
                float v1_a = val1_buf[tid];
                float v1_b = val1_buf[tid + st];
                float v2_a = val2_buf[tid];
                float v2_b = val2_buf[tid + st];
                
                uint i1_a = idx1_buf[tid];
                uint i1_b = idx1_buf[tid + st];
                uint i2_a = idx2_buf[tid];
                uint i2_b = idx2_buf[tid + st];

                // Merge two top-2 sets
                float res_v1, res_v2;
                uint res_i1, res_i2;
                
                if (v1_a > v1_b) {{
                    res_v1 = v1_a; res_i1 = i1_a;
                    if (v1_b > v2_a) {{
                        res_v2 = v1_b; res_i2 = i1_b;
                    }} else {{
                        res_v2 = v2_a; res_i2 = i2_a;
                    }}
                }} else {{
                    res_v1 = v1_b; res_i1 = i1_b;
                    if (v1_a > v2_b) {{
                        res_v2 = v1_a; res_i2 = i1_a;
                    }} else {{
                        res_v2 = v2_b; res_i2 = i2_b;
                    }}
                }}
                
                val1_buf[tid] = res_v1;
                idx1_buf[tid] = res_i1;
                val2_buf[tid] = res_v2;
                idx2_buf[tid] = res_i2;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0) {{
            float m = val1_buf[0];
            float v1 = metal::exp(val1_buf[0] - m);
            float v2 = metal::exp(val2_buf[0] - m);
            float s = v1 + v2;
            
            weights[row * 2] = (T)(v1 / s);
            weights[row * 2 + 1] = (T)(v2 / s);
            indices[row * 2] = idx1_buf[0];
            indices[row * 2 + 1] = idx2_buf[0];
        }}
    """
    return metal_kernel(
        name=f"kk_top2_gating_D{D}_TG{TG}",
        input_names=["inp"],
        output_names=["weights", "indices"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def top2_gating_softmax(x: Any, *, threadgroup: int = 256, compute_dtype: Any | None = None) -> tuple[Any, Any]:
    """Top-2 gating with softmax for Mixture of Experts.
    
    Returns:
      - weights: (..., 2) softmax probabilities
      - indices: (..., 2) expert indices (uint32)
    """
    D = x.shape[-1]
    cd = compute_dtype or mx.float32
    rows = x.size // D

    if D <= 32:
        k = _topk_gating_simd_kernel(D, 2)
        TG = 32
        weights, indices = k(
            x,
            template=[("T", cd)],
            grid=(rows * TG, 1, 1),
            threadgroup=(TG, 1, 1),
            output_shapes=[x.shape[:-1] + (2,), x.shape[:-1] + (2,)],
            output_dtypes=[cd, mx.uint32],
        )
    else:
        TG = _validate_tg(threadgroup)
        k = _top2_gating_kernel(D, TG)
        weights, indices = k(
            x,
            template=[("T", cd)],
            grid=(rows * TG, 1, 1),
            threadgroup=(TG, 1, 1),
            output_shapes=[x.shape[:-1] + (2,), x.shape[:-1] + (2,)],
            output_dtypes=[cd, mx.uint32],
        )
    return weights, indices


@cache
def _topk_gating_kernel(d: int, k: int, tg: int) -> Any:
    D = int(d)
    K = int(k)
    TG = _validate_tg(tg)
    if K <= 0:
        raise ValueError("topk_gating_softmax: k must be > 0")
    if K > _MAX_FUSED_TOPK:
        raise ValueError(f"topk_gating_softmax: k must be <= {_MAX_FUSED_TOPK} for fused kernel")

    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float val_buf[TG * K];
        threadgroup uint idx_buf[TG * K];

        thread float vals[K];
        thread uint idxs[K];
        KK_TOPK_INIT(vals, idxs);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            KK_TOPK_INSERT(vals, idxs, v, j);
        }}

        uint off = tid * K;
        for (uint i = 0; i < K; ++i) {{
            val_buf[off + i] = vals[i];
            idx_buf[off + i] = idxs[i];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint st = TG / 2; st > 0; st >>= 1) {{
            if (tid < st) {{
                for (uint i = 0; i < K; ++i) {{
                    vals[i] = val_buf[off + i];
                    idxs[i] = idx_buf[off + i];
                }}
                uint off_b = (tid + st) * K;
                for (uint i = 0; i < K; ++i) {{
                    float v = val_buf[off_b + i];
                    uint idx = idx_buf[off_b + i];
                    KK_TOPK_INSERT(vals, idxs, v, idx);
                }}
                for (uint i = 0; i < K; ++i) {{
                    val_buf[off + i] = vals[i];
                    idx_buf[off + i] = idxs[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0) {{
            float m = val_buf[0];
            float s = 0.0f;
            for (uint i = 0; i < K; ++i) {{
                s += metal::exp(val_buf[i] - m);
            }}
            float inv = 1.0f / s;
            uint out_base = row * K;
            for (uint i = 0; i < K; ++i) {{
                weights[out_base + i] = (T)(metal::exp(val_buf[i] - m) * inv);
                indices[out_base + i] = idx_buf[i];
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_topk_gating_D{D}_K{K}_TG{TG}",
        input_names=["inp"],
        output_names=["weights", "indices"],
        source=source,
        header=DEFAULT_HEADER + _TOPK_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _topk_softmax_kernel(d: int, k: int, tg: int, renorm: bool) -> Any:
    D = int(d)
    K = int(k)
    TG = _validate_tg(tg)
    if K <= 0:
        raise ValueError("topk_gating_softmax: k must be > 0")
    if K > _MAX_FUSED_TOPK:
        raise ValueError(f"topk_gating_softmax: k must be <= {_MAX_FUSED_TOPK} for fused kernel")

    renorm_literal = str(bool(renorm)).lower()

    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        constexpr uint TG = {TG};
        constexpr bool RENORM = {renorm_literal};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float red_buf[TG];
        threadgroup float val_buf[TG * K];
        threadgroup uint idx_buf[TG * K];

        // Pass 1: row max
        float local_max = -INFINITY;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            if (v > local_max) local_max = v;
        }}
        red_buf[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2; st > 0; st >>= 1) {{
            if (tid < st) {{
                float a = red_buf[tid];
                float b = red_buf[tid + st];
                red_buf[tid] = metal::max(a, b);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float row_max = red_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pass 2: sum exp
        float local_sum = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            local_sum += metal::exp(v - row_max);
        }}
        red_buf[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2; st > 0; st >>= 1) {{
            if (tid < st) {{
                red_buf[tid] += red_buf[tid + st];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float row_sum = red_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pass 3: top-k of softmax probabilities
        thread float vals[K];
        thread uint idxs[K];
        KK_TOPK_INIT(vals, idxs);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            float p = metal::exp(v - row_max) / row_sum;
            KK_TOPK_INSERT(vals, idxs, p, j);
        }}

        uint off = tid * K;
        for (uint i = 0; i < K; ++i) {{
            val_buf[off + i] = vals[i];
            idx_buf[off + i] = idxs[i];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint st = TG / 2; st > 0; st >>= 1) {{
            if (tid < st) {{
                for (uint i = 0; i < K; ++i) {{
                    vals[i] = val_buf[off + i];
                    idxs[i] = idx_buf[off + i];
                }}
                uint off_b = (tid + st) * K;
                for (uint i = 0; i < K; ++i) {{
                    float v = val_buf[off_b + i];
                    uint idx = idx_buf[off_b + i];
                    KK_TOPK_INSERT(vals, idxs, v, idx);
                }}
                for (uint i = 0; i < K; ++i) {{
                    val_buf[off + i] = vals[i];
                    idx_buf[off + i] = idxs[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0) {{
            float denom = 1.0f;
            if (RENORM) {{
                float sum_k = 0.0f;
                for (uint i = 0; i < K; ++i) {{
                    sum_k += val_buf[i];
                }}
                denom = sum_k + 1e-20f;
            }}
            uint out_base = row * K;
            for (uint i = 0; i < K; ++i) {{
                float v = val_buf[i];
                weights[out_base + i] = (T)(RENORM ? (v / denom) : v);
                indices[out_base + i] = idx_buf[i];
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_topk_softmax_D{D}_K{K}_TG{TG}_R{int(bool(renorm))}",
        input_names=["inp"],
        output_names=["weights", "indices"],
        source=source,
        header=DEFAULT_HEADER + _TOPK_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _topk_softmax_bias_kernel(d: int, k: int, tg: int, renorm: bool) -> Any:
    D = int(d)
    K = int(k)
    TG = _validate_tg(tg)
    if K <= 0:
        raise ValueError("topk_gating_softmax: k must be > 0")
    if K > _MAX_FUSED_TOPK:
        raise ValueError(f"topk_gating_softmax: k must be <= {_MAX_FUSED_TOPK} for fused kernel")

    renorm_literal = str(bool(renorm)).lower()

    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        constexpr uint TG = {TG};
        constexpr bool RENORM = {renorm_literal};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float red_buf[TG];
        threadgroup float val_buf[TG * K];
        threadgroup uint idx_buf[TG * K];

        // Pass 1: row max
        float local_max = -INFINITY;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            if (v > local_max) local_max = v;
        }}
        red_buf[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2; st > 0; st >>= 1) {{
            if (tid < st) {{
                float a = red_buf[tid];
                float b = red_buf[tid + st];
                red_buf[tid] = metal::max(a, b);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float row_max = red_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pass 2: sum exp
        float local_sum = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            local_sum += metal::exp(v - row_max);
        }}
        red_buf[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint st = TG / 2; st > 0; st >>= 1) {{
            if (tid < st) {{
                red_buf[tid] += red_buf[tid + st];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float row_sum = red_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Pass 3: top-k of softmax probabilities + bias
        thread float vals[K];
        thread uint idxs[K];
        KK_TOPK_INIT(vals, idxs);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            float p = metal::exp(v - row_max) / row_sum + (float)bias[j];
            KK_TOPK_INSERT(vals, idxs, p, j);
        }}

        uint off = tid * K;
        for (uint i = 0; i < K; ++i) {{
            val_buf[off + i] = vals[i];
            idx_buf[off + i] = idxs[i];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint st = TG / 2; st > 0; st >>= 1) {{
            if (tid < st) {{
                for (uint i = 0; i < K; ++i) {{
                    vals[i] = val_buf[off + i];
                    idxs[i] = idx_buf[off + i];
                }}
                uint off_b = (tid + st) * K;
                for (uint i = 0; i < K; ++i) {{
                    float v = val_buf[off_b + i];
                    uint idx = idx_buf[off_b + i];
                    KK_TOPK_INSERT(vals, idxs, v, idx);
                }}
                for (uint i = 0; i < K; ++i) {{
                    val_buf[off + i] = vals[i];
                    idx_buf[off + i] = idxs[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0) {{
            float denom = 1.0f;
            if (RENORM) {{
                float sum_k = 0.0f;
                for (uint i = 0; i < K; ++i) {{
                    sum_k += val_buf[i];
                }}
                denom = sum_k + 1e-20f;
            }}
            uint out_base = row * K;
            for (uint i = 0; i < K; ++i) {{
                float v = val_buf[i];
                weights[out_base + i] = (T)(RENORM ? (v / denom) : v);
                indices[out_base + i] = idx_buf[i];
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_topk_softmax_bias_D{D}_K{K}_TG{TG}_R{int(bool(renorm))}",
        input_names=["inp", "bias"],
        output_names=["weights", "indices"],
        source=source,
        header=DEFAULT_HEADER + _TOPK_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

@cache
def _moe_dispatch_kernel(d: int, k: int) -> Any:
    D = int(d)
    K = int(k)
    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;
        uint k_idx = thread_position_in_grid.z;
        
        // This is a simple gather-like dispatch
        // x: (B, D)
        // indices: (B, K)
        // out: (B, K, D)
        out[(token_idx * K + k_idx) * D + d_idx] = x[token_idx * D + d_idx];
    """
    return metal_kernel(
        name=f"kk_moe_dispatch_D{D}_K{K}",
        input_names=["x", "indices"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def moe_dispatch(x: Any, indices: Any) -> Any:
    """Dispatch tokens to expert slots.
    
    x: (..., D)
    indices: (..., K)
    Returns: (..., K, D)
    """
    original_shape = x.shape[:-1]
    D = x.shape[-1]
    K = indices.shape[-1]
    
    x_flat = x.reshape(-1, D)
    indices_flat = indices.reshape(-1, K)
    B = x_flat.shape[0]
    
    k = _moe_dispatch_kernel(D, K)
    out = k(
        x_flat, indices_flat,
        template=[("T", x.dtype)],
        grid=(D, B, K),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, K, D)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape((*original_shape, K, D))


@cache
def _moe_combine_kernel(d: int, k: int) -> Any:
    D = int(d)
    K = int(k)
    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;
        
        float acc = 0.0f;
        for (uint i = 0; i < K; ++i) {{
            float w = (float)weights[token_idx * K + i];
            float v = (float)expert_outputs[(token_idx * K + i) * D + d_idx];
            acc += w * v;
        }}
        out[token_idx * D + d_idx] = (T)acc;
    """
    return metal_kernel(
        name=f"kk_moe_combine_D{D}_K{K}",
        input_names=["expert_outputs", "weights"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def moe_combine(expert_outputs: Any, weights: Any) -> Any:
    """Combine expert outputs using gating weights.
    
    expert_outputs: (..., K, D)
    weights: (..., K)
    Returns: (..., D)
    """
    original_shape = weights.shape[:-1]
    K = weights.shape[-1]
    D = expert_outputs.shape[-1]
    
    expert_outputs_flat = expert_outputs.reshape(-1, K, D)
    weights_flat = weights.reshape(-1, K)
    B = weights_flat.shape[0]
    
    k = _moe_combine_kernel(D, K)
    out = k(
        expert_outputs_flat, weights_flat,
        template=[("T", expert_outputs.dtype)],
        grid=(D, B, 1),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, D)],
        output_dtypes=[expert_outputs.dtype],
    )[0]
    return out.reshape((*original_shape, D))


@cache
def _moe_combine_kernel_no_fma(d: int, k: int) -> Any:
    """Combine kernel that matches MLX's non-FMA multiply+sum semantics.

    MLX computes MoE combine as a float32 multiply followed by a float32
    reduction, which rounds the product to float32 *before* accumulation.
    Metal may contract ``acc += w * v`` into an FMA, which can change the final
    float32 sum enough to flip bfloat16/float16 rounding at ties.  This variant
    disables contraction to preserve token fidelity on models like GLM-4.
    """
    D = int(d)
    K = int(k)
    source = f"""
        #pragma clang fp contract(off)
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;

        float acc = 0.0f;
        for (uint i = 0; i < K; ++i) {{
            float w = (float)weights[token_idx * K + i];
            float v = (float)expert_outputs[(token_idx * K + i) * D + d_idx];
            float prod = w * v;
            acc = acc + prod;
        }}
        out[token_idx * D + d_idx] = (T)acc;
    """
    return metal_kernel(
        name=f"kk_moe_combine_no_fma_D{D}_K{K}",
        input_names=["expert_outputs", "weights"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def moe_combine_no_fma(expert_outputs: Any, weights: Any) -> Any:
    """Combine expert outputs using gating weights, preventing FMA contraction.

    expert_outputs: (..., K, D)
    weights: (..., K)
    Returns: (..., D)
    """
    original_shape = weights.shape[:-1]
    K = weights.shape[-1]
    D = expert_outputs.shape[-1]

    expert_outputs_flat = expert_outputs.reshape(-1, K, D)
    weights_flat = weights.reshape(-1, K)
    B = weights_flat.shape[0]

    k = _moe_combine_kernel_no_fma(D, K)
    out = k(
        expert_outputs_flat,
        weights_flat,
        template=[("T", expert_outputs.dtype)],
        grid=(D, B, 1),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, D)],
        output_dtypes=[expert_outputs.dtype],
    )[0]
    return out.reshape((*original_shape, D))


@cache
def _moe_combine_kernel_exact(d: int, k: int) -> Any:
    D = int(d)
    K = int(k)
    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;

        T acc = (T)0;
        for (uint i = 0; i < K; ++i) {{
            T w = weights[token_idx * K + i];
            T v = expert_outputs[(token_idx * K + i) * D + d_idx];
            // Match MLX bf16 semantics: round after multiply, then after add.
            T prod = (T)(w * v);
            acc = (T)(acc + prod);
        }}
        out[token_idx * D + d_idx] = acc;
    """
    return metal_kernel(
        name=f"kk_moe_combine_exact_D{D}_K{K}",
        input_names=["expert_outputs", "weights"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def moe_combine_exact(expert_outputs: Any, weights: Any) -> Any:
    """Combine expert outputs using gating weights (dtype-accurate accumulation).

    Matches MX behavior by accumulating in the input dtype (e.g., float16).
    """
    original_shape = weights.shape[:-1]
    K = weights.shape[-1]
    D = expert_outputs.shape[-1]

    expert_outputs_flat = expert_outputs.reshape(-1, K, D)
    weights_flat = weights.reshape(-1, K)
    B = weights_flat.shape[0]

    k = _moe_combine_kernel_exact(D, K)
    out = k(
        expert_outputs_flat, weights_flat,
        template=[("T", expert_outputs.dtype)],
        grid=(D, B, 1),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, D)],
        output_dtypes=[expert_outputs.dtype],
    )[0]
    return out.reshape((*original_shape, D))


@cache
def _moe_combine_kernel_fp32(d: int, k: int) -> Any:
    """Combine kernel: reads experts in T, weights in float, accumulates and outputs float32."""
    D = int(d)
    K = int(k)
    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;

        float acc = 0.0f;
        for (uint i = 0; i < K; ++i) {{
            float w = weights[token_idx * K + i];
            float v = (float)expert_outputs[(token_idx * K + i) * D + d_idx];
            acc += w * v;
        }}
        out[token_idx * D + d_idx] = acc;
    """
    return metal_kernel(
        name=f"kk_moe_combine_fp32_D{D}_K{K}",
        input_names=["expert_outputs", "weights"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def moe_combine_fp32(expert_outputs: Any, weights: Any) -> Any:
    """Combine expert outputs with float32 weights, accumulate and output in float32.

    Matches MLX's dtype promotion behavior when expert_outputs is bfloat16/float16
    and weights is float32: the multiply promotes to float32 and the sum stays float32.

    expert_outputs: (..., K, D) — any dtype (read as float32 internally)
    weights: (..., K) — float32
    Returns: (..., D) — float32
    """
    original_shape = weights.shape[:-1]
    K = weights.shape[-1]
    D = expert_outputs.shape[-1]

    expert_outputs_flat = expert_outputs.reshape(-1, K, D)
    weights_flat = weights.reshape(-1, K).astype(mx.float32)
    B = weights_flat.shape[0]

    k = _moe_combine_kernel_fp32(D, K)
    out = k(
        expert_outputs_flat, weights_flat,
        template=[("T", expert_outputs.dtype)],
        grid=(D, B, 1),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, D)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape((*original_shape, D))


def topk_gating_softmax(
    x: Any,
    k: int = 2,
    *,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
    expert_bias: Any | None = None,
    norm_topk_prob: bool | None = None,
) -> tuple[Any, Any]:
    """Top-k gating with softmax for Mixture of Experts.

    Defaults to the fast top-k softmax-on-selected-logits path (renormalized).
    When ``expert_bias`` is provided or ``norm_topk_prob=False``, falls back to
    full softmax then top-k selection, optionally fused into a single Metal kernel.

    Returns:
      - weights: (..., k) softmax probabilities
      - indices: (..., k) expert indices (uint32)
    """
    cd = compute_dtype or mx.float32
    x_cast = x.astype(cd) if x.dtype != cd else x
    K = int(k)
    if K <= 0:
        raise ValueError("topk_gating_softmax: k must be > 0")
    D = int(x.shape[-1])
    if K > D:
        raise ValueError(f"topk_gating_softmax: k={K} exceeds last dimension D={D}")

    norm = True if norm_topk_prob is None else bool(norm_topk_prob)

    bias = expert_bias
    bias_supported = False
    if bias is None:
        bias_supported = True
    else:
        try:
            if bias.ndim <= 2 and int(bias.shape[-1]) == D and int(bias.size) == D:
                bias_supported = True
                if bias.ndim != 1:
                    bias = bias.reshape((D,))
            else:
                bias_supported = False
        except Exception:
            bias_supported = False

    use_simd = D <= 32 and K <= _MAX_FUSED_TOPK

    # Fast path: top-k logits then softmax (renormalized). Exact when norm_topk_prob=True.
    if bias is None and norm:
        if K == 2:
            return top2_gating_softmax(x, threadgroup=threadgroup, compute_dtype=cd)
        if use_simd:
            TG = 32
            rows = x_cast.size // D
            kernel = _topk_gating_simd_kernel(D, K)
            weights, indices = kernel(
                x_cast,
                template=[("T", cd)],
                grid=(rows * TG, 1, 1),
                threadgroup=(TG, 1, 1),
                output_shapes=[x_cast.shape[:-1] + (K,), x_cast.shape[:-1] + (K,)],
                output_dtypes=[cd, mx.uint32],
            )
            return weights, indices
        if K <= _MAX_FUSED_TOPK:
            TG = _validate_tg(threadgroup)
            kernel = _topk_gating_kernel(D, K, TG)
            rows = x_cast.size // D
            weights, indices = kernel(
                x_cast,
                template=[("T", cd)],
                grid=(rows * TG, 1, 1),
                threadgroup=(TG, 1, 1),
                output_shapes=[x_cast.shape[:-1] + (K,), x_cast.shape[:-1] + (K,)],
                output_dtypes=[cd, mx.uint32],
            )
            return weights, indices

    # Full softmax path (exact for bias and/or norm_topk_prob=False).
    if K <= _MAX_FUSED_TOPK and bias_supported:
        rows = x_cast.size // D
        if use_simd:
            TG = 32
            if bias is None:
                kernel = _topk_softmax_simd_kernel(D, K, norm)
                weights, indices = kernel(
                    x_cast,
                    template=[("T", cd)],
                    grid=(rows * TG, 1, 1),
                    threadgroup=(TG, 1, 1),
                    output_shapes=[x_cast.shape[:-1] + (K,), x_cast.shape[:-1] + (K,)],
                    output_dtypes=[cd, mx.uint32],
                )
            else:
                bias_cast = bias.astype(cd) if bias.dtype != cd else bias
                kernel = _topk_softmax_bias_simd_kernel(D, K, norm)
                weights, indices = kernel(
                    x_cast,
                    bias_cast,
                    template=[("T", cd)],
                    grid=(rows * TG, 1, 1),
                    threadgroup=(TG, 1, 1),
                    output_shapes=[x_cast.shape[:-1] + (K,), x_cast.shape[:-1] + (K,)],
                    output_dtypes=[cd, mx.uint32],
                )
            return weights, indices

        TG = _validate_tg(threadgroup)
        if bias is None:
            kernel = _topk_softmax_kernel(D, K, TG, norm)
            weights, indices = kernel(
                x_cast,
                template=[("T", cd)],
                grid=(rows * TG, 1, 1),
                threadgroup=(TG, 1, 1),
                output_shapes=[x_cast.shape[:-1] + (K,), x_cast.shape[:-1] + (K,)],
                output_dtypes=[cd, mx.uint32],
            )
        else:
            bias_cast = bias.astype(cd) if bias.dtype != cd else bias
            kernel = _topk_softmax_bias_kernel(D, K, TG, norm)
            weights, indices = kernel(
                x_cast,
                bias_cast,
                template=[("T", cd)],
                grid=(rows * TG, 1, 1),
                threadgroup=(TG, 1, 1),
                output_shapes=[x_cast.shape[:-1] + (K,), x_cast.shape[:-1] + (K,)],
                output_dtypes=[cd, mx.uint32],
            )
        return weights, indices

    # Fallback MLX ops (exact).
    if bias is None and norm:
        sorted_indices = mx.argpartition(-x_cast, kth=K - 1, axis=-1)
        indices = sorted_indices[..., :K]
        values = mx.take_along_axis(x_cast, indices, axis=-1)
        weights = mx.softmax(values, axis=-1)
        return weights, indices.astype(mx.uint32)

    gates = mx.softmax(x_cast, axis=-1)
    if bias is not None:
        bias_cast = bias.astype(cd) if bias.dtype != cd else bias
        gates = gates + bias_cast
    inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if norm:
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)

    return scores, inds.astype(mx.uint32)


@cache
def _weighted_accumulate_kernel(d_out: int) -> Any:
    """Fused out = acc + gate_weight * projection — avoids materializing gate * proj."""
    D_out = int(d_out)
    source = f"""
        uint elem = thread_position_in_grid.x;
        constexpr uint D = {D_out};
        uint row = elem / D;
        float w = (float)gate[row];
        float v = (float)proj[elem];
        out[elem] = (T)((float)acc[elem] + w * v);
    """
    return metal_kernel(
        name=f"kk_weighted_accumulate_D{D_out}",
        input_names=["acc", "proj", "gate"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _weighted_accumulate_kernel_no_fma(d_out: int) -> Any:
    """Like `_weighted_accumulate_kernel` but prevents FMA contraction.

    This mirrors `moe_combine_no_fma` and is needed for token-fidelity on some
    models (e.g. GLM) where Metal's default FP contraction can change rounding.
    """
    D_out = int(d_out)
    source = f"""
        #pragma clang fp contract(off)
        uint elem = thread_position_in_grid.x;
        constexpr uint D = {D_out};
        uint row = elem / D;
        float w = (float)gate[row];
        float v = (float)proj[elem];
        float prod = w * v;
        out[elem] = (T)((float)acc[elem] + prod);
    """
    return metal_kernel(
        name=f"kk_weighted_accumulate_no_fma_D{D_out}",
        input_names=["acc", "proj", "gate"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def _should_use_streaming(batch_size: int, streaming: bool | None) -> bool:
    """Decide whether to use streaming accumulation.

    Default ON for B <= 16 (decode-like), OFF for B > 16 (batch prefill).
    Always overridable via explicit streaming kwarg.
    """
    if streaming is not None:
        return streaming
    return batch_size <= 16


def gather_qmm_combine(
    act: Any,
    weights: Any,
    gate: Any,
    indices: Any,
    *,
    streaming: bool | None = None,
    no_fma: bool = False,
) -> Any:
    """Fused gather-matmul-combine for MoE down-projection (dense weights).

    Computes ``output = sum_k(gate[:, k] * (act[:, k, :] @ weights[indices[:, k]]))``.
    When streaming is enabled, accumulates results without materializing the
    full ``(B, K, D_out)`` intermediate.

    Args:
        act: Dispatched activations, shape ``(B, K, D_in)``.
        weights: Expert weight matrices, shape ``(E, D_in, D_out)``.
        gate: Gating weights, shape ``(B, K)``.
        indices: Expert indices, shape ``(B, K)`` with dtype uint32.
        streaming: Force streaming mode on/off. Default: auto (ON for B <= 16).

    Returns:
        Combined output, shape ``(B, D_out)``.
    """
    if act.ndim != 3:
        raise ValueError("gather_qmm_combine: act must have shape (B, K, D_in)")
    B, K, _ = act.shape
    if weights.ndim != 3:
        raise ValueError("gather_qmm_combine: weights must have shape (E, D_in, D_out)")
    D_out = int(weights.shape[2])

    if not _should_use_streaming(B, streaming):
        # Non-streaming: batch matmul then combine
        # Gather expert weights: (B, K, D_in, D_out)
        proj_all = []
        for k_idx in range(K):
            # act[:, k_idx, :] -> (B, D_in)
            a_k = act[:, k_idx, :]
            # Gather weights for this k: indices[:, k_idx] -> (B,)
            w_k = weights[indices[:, k_idx]]  # (B, D_in, D_out)
            # Batched matmul: (B, 1, D_in) @ (B, D_in, D_out) -> (B, 1, D_out)
            proj_k = mx.matmul(mx.expand_dims(a_k, axis=1), w_k).squeeze(axis=1)
            proj_all.append(proj_k)
        # Stack: (B, K, D_out), then weighted sum
        proj_stacked = mx.stack(proj_all, axis=1)
        return mx.sum(proj_stacked * mx.expand_dims(gate, axis=-1), axis=1)

    # Streaming: accumulate without (B, K, D_out) intermediate
    k_acc = (
        _weighted_accumulate_kernel_no_fma(D_out)
        if no_fma
        else _weighted_accumulate_kernel(D_out)
    )

    output = mx.zeros((B, D_out), dtype=act.dtype)
    for k_idx in range(K):
        a_k = act[:, k_idx, :]
        w_k = weights[indices[:, k_idx]]
        proj_k = mx.matmul(mx.expand_dims(a_k, axis=1), w_k).squeeze(axis=1)
        gate_k = gate[:, k_idx]

        output = k_acc(
            output,
            proj_k,
            gate_k,
            template=[("T", act.dtype)],
            grid=(B * D_out, 1, 1),
            threadgroup=(min(D_out, 256), 1, 1),
            output_shapes=[(B, D_out)],
            output_dtypes=[act.dtype],
        )[0]

    return output


def gather_qmm_combine_quantized(
    act: Any,
    weights: Any,
    scales: Any,
    biases: Any,
    gate: Any,
    indices: Any,
    *,
    group_size: int = 64,
    bits: int = 4,
    streaming: bool | None = None,
    no_fma: bool = False,
) -> Any:
    """Fused gather-qmm-combine for MoE down-projection (quantized weights).

    Uses ``mx.gather_qmm`` for quantized matmul, then accumulates with a
    custom Metal kernel to avoid materializing ``(B, K, D_out)``.

    Args:
        act: Dispatched activations, shape ``(B, K, D_in)``.
        weights: Quantized expert weights (packed), shape ``(E, D_out, D_in_packed)``.
        scales: Quantization scales, shape ``(E, D_out, n_groups)``.
        biases: Quantization biases, shape ``(E, D_out, n_groups)``.
        gate: Gating weights, shape ``(B, K)``.
        indices: Expert indices, shape ``(B, K)`` with dtype uint32.
        group_size: Quantization group size.
        bits: Number of bits per weight.
        streaming: Force streaming mode on/off. Default: auto (ON for B <= 16).

    Returns:
        Combined output, shape ``(B, D_out)``.
    """
    if act.ndim != 3:
        raise ValueError(
            "gather_qmm_combine_quantized: act must have shape (B, K, D_in)"
        )
    B, K, _ = act.shape
    D_out = int(weights.shape[1])

    if not _should_use_streaming(B, streaming):
        # Non-streaming: gather_qmm per expert then combine
        proj_all = []
        for k_idx in range(K):
            a_k = act[:, k_idx, :]
            proj_k = mx.gather_qmm(
                a_k,
                weights,
                scales,
                biases,
                rhs_indices=indices[:, k_idx],
                transpose=True,
                group_size=group_size,
                bits=bits,
            )
            proj_all.append(proj_k)
        proj_stacked = mx.stack(proj_all, axis=1)
        return mx.sum(proj_stacked * mx.expand_dims(gate, axis=-1), axis=1)

    # Streaming: accumulate without (B, K, D_out) intermediate
    k_acc = (
        _weighted_accumulate_kernel_no_fma(D_out)
        if no_fma
        else _weighted_accumulate_kernel(D_out)
    )

    output = mx.zeros((B, D_out), dtype=act.dtype)
    for k_idx in range(K):
        a_k = act[:, k_idx, :]
        proj_k = mx.gather_qmm(
            a_k,
            weights,
            scales,
            biases,
            rhs_indices=indices[:, k_idx],
            transpose=True,
            group_size=group_size,
            bits=bits,
        )
        gate_k = gate[:, k_idx]

        output = k_acc(
            output,
            proj_k,
            gate_k,
            template=[("T", act.dtype)],
            grid=(B * D_out, 1, 1),
            threadgroup=(min(D_out, 256), 1, 1),
            output_shapes=[(B, D_out)],
            output_dtypes=[act.dtype],
        )[0]

    return output


__all__ = [
    "top2_gating_softmax",
    "topk_gating_softmax",
    "moe_dispatch",
    "moe_combine",
    "moe_combine_no_fma",
    "moe_combine_exact",
    "moe_combine_fp32",
    "gather_qmm_combine",
    "gather_qmm_combine_quantized",
]
