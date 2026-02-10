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

_TOPK_TIE_HEADER = r"""
#define KK_TOPK_BETTER_TIE(v, i, cur_v, cur_i)            \
    (((v) > (cur_v)) || (((v) == (cur_v)) && ((i) < (cur_i))))

#define KK_TOPK_INIT_TIE(vals, idxs)                      \
    do {                                                  \
        for (uint _i = 0; _i < K; ++_i) {                 \
            (vals)[_i] = -INFINITY;                       \
            (idxs)[_i] = 0xFFFFFFFFu;                     \
        }                                                 \
    } while (0)

#define KK_TOPK_INSERT_TIE(vals, idxs, v, i)              \
    do {                                                  \
        if (!KK_TOPK_BETTER_TIE((v), (i), (vals)[K - 1], (idxs)[K - 1])) { \
            break;                                        \
        }                                                 \
        uint _pos = K - 1;                                \
        for (; _pos > 0; --_pos) {                        \
            float _pv = (vals)[_pos - 1];                 \
            uint _pi = (idxs)[_pos - 1];                  \
            if (!KK_TOPK_BETTER_TIE((v), (i), _pv, _pi)) { \
                break;                                    \
            }                                             \
            (vals)[_pos] = _pv;                           \
            (idxs)[_pos] = _pi;                           \
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


@cache
def _moe_combine_kernel_fp32_no_fma(d: int, k: int) -> Any:
    """FP32 combine kernel with FMA contraction disabled."""
    D = int(d)
    K = int(k)
    source = f"""
        #pragma clang fp contract(off)
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;

        threadgroup float wbuf[K];
        for (uint i = tid; i < K; i += threads_per_threadgroup.x) {{
            wbuf[i] = weights[token_idx * K + i];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float acc = 0.0f;
        for (uint i = 0; i < K; ++i) {{
            float w = wbuf[i];
            float v = (float)expert_outputs[(token_idx * K + i) * D + d_idx];
            float prod = w * v;
            acc = acc + prod;
        }}
        out[token_idx * D + d_idx] = acc;
    """
    return metal_kernel(
        name=f"kk_moe_combine_fp32_no_fma_D{D}_K{K}",
        input_names=["expert_outputs", "weights"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def moe_combine_fp32_no_fma(expert_outputs: Any, weights: Any) -> Any:
    """Like ``moe_combine_fp32`` but disables FMA contraction in accumulation."""
    original_shape = weights.shape[:-1]
    K = weights.shape[-1]
    D = expert_outputs.shape[-1]

    expert_outputs_flat = expert_outputs.reshape(-1, K, D)
    weights_flat = weights.reshape(-1, K).astype(mx.float32)
    B = weights_flat.shape[0]

    k = _moe_combine_kernel_fp32_no_fma(D, K)
    out = k(
        expert_outputs_flat,
        weights_flat,
        grid=(D, B, 1),
        threadgroup=(min(D, 256), 1, 1),
        output_shapes=[(B, D)],
        output_dtypes=[mx.float32],
    )[0]
    return out.reshape((*original_shape, D))


def _combine_projected_experts(
    projected: Any,
    gate: Any,
    *,
    no_fma: bool,
    output_dtype: Any | None,
) -> Any:
    """Combine pre-projected expert outputs with gating weights."""
    out_dtype = projected.dtype if output_dtype is None else output_dtype

    if out_dtype == mx.float32:
        gate_fp32 = gate.astype(mx.float32) if gate.dtype != mx.float32 else gate
        return (
            moe_combine_fp32_no_fma(projected, gate_fp32)
            if no_fma
            else moe_combine_fp32(projected, gate_fp32)
        )

    gate_cast = gate.astype(projected.dtype) if gate.dtype != projected.dtype else gate
    out = moe_combine_no_fma(projected, gate_cast) if no_fma else moe_combine(projected, gate_cast)
    if out.dtype != out_dtype:
        out = out.astype(out_dtype)
    return out


def gather_mmk_combine(
    act: Any,
    weights: Any,
    gate: Any,
    indices: Any,
    *,
    no_fma: bool = False,
    output_dtype: Any | None = None,
) -> Any:
    """Vectorized dense gather-matmul-combine using ``mx.gather_mm``.

    This path computes all ``K`` expert projections in one ``gather_mm`` call,
    then combines via ZMLX MoE combine kernels.
    """
    if act.ndim != 3:
        raise ValueError("gather_mmk_combine: act must have shape (B, K, D_in)")
    if weights.ndim != 3:
        raise ValueError("gather_mmk_combine: weights must have shape (E, D_in, D_out)")

    B, K, D_in = act.shape
    if weights.shape[1] != D_in:
        raise ValueError(
            "gather_mmk_combine: weights second dimension must match act last dimension"
        )

    rhs_indices = indices.astype(mx.uint32) if indices.dtype != mx.uint32 else indices
    lhs_indices = mx.arange(B * K, dtype=mx.uint32).reshape(B, K)

    proj = mx.gather_mm(
        act.reshape(B * K, 1, D_in),
        weights,
        lhs_indices=lhs_indices,
        rhs_indices=rhs_indices,
    ).squeeze(axis=-2)

    return _combine_projected_experts(
        proj,
        gate,
        no_fma=no_fma,
        output_dtype=output_dtype,
    )


def gather_qmmk_combine_quantized(
    act: Any,
    weights: Any,
    scales: Any,
    biases: Any,
    gate: Any,
    indices: Any,
    *,
    group_size: int = 64,
    bits: int = 4,
    no_fma: bool = False,
    output_dtype: Any | None = None,
) -> Any:
    """Vectorized quantized gather-QMM-combine using ``mx.gather_qmm``."""
    if act.ndim != 3:
        raise ValueError("gather_qmmk_combine_quantized: act must have shape (B, K, D_in)")

    B, K, D_in = act.shape
    rhs_indices = indices.astype(mx.uint32) if indices.dtype != mx.uint32 else indices
    lhs_indices = mx.arange(B * K, dtype=mx.uint32).reshape(B, K)

    proj = mx.gather_qmm(
        act.reshape(B * K, 1, D_in),
        weights,
        scales,
        biases,
        lhs_indices=lhs_indices,
        rhs_indices=rhs_indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
    ).squeeze(axis=-2)

    return _combine_projected_experts(
        proj,
        gate,
        no_fma=no_fma,
        output_dtype=output_dtype,
    )


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

    # Full-softmax, unnormalized top-k path must match MLX reference exactly.
    # Keep this path on pure MLX ops to avoid set differences in edge tie cases.
    if bias is None and not norm:
        gates = mx.softmax(x_cast, axis=-1)
        inds = mx.argpartition(gates, kth=-K, axis=-1)[..., -K:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        return scores, inds.astype(mx.uint32)

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
def _router_argpartition_logits_topk_kernel(d: int, k: int) -> Any:
    D = int(d)
    K = int(k)
    if D <= 0:
        raise ValueError("router_argpartition_logits_topk: last dimension must be > 0")
    if K <= 0:
        raise ValueError("router_argpartition_logits_topk: k must be > 0")
    if K > D:
        raise ValueError(f"router_argpartition_logits_topk: k={K} exceeds D={D}")

    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};

        uint row = thread_position_in_grid.x;
        uint logits_base = row * D;
        uint idx_base = row * K;

        float vals[K];
        float m = -INFINITY;
        for (uint i = 0; i < K; ++i) {{
            uint idx = (uint)indices[idx_base + i];
            float v = (float)logits[logits_base + idx];
            vals[i] = v;
            if (v > m) m = v;
        }}

        float sum = 0.0f;
        for (uint i = 0; i < K; ++i) {{
            sum += metal::exp(vals[i] - m);
        }}
        float inv = 1.0f / sum;

        for (uint i = 0; i < K; ++i) {{
            weights[idx_base + i] = (T)(metal::exp(vals[i] - m) * inv);
        }}
    """
    return metal_kernel(
        name=f"kk_router_argpartition_logits_topk_D{D}_K{K}",
        input_names=["logits", "indices"],
        output_names=["weights"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def router_argpartition_logits_topk(
    logits: Any,
    *,
    k: int,
    compute_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """Qwen-style argpartition(logits) + top-k softmax without index reordering.

    This preserves argpartition index ordering semantics, then computes softmax
    over the selected logits only.
    """
    if logits.ndim < 1:
        raise ValueError("router_argpartition_logits_topk: logits must have rank >= 1")
    cd = compute_dtype or mx.float32
    x = logits.astype(cd) if logits.dtype != cd else logits
    D = int(x.shape[-1])
    K = int(k)
    if K <= 0:
        raise ValueError("router_argpartition_logits_topk: k must be > 0")
    if K > D:
        raise ValueError(f"router_argpartition_logits_topk: k={K} exceeds D={D}")

    inds = mx.argpartition(x, kth=-K, axis=-1)[..., -K:]
    inds_u32 = inds.astype(mx.uint32) if inds.dtype != mx.uint32 else inds

    # Keep a pure-MLX fallback for large K to avoid expensive kernel variants.
    if K > 16:
        topk_logits = mx.take_along_axis(x, inds_u32, axis=-1)
        scores = mx.softmax(topk_logits, axis=-1, precise=True)  # type: ignore[call-arg]
        return scores, inds_u32

    rows = x.size // D
    kernel = _router_argpartition_logits_topk_kernel(D, K)
    scores = kernel(
        x,
        inds_u32,
        template=[("T", cd)],
        grid=(rows, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[x.shape[:-1] + (K,)],
        output_dtypes=[cd],
    )[0]
    return scores, inds_u32


# ---------------------------------------------------------------------------
# DeepSeek/Kimi router: sigmoid affinity + bias (selection only)
# ---------------------------------------------------------------------------

_DEEPSEEK_ROUTER_SUPPORTED_EXPERTS = {256, 384}
_DEEPSEEK_ROUTER_K = 8


@cache
def _deepseek_router_topk_sigmoid_kernel(d: int, tg: int, n_group: int, topk_group: int) -> Any:
    D = int(d)
    TG = _validate_tg(tg)
    K = int(_DEEPSEEK_ROUTER_K)
    N_GROUP = int(n_group)
    TOPK_GROUP = int(topk_group)
    if D not in _DEEPSEEK_ROUTER_SUPPORTED_EXPERTS:
        raise ValueError(
            f"deepseek_router_topk_sigmoid: fused kernel supports D in "
            f"{sorted(_DEEPSEEK_ROUTER_SUPPORTED_EXPERTS)}, got D={D}"
        )
    if N_GROUP <= 0:
        raise ValueError("deepseek_router_topk_sigmoid: n_group must be > 0")
    if TOPK_GROUP <= 0 or TOPK_GROUP > N_GROUP:
        raise ValueError("deepseek_router_topk_sigmoid: invalid topk_group")
    if D % N_GROUP != 0:
        raise ValueError("deepseek_router_topk_sigmoid: n_group must divide D")
    if N_GROUP > TG:
        raise ValueError("deepseek_router_topk_sigmoid: n_group must be <= threadgroup size")

    source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        constexpr uint TG = {TG};
        constexpr uint N_GROUP = {N_GROUP};
        constexpr uint TOPK_GROUP = {TOPK_GROUP};
        constexpr uint GROUP_SIZE = D / N_GROUP;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float score_buf[D];
        threadgroup float group_scores[N_GROUP];
        threadgroup uint keep_mask[N_GROUP];

        threadgroup float val_buf[TG * K];
        threadgroup uint idx_buf[TG * K];

        thread float vals[K];
        thread uint idxs[K];
        KK_TOPK_INIT_TIE(vals, idxs);

        // Pass 1: compute selection scores (sigmoid(logits) + bias)
        for (uint j = tid; j < D; j += TG) {{
            float logit = (float)logits[base + j];
            float affinity = (float)kk_sigmoid(logit);
            score_buf[j] = affinity + (float)bias[j];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Optional: group selection (mask lowest-scoring groups to 0.0)
        if (N_GROUP > 1) {{
            if (tid < N_GROUP) {{
                uint start = tid * GROUP_SIZE;
                float top1 = -INFINITY;
                float top2 = -INFINITY;
                for (uint j = 0; j < GROUP_SIZE; ++j) {{
                    float v = score_buf[start + j];
                    if (v > top1) {{
                        top2 = top1;
                        top1 = v;
                    }} else if (v > top2) {{
                        top2 = v;
                    }}
                }}
                group_scores[tid] = top1 + top2;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0) {{
                // Keep TOPK_GROUP groups with highest group_scores.
                float best_vals[TOPK_GROUP];
                uint best_idx[TOPK_GROUP];
                for (uint i = 0; i < TOPK_GROUP; ++i) {{
                    best_vals[i] = -INFINITY;
                    best_idx[i] = 0;
                }}
                for (uint g = 0; g < N_GROUP; ++g) {{
                    float v = group_scores[g];
                    uint pos = TOPK_GROUP;
                    for (uint i = 0; i < TOPK_GROUP; ++i) {{
                        float cur = best_vals[i];
                        uint cur_i = best_idx[i];
                        bool better = (v > cur) || ((v == cur) && (g < cur_i));
                        if (better) {{ pos = i; break; }}
                    }}
                    if (pos < TOPK_GROUP) {{
                        for (uint i = TOPK_GROUP - 1; i > pos; --i) {{
                            best_vals[i] = best_vals[i - 1];
                            best_idx[i] = best_idx[i - 1];
                        }}
                        best_vals[pos] = v;
                        best_idx[pos] = g;
                    }}
                }}
                for (uint g = 0; g < N_GROUP; ++g) {{
                    keep_mask[g] = 0;
                }}
                for (uint i = 0; i < TOPK_GROUP; ++i) {{
                    keep_mask[best_idx[i]] = 1;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }} else {{
            if (tid == 0) {{
                keep_mask[0] = 1;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Pass 2: local top-k on masked selection scores
        for (uint j = tid; j < D; j += TG) {{
            float v = score_buf[j];
            if (N_GROUP > 1) {{
                uint g = j / GROUP_SIZE;
                if (keep_mask[g] == 0) {{
                    v = 0.0f;
                }}
            }}
            KK_TOPK_INSERT_TIE(vals, idxs, v, j);
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
                    KK_TOPK_INSERT_TIE(vals, idxs, v, idx);
                }}
                for (uint i = 0; i < K; ++i) {{
                    val_buf[off + i] = vals[i];
                    idx_buf[off + i] = idxs[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0) {{
            float w_vals[K];
            float sum = 0.0f;
            for (uint i = 0; i < K; ++i) {{
                uint idx = idx_buf[i];
                float logit = (float)logits[base + idx];
                float affinity = (float)kk_sigmoid(logit);
                w_vals[i] = affinity;
                sum += affinity;
            }}
            float inv = 1.0f / sum;
            uint out_base = row * K;
            for (uint i = 0; i < K; ++i) {{
                weights[out_base + i] = (T)(w_vals[i] * inv);
                indices[out_base + i] = idx_buf[i];
            }}
        }}
    """
    return metal_kernel(
        name=f"kk_deepseek_router_topk_sigmoid_D{D}_K{K}_TG{TG}",
        input_names=["logits", "bias"],
        output_names=["weights", "indices"],
        source=source,
        header=DEFAULT_HEADER + _TOPK_TIE_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def _deepseek_router_topk_sigmoid_reference(
    logits: Any,
    bias: Any,
    *,
    k: int,
    n_group: int,
    topk_group: int,
) -> tuple[Any, Any]:
    """Pure-MLX reference: DeepSeek group selection + stable top-k."""
    K = int(k)
    if K <= 0:
        raise ValueError("deepseek_router_topk_sigmoid: k must be > 0")
    if logits.ndim < 1:
        raise ValueError("deepseek_router_topk_sigmoid: logits must have rank >= 1")
    D = int(logits.shape[-1])
    if K > D:
        raise ValueError(f"deepseek_router_topk_sigmoid: k={K} exceeds D={D}")
    if bias.ndim != 1 or int(bias.shape[0]) != D:
        raise ValueError(f"deepseek_router_topk_sigmoid: bias must have shape ({D},)")

    N_GROUP = int(n_group)
    TOPK_GROUP = int(topk_group)
    if N_GROUP <= 0:
        raise ValueError("deepseek_router_topk_sigmoid: n_group must be > 0")
    if TOPK_GROUP <= 0 or TOPK_GROUP > N_GROUP:
        raise ValueError("deepseek_router_topk_sigmoid: invalid topk_group")
    if D % N_GROUP != 0:
        raise ValueError("deepseek_router_topk_sigmoid: n_group must divide D")

    affinity = mx.sigmoid(logits)
    scores = affinity + bias

    if N_GROUP > 1 and TOPK_GROUP < N_GROUP:
        per_group = D // N_GROUP
        grouped = mx.unflatten(scores, axis=-1, shape=(N_GROUP, per_group))
        group_scores = mx.topk(grouped, 2, axis=-1).sum(axis=-1, keepdims=False)
        k_drop = N_GROUP - TOPK_GROUP
        drop_idx = mx.argpartition(group_scores, kth=k_drop - 1, axis=-1)[..., :k_drop]
        keep = mx.ones_like(group_scores)
        keep = mx.put_along_axis(keep, drop_idx, mx.array(0.0, dtype=keep.dtype), axis=-1)
        grouped = grouped * keep[..., :, None]
        scores = mx.flatten(grouped, -2, -1)

    idx = mx.arange(D, dtype=mx.int32)
    neg_inf = mx.array(-float("inf"), dtype=scores.dtype)

    chosen_idx: list[Any] = []
    chosen_aff: list[Any] = []
    work = scores
    for _ in range(K):
        max_val = mx.max(work, axis=-1, keepdims=True)
        mask = work == max_val
        min_idx = mx.min(mx.where(mask, idx, D), axis=-1)
        chosen_idx.append(min_idx)
        chosen_aff.append(mx.take_along_axis(affinity, min_idx[..., None], axis=-1)[..., 0])
        work = mx.where(idx == min_idx[..., None], neg_inf, work)

    indices = mx.stack(chosen_idx, axis=-1).astype(mx.uint32)
    weights = mx.stack(chosen_aff, axis=-1)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return weights, indices


def deepseek_router_topk_sigmoid(
    logits: Any,
    bias: Any,
    *,
    k: int = _DEEPSEEK_ROUTER_K,
    n_group: int = 1,
    topk_group: int = 1,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """DeepSeek/Kimi router top-k with sigmoid affinity and selection bias.

    The selection scores are ``sigmoid(logits) + bias``, but the returned weights
    are the normalized sigmoid values (bias does not affect weights).

    Args:
        logits: Router logits with shape ``(..., Nr)``.
        bias: Expert bias with shape ``(Nr,)`` (applied only for top-k selection).
        k: Top-k. Currently optimized for ``k=8``.
        threadgroup: Threadgroup size for the fused kernel.
        compute_dtype: Optional dtype for fused compute + output weights.

    Returns:
        weights: ``(..., k)`` normalized sigmoid affinities.
        indices: ``(..., k)`` expert indices (uint32), ordered by score desc and
            index asc (stable tie-break).
    """
    cd = compute_dtype or mx.float32
    logits_cast = logits.astype(cd) if logits.dtype != cd else logits
    D = int(logits_cast.shape[-1])

    if bias.ndim != 1:
        if int(bias.size) == D:
            bias = bias.reshape((D,))
        else:
            raise ValueError(f"deepseek_router_topk_sigmoid: bias must have shape ({D},)")
    if int(bias.shape[0]) != D:
        raise ValueError(f"deepseek_router_topk_sigmoid: bias must have shape ({D},)")
    bias_cast = bias.astype(cd) if bias.dtype != cd else bias

    K = int(k)
    N_GROUP = int(n_group)
    TOPK_GROUP = int(topk_group)
    if (
        K != _DEEPSEEK_ROUTER_K
        or D not in _DEEPSEEK_ROUTER_SUPPORTED_EXPERTS
        or N_GROUP <= 0
        or TOPK_GROUP <= 0
        or TOPK_GROUP > N_GROUP
        or D % N_GROUP != 0
    ):
        # Reference fallback (group selection + stable tie-break) for unsupported shapes.
        return _deepseek_router_topk_sigmoid_reference(
            logits_cast,
            bias_cast,
            k=K,
            n_group=N_GROUP,
            topk_group=TOPK_GROUP,
        )

    TG = _validate_tg(threadgroup)
    kernel = _deepseek_router_topk_sigmoid_kernel(D, TG, N_GROUP, TOPK_GROUP)
    rows = logits_cast.size // D
    weights, indices = kernel(
        logits_cast,
        bias_cast,
        template=[("T", cd)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[logits_cast.shape[:-1] + (K,), logits_cast.shape[:-1] + (K,)],
        output_dtypes=[cd, mx.uint32],
    )
    return weights, indices


@cache
def _weighted_accumulate_kernel(d_out: int) -> Any:
    """Fused out = acc + gate_weight * projection — avoids materializing gate * proj."""
    D_out = int(d_out)
    source = f"""
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        constexpr uint D = {D_out};
        uint elem = row * D + col;
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
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        constexpr uint D = {D_out};
        uint elem = row * D + col;
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


@cache
def _weighted_accumulate_kernel_fp32(d_out: int) -> Any:
    """Like `_weighted_accumulate_kernel` but accumulates/outputs float32."""
    D_out = int(d_out)
    source = f"""
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        constexpr uint D = {D_out};
        uint elem = row * D + col;
        float w = (float)gate[row];
        float v = (float)proj[elem];
        out[elem] = (float)acc[elem] + w * v;
    """
    return metal_kernel(
        name=f"kk_weighted_accumulate_fp32_D{D_out}",
        input_names=["acc", "proj", "gate"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


@cache
def _weighted_accumulate_kernel_fp32_no_fma(d_out: int) -> Any:
    """FP32 accumulate/output variant with FMA contraction disabled."""
    D_out = int(d_out)
    source = f"""
        #pragma clang fp contract(off)
        uint col = thread_position_in_grid.x;
        uint row = thread_position_in_grid.y;
        constexpr uint D = {D_out};
        uint elem = row * D + col;
        float w = (float)gate[row];
        float v = (float)proj[elem];
        float prod = w * v;
        out[elem] = (float)acc[elem] + prod;
    """
    return metal_kernel(
        name=f"kk_weighted_accumulate_fp32_no_fma_D{D_out}",
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
    output_dtype: Any | None = None,
    vectorized_k: bool = False,
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
        output_dtype: Optional output dtype. When set to ``mx.float32``, combine
            accumulates and returns float32 to match promotion-sensitive paths.
        vectorized_k: When True, use ``mx.gather_mm`` across all ``K`` experts
            in one call, then combine with ZMLX MoE combine kernels.

    Returns:
        Combined output, shape ``(B, D_out)``.
    """
    if act.ndim != 3:
        raise ValueError("gather_qmm_combine: act must have shape (B, K, D_in)")
    B, K, _ = act.shape
    if weights.ndim != 3:
        raise ValueError("gather_qmm_combine: weights must have shape (E, D_in, D_out)")
    D_out = int(weights.shape[2])
    out_dtype = act.dtype if output_dtype is None else output_dtype

    if vectorized_k:
        return gather_mmk_combine(
            act,
            weights,
            gate,
            indices,
            no_fma=no_fma,
            output_dtype=out_dtype,
        )

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
        out = mx.sum(proj_stacked * mx.expand_dims(gate, axis=-1), axis=1)
        if out.dtype != out_dtype:
            out = out.astype(out_dtype)
        return out

    # Streaming: accumulate without (B, K, D_out) intermediate
    if out_dtype == mx.float32:
        k_acc = (
            _weighted_accumulate_kernel_fp32_no_fma(D_out)
            if no_fma
            else _weighted_accumulate_kernel_fp32(D_out)
        )
        k_acc_template = None
    else:
        k_acc = (
            _weighted_accumulate_kernel_no_fma(D_out)
            if no_fma
            else _weighted_accumulate_kernel(D_out)
        )
        k_acc_template = [("T", out_dtype)]

    output = mx.zeros((B, D_out), dtype=out_dtype)
    for k_idx in range(K):
        a_k = act[:, k_idx, :]
        w_k = weights[indices[:, k_idx]]
        proj_k = mx.matmul(mx.expand_dims(a_k, axis=1), w_k).squeeze(axis=1)
        gate_k = gate[:, k_idx]

        kernel_kwargs: dict[str, Any] = {}
        if k_acc_template is not None:
            kernel_kwargs["template"] = k_acc_template
        output = k_acc(
            output,
            proj_k,
            gate_k,
            grid=(D_out, B, 1),
            threadgroup=(min(D_out, 256), 1, 1),
            output_shapes=[(B, D_out)],
            output_dtypes=[out_dtype],
            **kernel_kwargs,
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
    output_dtype: Any | None = None,
    vectorized_k: bool = False,
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
        output_dtype: Optional output dtype. When set to ``mx.float32``, combine
            accumulates and returns float32 to match promotion-sensitive paths.
        vectorized_k: When True, use one ``mx.gather_qmm`` over all ``K``
            experts via batch-level gather indices.

    Returns:
        Combined output, shape ``(B, D_out)``.
    """
    if act.ndim != 3:
        raise ValueError(
            "gather_qmm_combine_quantized: act must have shape (B, K, D_in)"
        )
    B, K, _ = act.shape
    D_out = int(weights.shape[1])
    out_dtype = act.dtype if output_dtype is None else output_dtype

    if vectorized_k:
        return gather_qmmk_combine_quantized(
            act,
            weights,
            scales,
            biases,
            gate,
            indices,
            group_size=group_size,
            bits=bits,
            no_fma=no_fma,
            output_dtype=out_dtype,
        )

    if not _should_use_streaming(B, streaming):
        # Non-streaming: gather_qmm per expert then combine
        proj_all = []
        lhs_indices = mx.arange(B, dtype=mx.uint32)
        for k_idx in range(K):
            a_k = act[:, k_idx, :]
            proj_k = mx.gather_qmm(
                mx.expand_dims(a_k, axis=1),
                weights,
                scales,
                biases,
                lhs_indices=lhs_indices,
                rhs_indices=indices[:, k_idx],
                transpose=True,
                group_size=group_size,
                bits=bits,
            ).squeeze(axis=1)
            proj_all.append(proj_k)
        proj_stacked = mx.stack(proj_all, axis=1)
        out = mx.sum(proj_stacked * mx.expand_dims(gate, axis=-1), axis=1)
        if out.dtype != out_dtype:
            out = out.astype(out_dtype)
        return out

    # Streaming: accumulate without (B, K, D_out) intermediate
    if out_dtype == mx.float32:
        k_acc = (
            _weighted_accumulate_kernel_fp32_no_fma(D_out)
            if no_fma
            else _weighted_accumulate_kernel_fp32(D_out)
        )
        k_acc_template = None
    else:
        k_acc = (
            _weighted_accumulate_kernel_no_fma(D_out)
            if no_fma
            else _weighted_accumulate_kernel(D_out)
        )
        k_acc_template = [("T", out_dtype)]

    output = mx.zeros((B, D_out), dtype=out_dtype)
    lhs_indices = mx.arange(B, dtype=mx.uint32)
    for k_idx in range(K):
        a_k = act[:, k_idx, :]
        proj_k = mx.gather_qmm(
            mx.expand_dims(a_k, axis=1),
            weights,
            scales,
            biases,
            lhs_indices=lhs_indices,
            rhs_indices=indices[:, k_idx],
            transpose=True,
            group_size=group_size,
            bits=bits,
        ).squeeze(axis=1)
        gate_k = gate[:, k_idx]

        kernel_kwargs: dict[str, Any] = {}
        if k_acc_template is not None:
            kernel_kwargs["template"] = k_acc_template
        output = k_acc(
            output,
            proj_k,
            gate_k,
            grid=(D_out, B, 1),
            threadgroup=(min(D_out, 256), 1, 1),
            output_shapes=[(B, D_out)],
            output_dtypes=[out_dtype],
            **kernel_kwargs,
        )[0]

    return output


__all__ = [
    "top2_gating_softmax",
    "topk_gating_softmax",
    "router_argpartition_logits_topk",
    "deepseek_router_topk_sigmoid",
    "moe_dispatch",
    "moe_combine",
    "moe_combine_no_fma",
    "moe_combine_exact",
    "moe_combine_fp32",
    "moe_combine_fp32_no_fma",
    "gather_mmk_combine",
    "gather_qmmk_combine_quantized",
    "gather_qmm_combine",
    "gather_qmm_combine_quantized",
]
