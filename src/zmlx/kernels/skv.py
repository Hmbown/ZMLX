"""SKV kernels: spectral KV compression primitives for MLX/ZMLX.

All public entrypoints are explicitly prefixed with ``skv_`` to make the
integration path easy to audit.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@cache
def _skv_fused_project_quantize_kernel(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    rank: int,
    bits: int,
    group_size: int,
) -> Any:
    S = int(seq_len)
    H = int(num_heads)
    D = int(head_dim)
    R = int(rank)
    BITS = int(bits)
    GS = int(group_size)
    N = S * H * R
    NG = (N + GS - 1) // GS
    QMAX = (1 << BITS) - 1

    source = f"""
        constexpr uint S = {S};
        constexpr uint H = {H};
        constexpr uint D = {D};
        constexpr uint R = {R};
        constexpr uint N = {N};
        constexpr uint GS = {GS};
        constexpr uint NG = {NG};
        constexpr float QMAX = {float(QMAX)}f;

        uint gid = thread_position_in_grid.x;
        if (gid >= NG) {{
            return;
        }}

        uint start = gid * GS;
        float vals[GS];
        float vmin = INFINITY;
        float vmax = -INFINITY;

        for (uint i = 0; i < GS; ++i) {{
            uint idx = start + i;
            if (idx < N) {{
                uint r = idx % R;
                uint t = idx / R;
                uint h = t % H;
                uint s = t / H;

                uint kv_base = (s * H + h) * D;
                uint b_base = (h * D) * R + r;
                float acc = 0.0f;
                for (uint d0 = 0; d0 < D; ++d0) {{
                    acc += (float)kv[kv_base + d0] * (float)basis[b_base + d0 * R];
                }}
                vals[i] = acc;
                vmin = metal::min(vmin, acc);
                vmax = metal::max(vmax, acc);
            }} else {{
                vals[i] = 0.0f;
            }}
        }}

        float scale = (vmax > vmin) ? ((vmax - vmin) / QMAX) : 1.0f;
        scales[gid] = (T)scale;
        zeros[gid] = (T)vmin;

        for (uint i = 0; i < GS; ++i) {{
            uint idx = start + i;
            if (idx < N) {{
                float qf = metal::round((vals[i] - vmin) / scale);
                qf = metal::clamp(qf, 0.0f, QMAX);
                q_data[idx] = (T)qf;
            }}
        }}
    """
    return metal_kernel(
        name=f"skv_fused_project_quantize_s{S}_h{H}_d{D}_r{R}_b{BITS}_g{GS}",
        input_names=["kv", "basis"],
        output_names=["q_data", "scales", "zeros"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def skv_fused_project_quantize(
    kv: Any,
    basis: Any,
    *,
    bits: int = 4,
    group_size: int = 32,
    compute_dtype: Any | None = None,
) -> dict[str, Any]:
    """Project + quantize in one kernel launch.

    Args:
        kv: ``(seq, kv_heads, head_dim)``.
        basis: ``(kv_heads, head_dim, rank)``.
        bits: Quantization bits (2/4/8).
        group_size: Flat group size for per-group affine quantization.

    Returns:
        ``{"q_data","scales","zeros","shape","bits","group_size"}``.
        The quantized payload uses dtype ``compute_dtype`` (default float16) with
        integer-valued entries in ``[0, 2^bits-1]``.
    """
    _require(kv.ndim == 3, "skv_fused_project_quantize: kv must be rank-3")
    _require(basis.ndim == 3, "skv_fused_project_quantize: basis must be rank-3")
    _require(bits in (2, 4, 8), "skv_fused_project_quantize: bits must be one of {2,4,8}")
    _require(group_size > 0, "skv_fused_project_quantize: group_size must be > 0")

    S, H, D = map(int, kv.shape)
    Hb, Db, R = map(int, basis.shape)
    _require(H == Hb and D == Db, "skv_fused_project_quantize: kv/basis shape mismatch")

    N = S * H * R
    G = int(group_size)
    NG = (N + G - 1) // G
    cd = compute_dtype or mx.float16

    k = _skv_fused_project_quantize_kernel(S, H, D, R, int(bits), G)
    q_data, scales, zeros = k(
        kv,
        basis,
        template=[("T", cd)],
        grid=(NG, 1, 1),
        output_shapes=[(S, H, R), (NG,), (NG,)],
        output_dtypes=[cd, cd, cd],
    )
    return {
        "q_data": q_data,
        "scales": scales,
        "zeros": zeros,
        "shape": (S, H, R),
        "bits": int(bits),
        "group_size": G,
    }


@cache
def _skv_fused_dequantize_unproject_kernel(
    seq_len: int,
    num_heads: int,
    head_dim: int,
    rank: int,
    group_size: int,
) -> Any:
    S = int(seq_len)
    H = int(num_heads)
    D = int(head_dim)
    R = int(rank)
    GS = int(group_size)
    NQ = S * H * R
    NOUT = S * H * D

    source = f"""
        constexpr uint S = {S};
        constexpr uint H = {H};
        constexpr uint D = {D};
        constexpr uint R = {R};
        constexpr uint GS = {GS};
        constexpr uint NQ = {NQ};
        constexpr uint NOUT = {NOUT};

        uint gid = thread_position_in_grid.x;
        if (gid >= NOUT) {{
            return;
        }}

        uint d = gid % D;
        uint t = gid / D;
        uint h = t % H;
        uint s = t / H;

        float acc = 0.0f;
        uint b_base = (h * D + d) * R;
        uint q_base = (s * H + h) * R;
        for (uint r = 0; r < R; ++r) {{
            uint q_idx = q_base + r;
            uint g = q_idx / GS;
            float qv = (float)q_data[q_idx];
            float coeff = qv * (float)scales[g] + (float)zeros[g];
            acc += coeff * (float)basis[b_base + r];
        }}
        out[gid] = (T)acc;
    """
    return metal_kernel(
        name=f"skv_fused_dequantize_unproject_s{S}_h{H}_d{D}_r{R}_g{GS}",
        input_names=["q_data", "scales", "zeros", "basis"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def skv_fused_dequantize_unproject(
    state: dict[str, Any],
    basis: Any,
    *,
    compute_dtype: Any | None = None,
) -> Any:
    """Dequantize + unproject in one kernel launch."""
    q_data = state["q_data"]
    scales = state["scales"]
    zeros = state["zeros"]
    S, H, R = map(int, state["shape"])
    Hb, D, Rb = map(int, basis.shape)
    G = int(state["group_size"])

    _require(H == Hb and R == Rb, "skv_fused_dequantize_unproject: state/basis shape mismatch")
    cd = compute_dtype or basis.dtype

    NOUT = S * H * D
    k = _skv_fused_dequantize_unproject_kernel(S, H, D, R, G)
    return k(
        q_data,
        scales,
        zeros,
        basis,
        template=[("T", cd)],
        grid=(NOUT, 1, 1),
        output_shapes=[(S, H, D)],
        output_dtypes=[cd],
    )[0]


@cache
def _skv_compressed_attention_kernel(
    q_len: int,
    kv_len: int,
    num_heads: int,
    num_kv_heads: int,
    rank: int,
    group_size: int,
) -> Any:
    Q = int(q_len)
    K = int(kv_len)
    H = int(num_heads)
    HKV = int(num_kv_heads)
    R = int(rank)
    GS = int(group_size)
    REPEATS = H // HKV
    NQ = K * HKV * R
    NOUT = H * Q * K

    source = f"""
        constexpr uint Q = {Q};
        constexpr uint K = {K};
        constexpr uint H = {H};
        constexpr uint HKV = {HKV};
        constexpr uint R = {R};
        constexpr uint GS = {GS};
        constexpr uint REPEATS = {REPEATS};
        constexpr uint NOUT = {NOUT};
        constexpr uint NQ = {NQ};

        uint gid = thread_position_in_grid.x;
        if (gid >= NOUT) {{
            return;
        }}

        uint k = gid % K;
        uint t = gid / K;
        uint q = t % Q;
        uint h = t / Q;
        uint kv_h = h / REPEATS;

        float acc = 0.0f;
        uint q_base = (q * H + h) * R;
        uint k_base = (k * HKV + kv_h) * R;
        for (uint r = 0; r < R; ++r) {{
            uint k_idx = k_base + r;
            uint g = k_idx / GS;
            float kval = (float)k_q_data[k_idx] * (float)k_scales[g] + (float)k_zeros[g];
            float qval = (float)q_rank[q_base + r];
            acc += qval * kval;
        }}
        out[gid] = (T)(acc * (float)scale[0]);
    """
    return metal_kernel(
        name=f"skv_compressed_attention_q{Q}_k{K}_h{H}_hk{HKV}_r{R}_g{GS}",
        input_names=["q_rank", "k_q_data", "k_scales", "k_zeros", "scale"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def skv_compressed_attention(
    q_rank: Any,
    k_state: dict[str, Any],
    *,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
    compute_dtype: Any | None = None,
) -> Any:
    """Compute ``Q @ K^T`` directly from quantized rank-space keys.

    Args:
        q_rank: ``(q_len, num_heads, rank)``.
        k_state: output of :func:`skv_fused_project_quantize` for key latents.
        num_heads: attention heads for queries.
        num_kv_heads: key-value head count (for GQA mapping).
        scale: attention scale, usually ``head_dim ** -0.5``.

    Returns:
        Scores shaped ``(num_heads, q_len, kv_len)``.
    """
    _require(q_rank.ndim == 3, "skv_compressed_attention: q_rank must be rank-3")
    q_len, q_heads, rank = map(int, q_rank.shape)
    kv_len, kv_heads, kv_rank = map(int, k_state["shape"])
    _require(rank == kv_rank, "skv_compressed_attention: rank mismatch")
    _require(q_heads == int(num_heads), "skv_compressed_attention: q head mismatch")
    _require(kv_heads == int(num_kv_heads), "skv_compressed_attention: kv head mismatch")
    _require(num_heads % num_kv_heads == 0, "skv_compressed_attention: heads must be divisible by kv_heads")

    cd = compute_dtype or mx.float32
    scale_arr = mx.array([float(scale)], dtype=cd)
    NOUT = int(num_heads) * q_len * kv_len

    k = _skv_compressed_attention_kernel(
        q_len, kv_len, int(num_heads), int(num_kv_heads), rank, int(k_state["group_size"])
    )
    out = k(
        q_rank,
        k_state["q_data"],
        k_state["scales"],
        k_state["zeros"],
        scale_arr,
        template=[("T", cd)],
        grid=(NOUT, 1, 1),
        output_shapes=[(int(num_heads), q_len, kv_len)],
        output_dtypes=[cd],
    )[0]
    return out


__all__ = [
    "skv_fused_project_quantize",
    "skv_fused_dequantize_unproject",
    "skv_compressed_attention",
]
