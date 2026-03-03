"""Fused DeltaNet kernels for Qwen3.5 decode optimization.

These kernels target the GatedDeltaNet layer (30 of 40 layers in Qwen3.5-35B),
which accounts for ~66% of total decode time. Each kernel fuses multiple Metal
dispatches into one, reducing dispatch overhead at M=1 decode.

Kernels:
    fused_conv1d_silu: Depthwise conv1d + SiLU activation for decode (M=1).
    gated_rmsnorm_silu: RMSNorm(y) * SiLU(z) output gating.
    fused_input_proj: Fuse 4 input projections into 1 concatenated matmul.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


# ---------------------------------------------------------------------------
# Fused Conv1d + SiLU decode kernel (SHA-1417)
# ---------------------------------------------------------------------------
#
# At decode time (M=1), depthwise conv1d is just a dot product per channel:
#   for each channel c:
#     out[c] = dot(conv_state[c, :], weight[c, :])   # kernel_size taps
#     out[c] = silu(out[c])
#
# This replaces 2 Metal dispatches (conv1d + silu) with 1.
#
# Inputs:
#   conv_input: [B, kernel_size, conv_dim] — concatenated [conv_state, new_token]
#   weight: [conv_dim, 1, kernel_size, 1] or [conv_dim, 1, 1, kernel_size]
#     (depthwise conv weight, one per channel)
# Output:
#   out: [B, 1, conv_dim] — activated output
#
# Also outputs updated conv_state: conv_input[:, 1:, :] (last kernel_size-1 tokens)

@cache
def _fused_conv1d_silu_kernel(conv_dim: int, kernel_size: int, tg: int) -> Any:
    CD = int(conv_dim)
    KS = int(kernel_size)
    TG = int(tg)

    source = f"""
        constexpr uint CD = {CD};
        constexpr uint KS = {KS};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint batch = gid / (CD);
        uint ch = gid % (CD);

        if (batch * CD + ch >= threads_per_grid.x) return;

        // conv_input layout: [B, KS, CD] — row-major
        // weight layout: [CD, 1, KS, 1] → effectively [CD, KS] after reshape
        float acc = 0.0f;
        for (uint k = 0; k < KS; ++k) {{
            float x_val = (float)conv_input[(batch * KS + k) * CD + ch];
            float w_val = (float)weight[ch * KS + k];
            acc += x_val * w_val;
        }}

        // SiLU activation: x * sigmoid(x)
        float result = kk_silu((T)acc);

        // Output: [B, 1, CD]
        out[batch * CD + ch] = (T)result;

        // Updated conv_state: [B, KS-1, CD] = conv_input[:, 1:, :]
        for (uint k = 0; k < KS - 1u; ++k) {{
            new_state[(batch * (KS - 1u) + k) * CD + ch] =
                conv_input[(batch * KS + k + 1u) * CD + ch];
        }}
    """

    return metal_kernel(
        name=f"kk_fused_conv1d_silu_CD{CD}_KS{KS}",
        input_names=["conv_input", "weight"],
        output_names=["out", "new_state"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def fused_conv1d_silu(
    conv_input: mx.array,
    weight: mx.array,
    *,
    threadgroup: int = 256,
) -> tuple[mx.array, mx.array]:
    """Fused depthwise conv1d + SiLU for decode (M=1).

    Args:
        conv_input: [B, kernel_size, conv_dim] — concatenated conv state + new token
        weight: [conv_dim, 1, kernel_size, 1] — depthwise conv weights
                (or [conv_dim, kernel_size] after reshape)

    Returns:
        out: [B, 1, conv_dim] — activated output
        new_state: [B, kernel_size-1, conv_dim] — updated conv state
    """
    B = conv_input.shape[0]
    KS = conv_input.shape[1]
    CD = conv_input.shape[2]

    # Normalize weight shape to [conv_dim, kernel_size]
    w = weight.reshape(CD, KS)

    TG = min(threadgroup, CD)
    total_threads = B * CD
    grid_x = ((total_threads + TG - 1) // TG) * TG

    k = _fused_conv1d_silu_kernel(CD, KS, TG)

    out, new_state = k(
        conv_input, w,
        template=[("T", conv_input.dtype)],
        grid=(grid_x, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[(B, 1, CD), (B, KS - 1, CD)],
        output_dtypes=[conv_input.dtype, conv_input.dtype],
    )
    return out, new_state


# ---------------------------------------------------------------------------
# Fused Gated RMSNorm + SiLU kernel (SHA-1420)
# ---------------------------------------------------------------------------
#
# Computes: out = RMSNorm(y, weight, eps) * SiLU(z)
#
# This is the Qwen3NextRMSNormGated pattern:
#   x = rms_norm(y)    — rowwise reduction + scale
#   g = silu(z)        — elementwise
#   out = x * g        — elementwise
#
# Fuses 3 dispatches (rmsnorm + silu + multiply) into 1.
#
# Shape: y, z: [B, S, Hv, Dv], weight: [Dv], out: [B, S, Hv, Dv]
# At decode: B=1, S=1, so this is [1, 1, Hv, Dv] → Hv rows of Dv elements.

@cache
def _gated_rmsnorm_silu_kernel(d: int, tg: int, eps: float) -> Any:
    D = int(d)
    TG = int(tg)
    eps_f = float(eps)

    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // Step 1: Compute sum of squares for RMSNorm of y
        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)y[base + j];
            sumsq += v * v;
        }}
        KK_SIMD_REDUCE_SUM(buf, sumsq, tid, TG);

        float inv_rms = metal::rsqrt(buf[0] / (float)D + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: Fused output = RMSNorm(y) * SiLU(z)
        //   RMSNorm(y)[j] = y[j] * inv_rms * weight[j]
        //   SiLU(z)[j] = z[j] * sigmoid(z[j])
        for (uint j = tid; j < D; j += TG) {{
            float y_val = (float)y[base + j];
            float w_val = (float)weight[j];
            float y_normed = y_val * inv_rms * w_val;

            float z_val = (float)z[base + j];
            float z_silu = kk_silu((T)z_val);

            out[base + j] = (T)(y_normed * (float)z_silu);
        }}
    """

    eps_str = str(eps_f).replace(".", "_").replace("-", "_")

    return metal_kernel(
        name=f"kk_gated_rmsnorm_silu_D{D}_TG{TG}_E{eps_str}",
        input_names=["y", "z", "weight"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def gated_rmsnorm_silu(
    y: mx.array,
    z: mx.array,
    weight: mx.array,
    *,
    eps: float = 1e-6,
    threadgroup: int = 256,
) -> mx.array:
    """Fused gated RMSNorm + SiLU: RMSNorm(y, weight, eps) * SiLU(z).

    This is the Qwen3NextRMSNormGated pattern used in DeltaNet output gating.

    Args:
        y: (..., D) — input to RMSNorm
        z: (..., D) — gate input for SiLU
        weight: (D,) — RMSNorm per-channel weights
        eps: RMSNorm epsilon

    Returns:
        out: (..., D) — RMSNorm(y) * SiLU(z)
    """
    if y.shape != z.shape:
        raise ValueError(f"y and z must have same shape, got {y.shape} vs {z.shape}")
    D = int(y.shape[-1])
    if weight.ndim != 1 or int(weight.shape[0]) != D:
        raise ValueError(f"weight must have shape ({D},), got {weight.shape}")

    TG = min(threadgroup, D)
    # Ensure TG is a power of 2 for SIMD reduction
    tg_pow2 = 1
    while tg_pow2 * 2 <= TG:
        tg_pow2 *= 2
    TG = max(tg_pow2, 32)

    rows = y.size // D
    k = _gated_rmsnorm_silu_kernel(D, TG, float(eps))

    out = k(
        y, z, weight,
        template=[("T", y.dtype)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[y.shape],
        output_dtypes=[y.dtype],
    )[0]
    return out


# ---------------------------------------------------------------------------
# Fused input projections (dispatch reduction)
# ---------------------------------------------------------------------------
# The qwen3_5 GatedDeltaNet has 4 separate nn.Linear calls:
#   qkv = in_proj_qkv(x)    — [hidden, key_dim*2 + value_dim]
#   z   = in_proj_z(x)      — [hidden, value_dim]
#   b   = in_proj_b(x)      — [hidden, num_v_heads]
#   a   = in_proj_a(x)      — [hidden, num_v_heads]
#
# These all read the same input x. We can concatenate the weight matrices
# and do a single matmul, then split the output.
#
# This is NOT a custom Metal kernel — just a Python-level fusion that
# replaces 4 dispatches with 1 matmul + 1 split.

def fused_input_proj(
    x: mx.array,
    w_qkv: mx.array,
    w_z: mx.array,
    w_b: mx.array,
    w_a: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fuse 4 input projections into 1 concatenated matmul.

    All 4 projections read the same input x and have no bias.
    We concatenate weights along output dim and split after.

    Args:
        x: [B, S, hidden_size]
        w_qkv: [key_dim*2 + value_dim, hidden_size] — in_proj_qkv weight
        w_z: [value_dim, hidden_size] — in_proj_z weight
        w_b: [num_v_heads, hidden_size] — in_proj_b weight
        w_a: [num_v_heads, hidden_size] — in_proj_a weight

    Returns:
        qkv, z, b, a — split outputs matching original projections
    """
    # Concatenate weights: [total_out_dim, hidden_size]
    w_cat = mx.concatenate([w_qkv, w_z, w_b, w_a], axis=0)

    # Single matmul: x @ w_cat.T
    combined = x @ w_cat.T

    # Split at the right boundaries
    d_qkv = w_qkv.shape[0]
    d_z = w_z.shape[0]
    d_b = w_b.shape[0]
    # d_a = w_a.shape[0]

    splits = [d_qkv, d_qkv + d_z, d_qkv + d_z + d_b]
    qkv, z, b, a = mx.split(combined, splits, axis=-1)
    return qkv, z, b, a
