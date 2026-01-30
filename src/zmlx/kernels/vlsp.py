"""VLSP (Verified Latent Self-Play) kernels for recurrent latent reasoning.

Three custom kernels for the VLSP training pipeline:

1. fused_recurrent_step: h_new = h + silu(h_normed) * gate
   Fuses RMSNorm + SiLU gating + residual add per recurrence step.
   At K=8 recurrences, this saves 16+ kernel launches vs separate ops.

2. depth_gate_sigmoid: Predicts variable depth K with STE backward.
   Forward: continuous sigmoid * k_max -> discretized integer K.
   Backward: straight-through estimator (gradient of continuous sigmoid).

3. grpo_advantage_norm: Fused advantage normalization for GRPO.
   A_i = (r_i - mean(r)) / (std(r) + eps) in one reduction pass.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..autograd import unary_from_expr
from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .softmax import _validate_tg

# ---------------------------------------------------------------------------
# Kernel 1: fused_recurrent_step
#
# Per recurrence iteration of the latent block:
#   h_normed = RMSNorm(h, w_norm)
#   h_new = h + alpha * silu(h_normed) * gate
#
# This is the elementwise portion of the recurrent step. The matmul-based
# MLP projections (gate_proj, up_proj, down_proj) are handled separately
# by standard MLX ops. This kernel fuses norm + activation + residual.
# ---------------------------------------------------------------------------

@cache
def _fused_recurrent_step_kernel(d: int, tg: int, eps: float) -> Any:
    """Build Metal kernel for fused RMSNorm + SiLU gating + residual."""
    D = int(d)
    TG = _validate_tg(tg)
    eps_f = float(eps)

    # Two-phase kernel:
    # Phase 1: Parallel reduction for RMS normalization factor
    # Phase 2: Elementwise: h_new = h + alpha * silu(h_normed) * gate
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        // Phase 1: compute RMS normalization factor
        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)h[base + j];
            sumsq += v * v;
        }}
        buf[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                buf[tid] += buf[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float inv_rms = metal::rsqrt(buf[0] / (float)D + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: h_new = h + alpha * silu(normed) * gate
        // where normed = h * inv_rms * w_norm
        // alpha is read from a scalar input
        float a = (float)alpha[0];
        for (uint j = tid; j < D; j += TG) {{
            float hv = (float)h[base + j];
            float wn = (float)w_norm[j];
            float gv = (float)gate[base + j];
            float normed = hv * inv_rms * wn;
            float s = normed * kk_sigmoid(normed);  // silu
            out[base + j] = (T)(hv + a * s * gv);
        }}
    """

    eps_str = str(eps_f).replace(".", "_").replace("-", "_")
    return metal_kernel(
        name=f"kk_vlsp_recurrent_step_D{D}_TG{TG}_E{eps_str}",
        input_names=["h", "w_norm", "gate", "alpha"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def fused_recurrent_step(
    h: Any,
    w_norm: Any,
    gate: Any,
    alpha: Any,
    *,
    eps: float = 1e-6,
    threadgroup: int = 256,
) -> Any:
    """Fused recurrent step: h_new = h + alpha * silu(RMSNorm(h)) * gate.

    This fuses RMSNorm + SiLU activation + gating + residual addition
    into a single Metal kernel launch per recurrence step.

    Args:
        h: Hidden state tensor (..., D)
        w_norm: RMSNorm weight vector (D,)
        gate: Gating tensor (..., D), typically from a learned projection
        alpha: Scalar or (1,) tensor controlling residual scale
        eps: RMSNorm epsilon
        threadgroup: Metal threadgroup size

    Returns:
        h_new: Updated hidden state (..., D)
    """
    D = h.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _fused_recurrent_step_kernel(D, TG, eps)
    rows = h.size // D

    # Ensure alpha is a 1-element array
    if alpha.ndim == 0:
        alpha = alpha.reshape(1)

    return k(
        h, w_norm, gate, alpha,
        template=[("T", mx.float32)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[h.shape],
        output_dtypes=[h.dtype],
    )[0]


# ---------------------------------------------------------------------------
# Kernel 2: depth_gate_sigmoid
#
# Predicts recurrence depth K from hidden states.
# Forward: k_continuous = sigmoid(x) * k_max
#          k_discrete = round(k_continuous)  (for actual computation)
# Backward: straight-through estimator (STE) - gradient of sigmoid * k_max
# ---------------------------------------------------------------------------

@cache
def _depth_gate_sigmoid_op(k_max: int = 8) -> Any:
    """Build differentiable depth gate with STE backward."""
    K_MAX = int(k_max)

    # Forward: sigmoid(x) * k_max
    # We return the continuous value; discretization happens in Python
    # Backward (STE): d/dx = k_max * sigmoid(x) * (1 - sigmoid(x))
    return unary_from_expr(
        name=f"kk_vlsp_depth_gate_K{K_MAX}",
        fwd_expr=f"kk_sigmoid(x) * (T){K_MAX}",
        vjp_expr=f"g * (T){K_MAX} * s * ((T)1 - s)",
        compute_dtype=mx.float32,
        use_output=False,
        vjp_prelude="T s = kk_sigmoid(x);",
        header=DEFAULT_HEADER,
    )


def depth_gate_sigmoid(x: Any, *, k_max: int = 8) -> Any:
    """Predict recurrence depth with differentiable sigmoid gate.

    Forward:
        k_continuous = sigmoid(x) * k_max
    Backward (STE):
        dk/dx = k_max * sigmoid(x) * (1 - sigmoid(x))

    The continuous output should be discretized in the training loop:
        k_discrete = mx.round(k_continuous).astype(mx.int32)
        k_discrete = mx.clip(k_discrete, 1, k_max)

    Args:
        x: Input logits (any shape, typically scalar or (batch,))
        k_max: Maximum recurrence depth

    Returns:
        k_continuous: Continuous depth prediction in [0, k_max]
    """
    op = _depth_gate_sigmoid_op(k_max)
    return op(x)


# ---------------------------------------------------------------------------
# Kernel 3: grpo_advantage_norm
#
# Fused GRPO advantage normalization:
#   A_i = (r_i - mean(r)) / (std(r) + eps)
#
# Single-pass two-phase rowwise reduction + elementwise normalization.
# Each row represents a group of G rollout rewards.
# ---------------------------------------------------------------------------

@cache
def _grpo_advantage_kernel(g: int, tg: int, eps: float) -> Any:
    """Build Metal kernel for fused GRPO advantage normalization."""
    G = int(g)
    TG = _validate_tg(tg)
    eps_f = float(eps)

    source = f"""
        constexpr uint G = {G};
        constexpr uint TG = {TG};
        constexpr float EPS = {eps_f}f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * G;

        threadgroup float sum_buf[TG];
        threadgroup float sumsq_buf[TG];

        // Phase 1: compute sum and sum-of-squares
        float s = 0.0f;
        float sq = 0.0f;
        for (uint j = tid; j < G; j += TG) {{
            float v = (float)rewards[base + j];
            s += v;
            sq += v * v;
        }}
        sum_buf[tid] = s;
        sumsq_buf[tid] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                sum_buf[tid] += sum_buf[tid + stride];
                sumsq_buf[tid] += sumsq_buf[tid + stride];
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        float mean = sum_buf[0] / (float)G;
        float var = sumsq_buf[0] / (float)G - mean * mean;
        float inv_std = metal::rsqrt(var + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: normalize
        for (uint j = tid; j < G; j += TG) {{
            float v = (float)rewards[base + j];
            out[base + j] = (T)((v - mean) * inv_std);
        }}
    """

    eps_str = str(eps_f).replace(".", "_").replace("-", "_")
    return metal_kernel(
        name=f"kk_vlsp_grpo_adv_G{G}_TG{TG}_E{eps_str}",
        input_names=["rewards"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def grpo_advantage_norm(
    rewards: Any,
    *,
    eps: float = 1e-6,
    threadgroup: int = 256,
) -> Any:
    """Fused GRPO advantage normalization.

    Computes A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
    for each group of rewards in a single Metal kernel launch.

    Args:
        rewards: Reward tensor (num_problems, G) where G is group size.
            Each row is a group of rollout rewards for one problem.
        eps: Stability epsilon for std normalization
        threadgroup: Metal threadgroup size

    Returns:
        advantages: Normalized advantages, same shape as rewards
    """
    if rewards.ndim == 1:
        rewards = rewards.reshape(1, -1)

    G = rewards.shape[-1]
    TG = _validate_tg(threadgroup)
    k = _grpo_advantage_kernel(G, TG, eps)
    rows = rewards.size // G

    return k(
        rewards,
        template=[("T", mx.float32)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[rewards.shape],
        output_dtypes=[rewards.dtype],
    )[0]


# ---------------------------------------------------------------------------
# Convenience: silu_mul_residual (for post-MLP residual in recurrent block)
# ---------------------------------------------------------------------------

@cache
def _silu_mul_residual_op() -> Any:
    """Fused silu(a) * b + c (residual connection after MLP gating)."""
    mx_mod = mx

    # This is a ternary operation: out = silu(gate) * up + residual
    # Using nary_from_expr for 3 inputs
    from ..autograd import nary_from_expr

    fwd_source = """
        uint elem = thread_position_in_grid.x;
        T g = gate[elem];
        T u = up[elem];
        T r = residual[elem];
        T s = g * kk_sigmoid(g);
        out[elem] = (T)(s * u + r);
    """

    bwd_source = """
        uint elem = thread_position_in_grid.x;
        T g = gate[elem];
        T u = up[elem];
        T ct = cotan[elem];
        T s = kk_sigmoid(g);
        T silu_g = g * s;
        // d/d(gate) = ct * u * (s + g*s*(1-s))
        dgate[elem] = (T)(ct * u * (s + g * s * ((T)1 - s)));
        // d/d(up) = ct * silu(gate)
        dup[elem] = (T)(ct * silu_g);
        // d/d(residual) = ct
        dresidual[elem] = ct;
    """

    return nary_from_expr(
        name="kk_vlsp_silu_mul_residual",
        fwd_source=fwd_source,
        bwd_source=bwd_source,
        input_names=["gate", "up", "residual"],
        output_names=["out"],
        compute_dtype=mx_mod.float32,
        header=DEFAULT_HEADER,
    )


def silu_mul_residual(gate: Any, up: Any, residual: Any) -> Any:
    """Fused silu(gate) * up + residual.

    Combines SwiGLU gating with residual addition in one kernel launch.
    Fully differentiable via custom VJP.

    Args:
        gate: Gate projection output (..., D)
        up: Up projection output (..., D)
        residual: Residual connection (..., D)

    Returns:
        out: silu(gate) * up + residual
    """
    op = _silu_mul_residual_op()
    return op(gate, up, residual)


__all__ = [
    "fused_recurrent_step",
    "depth_gate_sigmoid",
    "grpo_advantage_norm",
    "silu_mul_residual",
]
