"""Fused Metal kernel for TTT-Linear single-token decode.

Fuses all 12 steps of the decode path into a single Metal dispatch:
  1.  Z1 = XK @ W1 + b1              (inner model forward)
  2.  l2_target = XV - XK
  3-4. LayerNorm forward on Z1
  5.  L2 loss gradient
  6.  LayerNorm backward
  7.  Scale by learning rate
  8.  Gradient accumulation
  9.  Weight update: W1_bar = W1 - token_idx * W1_grad
  10. Z1_bar = XQ @ W1_bar + b1_bar   (updated forward)
  11. LayerNorm on Z1_bar
  12. output = XQ + LN(Z1_bar)         (residual)

Grid: (B * num_heads, 1, 1) -- one threadgroup per (batch, head).
Each threadgroup loads the f x f weight matrix into shared memory and
computes the entire update in-place.

Memory budget for f=64, TG=64:
  s_W1[64*64]  = 16384 bytes (the big one)
  s_xk/xq/xv/b1/b1_grad/scaled/buf [64 each] = 7*256 = 1792 bytes
  Total ~18 KB, well under 32 KB limit.
  W1_grad is read/written directly from global memory to stay under budget.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER

# Elements per thread for a given (F, TG) config
_EPT = lambda f, tg: (f + tg - 1) // tg  # noqa: E731


@cache
def _ttt_linear_decode_kernel(f: int, tg: int) -> Any:
    """Build the fused TTT-Linear decode kernel for head_dim=f."""
    F = int(f)
    TG = int(tg)
    EPT = _EPT(F, TG)

    src = f"""
        constexpr uint F = {F};
        constexpr uint TG = {TG};
        constexpr float EPS = 1e-6f;

        uint bh = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;

        uint vec_off = bh * F;
        uint w_base = bh * F * F;

        // --- Shared memory: W1 + small vectors ---
        threadgroup float s_W1[F * F];
        threadgroup float s_xk[F];
        threadgroup float s_xq[F];
        threadgroup float s_xv[F];
        threadgroup float s_b1[F];
        threadgroup float s_b1_grad[F];
        threadgroup float s_scaled[F];  // reused for scaled gradient
        threadgroup float s_buf[TG];

        // --- Load inputs to shared ---
        for (uint j = tid; j < F; j += TG) {{
            s_xk[j] = (float)xk[vec_off + j];
            s_xq[j] = (float)xq[vec_off + j];
            s_xv[j] = (float)xv[vec_off + j];
            s_b1[j] = (float)b1[vec_off + j];
            s_b1_grad[j] = (float)b1_grad[vec_off + j];
        }}
        for (uint i = tid; i < F * F; i += TG) {{
            s_W1[i] = (float)W1[w_base + i];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float lr = (float)ttt_lr[bh];
        float tok_idx = (float)token_idx[0];

        // Load ln params into registers
        float ln_w_r[{EPT}];
        float ln_b_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            ln_w_r[idx] = (float)ln_weight[vec_off + j];
            ln_b_r[idx] = (float)ln_bias[vec_off + j];
        }}

        // ================================================================
        // STEP 1: Z1 = XK @ W1 + b1
        // ================================================================
        float Z1_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            float acc = s_b1[j];
            for (uint k = 0; k < F; k++) {{
                acc += s_xk[k] * s_W1[k * F + j];
            }}
            Z1_r[idx] = acc;
        }}

        // ================================================================
        // STEP 2: l2_target = XV - XK (in registers)
        // ================================================================
        float l2t_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            l2t_r[idx] = s_xv[j] - s_xk[j];
        }}

        // ================================================================
        // STEPS 3-4: LayerNorm forward on Z1
        // ================================================================
        float local_sum = 0.0f;
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            local_sum += Z1_r[idx];
        }}
        KK_SIMD_REDUCE_SUM(s_buf, local_sum, tid, TG);
        float mu = s_buf[0] / float(F);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_var = 0.0f;
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            float d = Z1_r[idx] - mu;
            local_var += d * d;
        }}
        KK_SIMD_REDUCE_SUM(s_buf, local_var, tid, TG);
        float inv_std = metal::rsqrt(s_buf[0] / float(F) + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float Z1_hat_r[{EPT}];
        float LN_out_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            Z1_hat_r[idx] = (Z1_r[idx] - mu) * inv_std;
            LN_out_r[idx] = ln_w_r[idx] * Z1_hat_r[idx] + ln_b_r[idx];
        }}

        // ================================================================
        // STEP 5: dl_dLN = LN_out - l2_target
        // ================================================================
        float dl_dLN_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            dl_dLN_r[idx] = LN_out_r[idx] - l2t_r[idx];
        }}

        // ================================================================
        // STEP 6: LayerNorm backward
        // ================================================================
        float dl_dx_hat_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            dl_dx_hat_r[idx] = dl_dLN_r[idx] * ln_w_r[idx];
        }}

        float s1 = 0.0f, s2 = 0.0f;
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            s1 += dl_dx_hat_r[idx];
            s2 += dl_dx_hat_r[idx] * Z1_hat_r[idx];
        }}
        KK_SIMD_REDUCE_SUM(s_buf, s1, tid, TG);
        float sum_dx_hat = s_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        KK_SIMD_REDUCE_SUM(s_buf, s2, tid, TG);
        float sum_dx_hat_z = s_buf[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float dl_dZ1_r[{EPT}];
        float inv_std_F = inv_std / float(F);
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            dl_dZ1_r[idx] = (float(F) * dl_dx_hat_r[idx] - sum_dx_hat
                             - Z1_hat_r[idx] * sum_dx_hat_z) * inv_std_F;
        }}

        // ================================================================
        // STEP 7: Scale by learning rate -> store to shared for outer product
        // ================================================================
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            s_scaled[j] = lr * dl_dZ1_r[idx];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // STEPS 8+9 fused: accumulate grads AND apply weight update
        // W1_grad_new = W1_grad + xk^T @ scaled
        // W1_bar = W1 - token_idx * W1_grad_new
        // ================================================================
        for (uint i = tid; i < F * F; i += TG) {{
            uint row = i / F;
            uint col = i % F;
            float g_new = (float)W1_grad[w_base + i] + s_xk[row] * s_scaled[col];
            W1_grad_out[w_base + i] = (T)g_new;
            s_W1[i] -= tok_idx * g_new;  // W1_bar in-place
        }}
        for (uint j = tid; j < F; j += TG) {{
            float g_new = s_b1_grad[j] + s_scaled[j];
            b1_grad_out[vec_off + j] = (T)g_new;
            s_b1[j] -= tok_idx * g_new;  // b1_bar in-place
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // STEP 10: Z1_bar = XQ @ W1_bar + b1_bar
        // ================================================================
        float Z1_bar_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            float acc = s_b1[j];
            for (uint k = 0; k < F; k++) {{
                acc += s_xq[k] * s_W1[k * F + j];
            }}
            Z1_bar_r[idx] = acc;
        }}

        // ================================================================
        // STEP 11: LayerNorm on Z1_bar
        // ================================================================
        local_sum = 0.0f;
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            local_sum += Z1_bar_r[idx];
        }}
        KK_SIMD_REDUCE_SUM(s_buf, local_sum, tid, TG);
        float mu2 = s_buf[0] / float(F);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        local_var = 0.0f;
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            float d = Z1_bar_r[idx] - mu2;
            local_var += d * d;
        }}
        KK_SIMD_REDUCE_SUM(s_buf, local_var, tid, TG);
        float inv_std2 = metal::rsqrt(s_buf[0] / float(F) + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ================================================================
        // STEP 12: output = XQ + LN(Z1_bar)
        // ================================================================
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            float z_hat = (Z1_bar_r[idx] - mu2) * inv_std2;
            float ln_val = ln_w_r[idx] * z_hat + ln_b_r[idx];
            out[vec_off + j] = (T)(s_xq[j] + ln_val);
        }}

        // ================================================================
        // Write back W1/b1 state
        // At mini-batch boundary: persist W1_bar, b1_bar, zero grads
        // Otherwise: keep W1/b1 unchanged
        // ================================================================
        if (last_in_mb[0] > 0) {{
            for (uint i = tid; i < F * F; i += TG) {{
                W1_out[w_base + i] = (T)s_W1[i];     // persist W1_bar
                W1_grad_out[w_base + i] = T(0);       // zero grad
            }}
            for (uint j = tid; j < F; j += TG) {{
                b1_out[vec_off + j] = (T)s_b1[j];     // persist b1_bar
                b1_grad_out[vec_off + j] = T(0);       // zero grad
            }}
        }} else {{
            for (uint i = tid; i < F * F; i += TG) {{
                W1_out[w_base + i] = W1[w_base + i];  // unchanged
            }}
            for (uint j = tid; j < F; j += TG) {{
                b1_out[vec_off + j] = b1[vec_off + j]; // unchanged
            }}
        }}
    """

    return metal_kernel(
        name=f"ttt_linear_decode_F{F}_TG{TG}",
        input_names=[
            "xq", "xk", "xv",
            "ttt_lr", "token_idx", "last_in_mb",
            "W1", "b1", "W1_grad", "b1_grad",
            "ln_weight", "ln_bias",
        ],
        output_names=["out", "W1_out", "b1_out", "W1_grad_out", "b1_grad_out"],
        source=src,
        header=DEFAULT_HEADER,
    )


def ttt_linear_decode(
    xq: Any,
    xk: Any,
    xv: Any,
    ttt_lr: Any,
    token_idx: Any,
    last_in_mb: Any,
    W1: Any,
    b1: Any,
    W1_grad: Any,
    b1_grad: Any,
    ln_weight: Any,
    ln_bias: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Fused TTT-Linear decode step.

    Args:
        xq, xk, xv: [B*nh, F] query/key/value vectors
        ttt_lr: [B*nh] per-head learning rate
        token_idx: [1] scalar normalizer
        last_in_mb: [1] int, 1 if last token in mini-batch
        W1: [B*nh, F, F] inner model weights
        b1: [B*nh, F] inner model bias
        W1_grad: [B*nh, F, F] gradient accumulator
        b1_grad: [B*nh, F] bias gradient accumulator
        ln_weight, ln_bias: [B*nh, F] layernorm params

    Returns:
        (output, W1_new, b1_new, W1_grad_new, b1_grad_new)
    """
    B_nh = xq.shape[0]
    F = xq.shape[-1]
    TG = min(F, 64)

    kern = _ttt_linear_decode_kernel(F, TG)

    out, W1_out, b1_out, W1_grad_out, b1_grad_out = kern(
        xq, xk, xv,
        ttt_lr, token_idx, last_in_mb,
        W1, b1, W1_grad, b1_grad,
        ln_weight, ln_bias,
        template=[("T", xq.dtype)],
        output_shapes=[
            (B_nh, F),
            W1.shape,
            (B_nh, F),
            W1_grad.shape,
            (B_nh, F),
        ],
        output_dtypes=[xq.dtype] * 5,
        grid=(B_nh * TG, 1, 1),
        threadgroup=(TG, 1, 1),
    )

    return out, W1_out, b1_out, W1_grad_out, b1_grad_out


__all__ = ["ttt_linear_decode"]
