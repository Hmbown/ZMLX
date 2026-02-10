"""Target registry: defines search spaces for kernel optimization."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .candidates import InputSpec, KernelCandidate, KernelSpec, OutputSpec, SearchSpace

# ---------------------------------------------------------------------------
# Grid computation helpers — each target defines how to map inputs to
# (grid, threadgroup) tuples for Metal kernel launch.
# ---------------------------------------------------------------------------


def _moe_combine_grid(
    inputs: Sequence[Any], D: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Grid for moe_combine: (D, B, 1)."""
    expert_outputs = inputs[0]
    B = int(expert_outputs.shape[0])
    return (D, B, 1), (min(D, 256), 1, 1)


def _elementwise_grid(
    inputs: Sequence[Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Grid for elementwise kernels: (N, 1, 1)."""
    N = int(inputs[0].size)
    return (N, 1, 1), (min(N, 256), 1, 1)


def _rmsnorm_grid(
    inputs: Sequence[Any], D: int, TG: int = 256
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Grid for rmsnorm: one threadgroup per row."""
    inp = inputs[0]
    B = int(inp.size) // D
    return (B * TG, 1, 1), (TG, 1, 1)


def _topk_gating_simd_grid(
    inputs: Sequence[Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Grid for SIMD topk gating: one SIMD group (32 threads) per row."""
    inp = inputs[0]
    D = int(inp.shape[-1])
    B = int(inp.size) // D
    return (B * 32, 1, 1), (32, 1, 1)


def _hcsa_active_row_grid(
    inputs: Sequence[Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Grid for hcsa_active_row: one thread per feature dim."""
    values = inputs[1]
    D = int(values.shape[-1])
    return (D, 1, 1), (min(D, 256), 1, 1)


def _hcsa_permute_window_grid(
    inputs: Sequence[Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Grid for hcsa_permute_window: one thread per (token, feature)."""
    inp = inputs[0]
    T = int(inp.shape[0])
    D = int(inp.shape[1])
    return (D, T, 1), (min(D, 256), 1, 1)


# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------


def moe_combine_target(D: int = 4096, K: int = 2) -> SearchSpace:
    """MoE combine kernel: weighted sum of expert outputs."""
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

    ref_python = """
import mlx.core as mx
def reference(*inputs):
    expert_outputs, weights = inputs
    return mx.sum(expert_outputs * weights[..., None], axis=-2)
"""

    seed_spec = KernelSpec(
        name=f"kk_moe_combine_D{D}_K{K}",
        input_names=("expert_outputs", "weights"),
        output_names=("out",),
        source=source,
        threadgroup=(min(D, 256), 1, 1),
        template_params=(("T", "float32"),),
    )
    seed = KernelCandidate(spec=seed_spec, generation=0, llm_reasoning="baseline")

    concrete_shapes = [(B, K, D) for B in (1, 4, 16)]

    _D = D

    def _grid(inputs: Sequence[Any]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return _moe_combine_grid(inputs, _D)

    return SearchSpace(
        name="moe_combine",
        description=(
            f"Combine expert outputs: weighted sum over K={K} experts, D={D} hidden dim. "
            "expert_outputs shape (B, K, D), weights shape (B, K), output (B, D). "
            "Kernel is launched with grid=(D, B, 1)."
        ),
        reference_source=source,
        reference_python=ref_python,
        input_specs=(
            InputSpec(
                name="expert_outputs",
                shape_expr="(B, K, D)",
                dtype="float32",
                concrete_shapes=concrete_shapes,
            ),
            InputSpec(
                name="weights",
                shape_expr="(B, K)",
                dtype="float32",
                concrete_shapes=[(B, K) for B in (1, 4, 16)],
            ),
        ),
        output_specs=(
            OutputSpec(
                name="out",
                shape_expr="(B, D)",
                dtype="float32",
                concrete_shapes=[(B, D) for B in (1, 4, 16)],
            ),
        ),
        constraints=(
            "Must accumulate in float32 for numerical stability.",
            "Grid: (D, B, 1), one thread per output element.",
            f"K={K} experts, D={D} hidden dimension.",
        ),
        seed_candidates=(seed,),
        input_names=("expert_outputs", "weights"),
        output_names=("out",),
        grid_fn=f"grid=(D, B, 1), threadgroup=(min(D, 256), 1, 1) where D={D}",
        compute_grid=_grid,
        template_params=(("T", "float32"),),
    )


def fused_swiglu_target(D: int = 4096) -> SearchSpace:
    """Fused SwiGLU: silu(gate) * up."""
    source = f"""
        constexpr uint N = {D};
        uint idx = thread_position_in_grid.x;
        if (idx >= N) return;
        float g = (float)gate[idx];
        float u = (float)up[idx];
        float sig = 1.0f / (1.0f + metal::exp(-g));
        out[idx] = (T)(g * sig * u);
    """

    ref_python = """
import mlx.core as mx
def reference(*inputs):
    gate, up = inputs
    return mx.sigmoid(gate) * gate * up
"""

    seed_spec = KernelSpec(
        name="kk_fused_swiglu",
        input_names=("gate", "up"),
        output_names=("out",),
        source=source,
        threadgroup=(256, 1, 1),
        template_params=(("T", "float32"),),
    )
    seed = KernelCandidate(spec=seed_spec, generation=0, llm_reasoning="baseline")

    concrete_shapes = [(D,), (4, D), (16, D)]

    return SearchSpace(
        name="fused_swiglu",
        description=(
            f"Fused SwiGLU activation: silu(gate) * up, elementwise over D={D}. "
            "gate shape (..., D), up shape (..., D), output (..., D)."
        ),
        reference_source=source,
        reference_python=ref_python,
        input_specs=(
            InputSpec(name="gate", shape_expr="(..., D)", dtype="float32",
                      concrete_shapes=concrete_shapes),
            InputSpec(name="up", shape_expr="(..., D)", dtype="float32",
                      concrete_shapes=concrete_shapes),
        ),
        output_specs=(
            OutputSpec(name="out", shape_expr="(..., D)", dtype="float32",
                       concrete_shapes=concrete_shapes),
        ),
        constraints=(
            f"constexpr uint N = {D} is defined — total number of elements.",
            "Elementwise: grid=(N, 1, 1), one thread per element in baseline.",
            "For vectorized variants (float4, etc), reduce grid and use N for bounds.",
            "Numerically stable sigmoid via abs trick.",
        ),
        seed_candidates=(seed,),
        input_names=("gate", "up"),
        output_names=("out",),
        grid_fn="grid=(N, 1, 1) where N = total elements",
        compute_grid=_elementwise_grid,
        template_params=(("T", "float32"),),
    )


def rmsnorm_target(D: int = 4096) -> SearchSpace:
    """RMSNorm kernel: x * rsqrt(mean(x^2) + eps) * weight."""
    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = 256;
        constexpr float EPS = 1e-6f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float buf[TG];

        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            sumsq += v * v;
        }}
        KK_SIMD_REDUCE_SUM(buf, sumsq, tid, TG);

        float inv = metal::rsqrt(buf[0] / (float)D + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = tid; j < D; j += TG) {{
            float v = (float)inp[base + j];
            float w = (float)weight[j];
            out[base + j] = (T)(v * inv * w);
        }}
    """

    ref_python = f"""
import mlx.core as mx
def reference(*inputs):
    x, weight = inputs
    D = {D}
    ms = mx.mean(x * x, axis=-1, keepdims=True)
    return x * mx.rsqrt(ms + 1e-6) * weight
"""

    seed_spec = KernelSpec(
        name=f"kk_rmsnorm_D{D}",
        input_names=("inp", "weight"),
        output_names=("out",),
        source=source,
        threadgroup=(256, 1, 1),
        template_params=(("T", "float32"),),
    )
    seed = KernelCandidate(spec=seed_spec, generation=0, llm_reasoning="baseline")

    _D = D

    def _grid(inputs: Sequence[Any]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return _rmsnorm_grid(inputs, _D)

    return SearchSpace(
        name="rmsnorm",
        description=(
            f"RMSNorm: x * rsqrt(mean(x^2) + eps) * weight, D={D}. "
            "inp shape (B, D), weight shape (D,), output (B, D). "
            "Grid: (B * TG, 1, 1), one threadgroup per row."
        ),
        reference_source=source,
        reference_python=ref_python,
        input_specs=(
            InputSpec(name="inp", shape_expr="(B, D)", dtype="float32",
                      concrete_shapes=[(B, D) for B in (1, 4, 16)]),
            InputSpec(name="weight", shape_expr="(D,)", dtype="float32",
                      concrete_shapes=[(D,)]),
        ),
        output_specs=(
            OutputSpec(name="out", shape_expr="(B, D)", dtype="float32",
                       concrete_shapes=[(B, D) for B in (1, 4, 16)]),
        ),
        constraints=(
            "Must use threadgroup reduction for sum-of-squares.",
            "EPS = 1e-6.",
            f"D={D} dimension, TG=256 threadgroup size.",
        ),
        seed_candidates=(seed,),
        input_names=("inp", "weight"),
        output_names=("out",),
        header="",  # Will use DEFAULT_HEADER
        grid_fn=f"grid=(B * 256, 1, 1), threadgroup=(256, 1, 1) where D={D}",
        compute_grid=_grid,
        template_params=(("T", "float32"),),
    )


def topk_gating_target(D: int = 8, K: int = 2) -> SearchSpace:
    """Top-K gating with softmax for MoE routing."""
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

    ref_python = f"""
import mlx.core as mx
def reference(*inputs):
    x = inputs[0]
    K = {K}
    sorted_indices = mx.argpartition(-x, kth=K - 1, axis=-1)
    indices = sorted_indices[..., :K]
    values = mx.take_along_axis(x, indices, axis=-1)
    weights = mx.softmax(values, axis=-1)
    return [weights, indices.astype(mx.uint32)]
"""

    seed_spec = KernelSpec(
        name=f"kk_topk_gating_simd_D{D}_K{K}",
        input_names=("inp",),
        output_names=("weights", "indices"),
        source=source,
        threadgroup=(32, 1, 1),
        template_params=(("T", "float32"),),
    )
    seed = KernelCandidate(spec=seed_spec, generation=0, llm_reasoning="baseline")

    return SearchSpace(
        name="topk_gating",
        description=(
            f"Top-{K} gating with softmax for MoE routing, D={D}. "
            "inp shape (B, D), output weights (B, K) and indices (B, K). "
            "SIMD-based (D <= 32): one SIMD group per row."
        ),
        reference_source=source,
        reference_python=ref_python,
        input_specs=(
            InputSpec(name="inp", shape_expr="(B, D)", dtype="float32",
                      concrete_shapes=[(B, D) for B in (1, 4, 16)]),
        ),
        output_specs=(
            OutputSpec(name="weights", shape_expr="(B, K)", dtype="float32",
                       concrete_shapes=[(B, K) for B in (1, 4, 16)]),
            OutputSpec(name="indices", shape_expr="(B, K)", dtype="uint32",
                       concrete_shapes=[(B, K) for B in (1, 4, 16)]),
        ),
        constraints=(
            f"D={D} must be <= 32 for SIMD path.",
            "Uses simd_max for top-k selection.",
            "Softmax is computed over selected top-k values only.",
        ),
        seed_candidates=(seed,),
        input_names=("inp",),
        output_names=("weights", "indices"),
        grid_fn=f"grid=(B * 32, 1, 1), threadgroup=(32, 1, 1) where D={D}",
        compute_grid=_topk_gating_simd_grid,
        template_params=(("T", "float32"),),
    )


def hcsa_active_row_target(D: int = 128, W: int = 65) -> SearchSpace:
    """HCSA active-row microkernel: softmax(scores) @ values for one query row."""
    source = f"""
        constexpr uint D = {D};
        constexpr uint W = {W};
        uint d_idx = thread_position_in_grid.x;
        if (d_idx >= D) return;

        float max_s = -INFINITY;
        for (uint j = 0; j < W; ++j) {{
            float s = (float)scores[j];
            max_s = (s > max_s) ? s : max_s;
        }}

        float denom = 0.0f;
        float acc = 0.0f;
        for (uint j = 0; j < W; ++j) {{
            float w = metal::exp((float)scores[j] - max_s);
            denom += w;
            acc += w * (float)values[j * D + d_idx];
        }}
        out[d_idx] = (T)(acc / denom);
    """

    ref_python = """
import mlx.core as mx
def reference(*inputs):
    scores, values = inputs
    w = mx.softmax(scores, axis=-1)
    return mx.sum(w[:, None] * values, axis=0)
"""

    seed_spec = KernelSpec(
        name=f"kk_hcsa_active_row_D{D}_W{W}",
        input_names=("scores", "values"),
        output_names=("out",),
        source=source,
        threadgroup=(min(D, 256), 1, 1),
        template_params=(("T", "float32"),),
    )
    seed = KernelCandidate(spec=seed_spec, generation=0, llm_reasoning="baseline")

    return SearchSpace(
        name="hcsa_active_row",
        description=(
            f"HCSA active-row primitive: softmax(scores[W={W}]) @ values[W, D={D}] -> out[D]."
        ),
        reference_source=source,
        reference_python=ref_python,
        input_specs=(
            InputSpec(
                name="scores",
                shape_expr="(W,)",
                dtype="float32",
                concrete_shapes=[(W,)],
            ),
            InputSpec(
                name="values",
                shape_expr="(W, D)",
                dtype="float32",
                concrete_shapes=[(W, D)],
            ),
        ),
        output_specs=(
            OutputSpec(
                name="out",
                shape_expr="(D,)",
                dtype="float32",
                concrete_shapes=[(D,)],
            ),
        ),
        constraints=(
            "Simple scalar baseline intended for correctness-first search bootstrapping.",
            f"W={W}, D={D}.",
        ),
        seed_candidates=(seed,),
        input_names=("scores", "values"),
        output_names=("out",),
        grid_fn="grid=(D, 1, 1), threadgroup=(min(D, 256), 1, 1)",
        compute_grid=_hcsa_active_row_grid,
        template_params=(("T", "float32"),),
    )


def hcsa_permute_window_target(T: int = 256, D: int = 128, W: int = 65) -> SearchSpace:
    """HCSA permute-window microkernel: gather+mean over per-token neighbor windows."""
    source = f"""
        constexpr uint SEQ = {T};
        constexpr uint D = {D};
        constexpr uint W = {W};

        uint t_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;
        if (t_idx >= SEQ || d_idx >= D) return;

        float acc = 0.0f;
        for (uint j = 0; j < W; ++j) {{
            uint src = (uint)neighbor_idx[t_idx * W + j];
            if (src >= SEQ) src = 0;
            acc += (float)inp[src * D + d_idx];
        }}
        out[t_idx * D + d_idx] = (T)(acc / (float)W);
    """

    ref_python = f"""
import mlx.core as mx
def reference(*inputs):
    inp, neighbor_idx = inputs
    T = {T}
    safe_idx = mx.where(neighbor_idx < T, neighbor_idx, mx.zeros_like(neighbor_idx))
    gathered = inp[safe_idx.astype(mx.int32)]
    return mx.mean(gathered, axis=1)
"""

    seed_spec = KernelSpec(
        name=f"kk_hcsa_permute_window_T{T}_D{D}_W{W}",
        input_names=("inp", "neighbor_idx"),
        output_names=("out",),
        source=source,
        threadgroup=(min(D, 256), 1, 1),
        template_params=(("T", "float32"),),
    )
    seed = KernelCandidate(spec=seed_spec, generation=0, llm_reasoning="baseline")

    return SearchSpace(
        name="hcsa_permute_window",
        description=(
            f"HCSA permute-window primitive: mean over gathered neighbors, "
            f"inp[T={T}, D={D}], neighbor_idx[T, W={W}] -> out[T, D]."
        ),
        reference_source=source,
        reference_python=ref_python,
        input_specs=(
            InputSpec(
                name="inp",
                shape_expr="(T, D)",
                dtype="float32",
                concrete_shapes=[(T, D)],
            ),
            InputSpec(
                name="neighbor_idx",
                shape_expr="(T, W)",
                dtype="uint32",
                concrete_shapes=[(T, W)],
            ),
        ),
        output_specs=(
            OutputSpec(
                name="out",
                shape_expr="(T, D)",
                dtype="float32",
                concrete_shapes=[(T, D)],
            ),
        ),
        constraints=(
            "Neighbor indices are clamped to valid range [0, T).",
            "Baseline computes simple mean over window elements.",
            f"T={T}, W={W}, D={D}.",
        ),
        seed_candidates=(seed,),
        input_names=("inp", "neighbor_idx"),
        output_names=("out",),
        grid_fn="grid=(D, T, 1), threadgroup=(min(D, 256), 1, 1)",
        compute_grid=_hcsa_permute_window_grid,
        template_params=(("T", "float32"),),
    )


def _ttt_decode_grid(
    inputs: Sequence[Any],
    _F: int = 64,
    TG: int = 64,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Grid for ttt_linear_decode: one threadgroup per (batch, head)."""
    B_nh = int(inputs[0].shape[0])
    return (B_nh * TG, 1, 1), (TG, 1, 1)


def ttt_linear_decode_target(F: int = 64) -> SearchSpace:
    """TTT-Linear decode: fused forward + LN + L2 grad + LN bwd + update + output.

    This is the hot path for TTT inference: a single token decode step that
    updates the inner model weights and produces the output in one dispatch.

    The kernel operates per (batch, head) with one threadgroup each, loading
    the F x F weight matrix into shared memory.
    """
    TG = min(F, 64)
    EPT = (F + TG - 1) // TG

    source = f"""
        constexpr uint F = {F};
        constexpr uint TG = {TG};
        constexpr float EPS = 1e-6f;

        uint bh = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;

        uint vec_off = bh * F;
        uint w_base = bh * F * F;

        threadgroup float s_W1[F * F];
        threadgroup float s_xk[F];
        threadgroup float s_xq[F];
        threadgroup float s_xv[F];
        threadgroup float s_b1[F];
        threadgroup float s_b1_grad[F];
        threadgroup float s_scaled[F];
        threadgroup float s_buf[TG];

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

        float ln_w_r[{EPT}];
        float ln_b_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            ln_w_r[idx] = (float)ln_weight[vec_off + j];
            ln_b_r[idx] = (float)ln_bias[vec_off + j];
        }}

        // STEP 1: Z1 = XK @ W1 + b1
        float Z1_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            float acc = s_b1[j];
            for (uint k = 0; k < F; k++) {{
                acc += s_xk[k] * s_W1[k * F + j];
            }}
            Z1_r[idx] = acc;
        }}

        // STEP 2: l2_target = XV - XK
        float l2t_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            l2t_r[idx] = s_xv[j] - s_xk[j];
        }}

        // STEPS 3-4: LayerNorm forward
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

        // STEP 5: dl_dLN = LN_out - l2_target
        float dl_dLN_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            dl_dLN_r[idx] = LN_out_r[idx] - l2t_r[idx];
        }}

        // STEP 6: LayerNorm backward
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

        // STEP 7: Scale by learning rate
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            s_scaled[j] = lr * dl_dZ1_r[idx];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // STEPS 8+9: Grad accumulation + weight update
        for (uint i = tid; i < F * F; i += TG) {{
            uint row = i / F;
            uint col = i % F;
            float g_new = (float)W1_grad[w_base + i] + s_xk[row] * s_scaled[col];
            W1_grad_out[w_base + i] = (T)g_new;
            s_W1[i] -= tok_idx * g_new;
        }}
        for (uint j = tid; j < F; j += TG) {{
            float g_new = s_b1_grad[j] + s_scaled[j];
            b1_grad_out[vec_off + j] = (T)g_new;
            s_b1[j] -= tok_idx * g_new;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // STEP 10: Z1_bar = XQ @ W1_bar + b1_bar
        float Z1_bar_r[{EPT}];
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            float acc = s_b1[j];
            for (uint k = 0; k < F; k++) {{
                acc += s_xq[k] * s_W1[k * F + j];
            }}
            Z1_bar_r[idx] = acc;
        }}

        // STEP 11: LayerNorm on Z1_bar
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

        // STEP 12: output = XQ + LN(Z1_bar)
        for (uint j = tid, idx = 0; j < F; j += TG, idx++) {{
            float z_hat = (Z1_bar_r[idx] - mu2) * inv_std2;
            float ln_val = ln_w_r[idx] * z_hat + ln_b_r[idx];
            out[vec_off + j] = (T)(s_xq[j] + ln_val);
        }}

        // Write back W1/b1 state
        if (last_in_mb[0] > 0) {{
            for (uint i = tid; i < F * F; i += TG) {{
                W1_out[w_base + i] = (T)s_W1[i];
                W1_grad_out[w_base + i] = T(0);
            }}
            for (uint j = tid; j < F; j += TG) {{
                b1_out[vec_off + j] = (T)s_b1[j];
                b1_grad_out[vec_off + j] = T(0);
            }}
        }} else {{
            for (uint i = tid; i < F * F; i += TG) {{
                W1_out[w_base + i] = W1[w_base + i];
            }}
            for (uint j = tid; j < F; j += TG) {{
                b1_out[vec_off + j] = b1[vec_off + j];
            }}
        }}
    """

    ref_python = f"""
import mlx.core as mx

def reference(*inputs):
    xq, xk, xv, ttt_lr, token_idx_arr, last_in_mb, W1, b1, W1_grad, b1_grad, ln_weight, ln_bias = inputs
    F = {F}
    B_nh = xq.shape[0]

    # Reshape for matmuls
    xk3 = xk[:, None, :]
    xq3 = xq[:, None, :]
    xv3 = xv[:, None, :]
    b13 = b1[:, None, :]
    lnw3 = ln_weight[:, None, :]
    lnb3 = ln_bias[:, None, :]

    # Step 1: Z1 = XK @ W1 + b1
    Z1 = xk3 @ W1 + b13

    # Step 2: l2_target
    l2t = xv3 - xk3

    # Steps 3-4: LN forward
    mu = mx.mean(Z1, axis=-1, keepdims=True)
    var = mx.var(Z1, axis=-1, keepdims=True)
    std = mx.sqrt(var + 1e-6)
    Z1_hat = (Z1 - mu) / std
    LN_out = lnw3 * Z1_hat + lnb3

    # Step 5: loss grad
    dl_dLN = LN_out - l2t

    # Step 6: LN backward
    dl_dx_hat = dl_dLN * lnw3
    s1 = mx.sum(dl_dx_hat, axis=-1, keepdims=True)
    s2 = mx.sum(dl_dx_hat * Z1_hat, axis=-1, keepdims=True)
    dl_dZ1 = (F * dl_dx_hat - s1 - Z1_hat * s2) / (std * F)

    # Step 7: scale
    lr3 = ttt_lr[:, None, None]
    scaled = lr3 * dl_dZ1

    # Step 8: grad accumulation
    W1_grad_new = W1_grad + mx.transpose(xk3, (0, 2, 1)) @ scaled
    b1_grad_new = b1_grad + scaled.squeeze(1)

    # Step 9: weight update
    ti = token_idx_arr[0]
    W1_bar = W1 - ti * W1_grad_new
    b1_bar = b1 - ti * b1_grad_new

    # Step 10: forward with updated weights
    b1_bar3 = b1_bar[:, None, :]
    Z1_bar = xq3 @ W1_bar + b1_bar3

    # Step 11: LN on output
    mu2 = mx.mean(Z1_bar, axis=-1, keepdims=True)
    var2 = mx.var(Z1_bar, axis=-1, keepdims=True)
    std2 = mx.sqrt(var2 + 1e-6)
    Z1_bar_hat = (Z1_bar - mu2) / std2
    LN_bar = lnw3 * Z1_bar_hat + lnb3

    # Step 12: residual
    output = (xq3 + LN_bar).squeeze(1)

    # For correctness we only check the output (first return value)
    return output
"""

    input_names = (
        "xq", "xk", "xv",
        "ttt_lr", "token_idx", "last_in_mb",
        "W1", "b1", "W1_grad", "b1_grad",
        "ln_weight", "ln_bias",
    )
    output_names = ("out", "W1_out", "b1_out", "W1_grad_out", "b1_grad_out")

    seed_spec = KernelSpec(
        name=f"ttt_linear_decode_F{F}_TG{TG}",
        input_names=input_names,
        output_names=output_names,
        source=source,
        threadgroup=(TG, 1, 1),
        template_params=(("T", "float32"),),
    )
    seed = KernelCandidate(spec=seed_spec, generation=0, llm_reasoning="baseline fused 12-step")

    B_vals = (2, 8, 32)

    _F = F
    _TG = TG

    def _grid(inputs: Sequence[Any]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return _ttt_decode_grid(inputs, _F, _TG)

    return SearchSpace(
        name="ttt_linear_decode",
        description=(
            f"TTT-Linear single-token decode: fused forward + LayerNorm + L2 loss gradient "
            f"+ LN backward + gradient accumulation + weight update + second forward + LN + "
            f"residual. F={F} head_dim, one threadgroup per (batch, head). "
            f"The F x F weight matrix (W1) lives in threadgroup shared memory. "
            f"W1_grad is in global memory to stay under 32KB threadgroup limit. "
            "Inputs: xq/xk/xv [B_nh, F], ttt_lr [B_nh], token_idx [1], last_in_mb [1], "
            "W1 [B_nh, F, F], b1 [B_nh, F], W1_grad [B_nh, F, F], b1_grad [B_nh, F], "
            "ln_weight/ln_bias [B_nh, F]. "
            "Outputs: out [B_nh, F], W1_out [B_nh, F, F], b1_out [B_nh, F], "
            "W1_grad_out [B_nh, F, F], b1_grad_out [B_nh, F]. "
            "Key optimization targets: the two vec-mat multiplies (steps 1, 10) and "
            "the outer product (step 8) dominate. SIMD and vectorized loads can help."
        ),
        reference_source=source,
        reference_python=ref_python,
        input_specs=(
            InputSpec(name="xq", shape_expr="(B_nh, F)", dtype="float32",
                      concrete_shapes=[(B, F) for B in B_vals]),
            InputSpec(name="xk", shape_expr="(B_nh, F)", dtype="float32",
                      concrete_shapes=[(B, F) for B in B_vals]),
            InputSpec(name="xv", shape_expr="(B_nh, F)", dtype="float32",
                      concrete_shapes=[(B, F) for B in B_vals]),
            InputSpec(name="ttt_lr", shape_expr="(B_nh,)", dtype="float32",
                      concrete_shapes=[(B,) for B in B_vals]),
            InputSpec(name="token_idx", shape_expr="(1,)", dtype="float32",
                      concrete_shapes=[(1,)]),
            InputSpec(name="last_in_mb", shape_expr="(1,)", dtype="int32",
                      concrete_shapes=[(1,)]),
            InputSpec(name="W1", shape_expr="(B_nh, F, F)", dtype="float32",
                      concrete_shapes=[(B, F, F) for B in B_vals]),
            InputSpec(name="b1", shape_expr="(B_nh, F)", dtype="float32",
                      concrete_shapes=[(B, F) for B in B_vals]),
            InputSpec(name="W1_grad", shape_expr="(B_nh, F, F)", dtype="float32",
                      concrete_shapes=[(B, F, F) for B in B_vals]),
            InputSpec(name="b1_grad", shape_expr="(B_nh, F)", dtype="float32",
                      concrete_shapes=[(B, F) for B in B_vals]),
            InputSpec(name="ln_weight", shape_expr="(B_nh, F)", dtype="float32",
                      concrete_shapes=[(B, F) for B in B_vals]),
            InputSpec(name="ln_bias", shape_expr="(B_nh, F)", dtype="float32",
                      concrete_shapes=[(B, F) for B in B_vals]),
        ),
        output_specs=(
            OutputSpec(name="out", shape_expr="(B_nh, F)", dtype="float32",
                       concrete_shapes=[(B, F) for B in B_vals]),
            OutputSpec(name="W1_out", shape_expr="(B_nh, F, F)", dtype="float32",
                       concrete_shapes=[(B, F, F) for B in B_vals]),
            OutputSpec(name="b1_out", shape_expr="(B_nh, F)", dtype="float32",
                       concrete_shapes=[(B, F) for B in B_vals]),
            OutputSpec(name="W1_grad_out", shape_expr="(B_nh, F, F)", dtype="float32",
                       concrete_shapes=[(B, F, F) for B in B_vals]),
            OutputSpec(name="b1_grad_out", shape_expr="(B_nh, F)", dtype="float32",
                       concrete_shapes=[(B, F) for B in B_vals]),
        ),
        constraints=(
            f"F={F} head dimension, TG={TG} threadgroup size.",
            f"Threadgroup memory budget: s_W1[{F}*{F}] = {F*F*4} bytes is the dominant allocation. "
            f"Total threadgroup memory MUST stay under 32768 bytes.",
            "W1_grad is in global memory (not shared) to respect the 32KB limit.",
            "All computation in float32 for numerical stability.",
            "The two vector-matrix multiplies (steps 1 and 10) each iterate F times per output "
            "element — these are the main compute bottleneck.",
            "The outer product in step 8 iterates F*F elements. Consider vectorized loads "
            "(float4) or loop unrolling for the inner loops.",
            "SIMD reductions use the KK_SIMD_REDUCE_SUM macro from the header.",
            "The last_in_mb flag controls whether W1/b1 state is persisted or kept unchanged.",
        ),
        seed_candidates=(seed,),
        input_names=input_names,
        output_names=output_names,
        grid_fn=f"grid=(B_nh * {TG}, 1, 1), threadgroup=({TG}, 1, 1)",
        compute_grid=_grid,
        template_params=(("T", "float32"),),
    )


def glm_moe_combine_target() -> SearchSpace:
    """MoE combine for GLM-4.7-Flash: D=2048, K=4 experts."""
    return moe_combine_target(D=2048, K=4)


def glm_fused_swiglu_target() -> SearchSpace:
    """Fused SwiGLU for GLM-4.7-Flash MoE experts: D=1536."""
    return fused_swiglu_target(D=1536)


def glm_rmsnorm_target() -> SearchSpace:
    """RMSNorm for GLM-4.7-Flash: D=2048."""
    return rmsnorm_target(D=2048)


def glm_topk_gating_target() -> SearchSpace:
    """Top-4 gating for GLM-4.7-Flash: 64 experts, top 4."""
    return topk_gating_target(D=64, K=4)


TARGETS: dict[str, Callable[..., SearchSpace]] = {
    "moe_combine": moe_combine_target,
    "fused_swiglu": fused_swiglu_target,
    "rmsnorm": rmsnorm_target,
    "topk_gating": topk_gating_target,
    "hcsa_active_row": hcsa_active_row_target,
    "hcsa_permute_window": hcsa_permute_window_target,
    "ttt_linear_decode": ttt_linear_decode_target,
    "glm_moe_combine": glm_moe_combine_target,
    "glm_fused_swiglu": glm_fused_swiglu_target,
    "glm_rmsnorm": glm_rmsnorm_target,
    "glm_topk_gating": glm_topk_gating_target,
}
