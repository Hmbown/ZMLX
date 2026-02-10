"""Tests for gather_qmm_combine and gather_qmm_combine_quantized.

Tolerance calibration (float32):
  max_abs_err ~ 1e-5 for small shapes, ~ 1e-4 for D_in=4096
  Thresholds set at ~3x worst-case.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from numerical_harness import assert_numerical_quality

from zmlx.kernels.moe import (
    gather_mmk_combine,
    gather_qmm_combine,
    gather_qmm_combine_quantized,
    gather_qmmk_combine_quantized,
)


def _ref_gather_mm_combine(act, weights, gate, indices):
    """Decomposed reference: loop over K experts, matmul, weighted sum."""
    B, K, _ = act.shape
    D_out = weights.shape[2]
    output = mx.zeros((B, D_out), dtype=act.dtype)
    for k_idx in range(K):
        a_k = act[:, k_idx, :]  # (B, D_in)
        w_k = weights[indices[:, k_idx]]  # (B, D_in, D_out)
        proj_k = mx.matmul(mx.expand_dims(a_k, axis=1), w_k).squeeze(axis=1)
        output = output + gate[:, k_idx : k_idx + 1] * proj_k
    return output


def _quantize_expert_bank(E, D_out, D_in, bits, group_size, dtype=mx.float16):
    """Create quantized expert bank tensors for gather_qmm tests."""
    w_list, s_list, b_list = [], [], []
    for _ in range(E):
        fp = mx.random.normal((D_out, D_in)).astype(dtype) * 0.02
        w, s, b = mx.quantize(fp, group_size=group_size, bits=bits)
        w_list.append(w)
        s_list.append(s)
        b_list.append(b)
    return mx.stack(w_list), mx.stack(s_list), mx.stack(b_list)


def _ref_gather_qmm_combine_quantized(act, weights, scales, biases, gate, indices, *, group_size, bits):
    """Decomposed quantized reference: loop over K gather_qmm then weighted sum."""
    B, K, _ = act.shape
    D_out = int(weights.shape[1])
    output = mx.zeros((B, D_out), dtype=act.dtype)
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
        output = output + gate[:, k_idx : k_idx + 1] * proj_k
    return output


# ---------------------------------------------------------------------------
# Reference validation
# ---------------------------------------------------------------------------


class TestGatherQmmCombineReference:
    """Validate decomposed reference against batched matmul."""

    @pytest.mark.parametrize("B", [1, 4])
    @pytest.mark.parametrize("K", [1, 2, 4])
    @pytest.mark.parametrize("D_in,D_out", [(64, 64), (128, 256)])
    def test_reference_consistency(self, B, K, D_in, D_out):
        mx.random.seed(42)
        E = 8
        act = mx.random.normal((B, K, D_in))
        weights = mx.random.normal((E, D_in, D_out))
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E

        ref = _ref_gather_mm_combine(act, weights, gate, indices)
        mx.eval(ref)

        # Verify reference is self-consistent by computing differently
        proj_all = []
        for k_idx in range(K):
            a_k = act[:, k_idx, :]
            w_k = weights[indices[:, k_idx]]
            proj_k = mx.matmul(mx.expand_dims(a_k, axis=1), w_k).squeeze(axis=1)
            proj_all.append(proj_k)
        proj_stacked = mx.stack(proj_all, axis=1)  # (B, K, D_out)
        ref2 = mx.sum(proj_stacked * mx.expand_dims(gate, axis=-1), axis=1)
        mx.eval(ref2)

        assert_numerical_quality(
            np.array(ref.tolist()), np.array(ref2.tolist()),
            max_abs_tol=1e-4, mean_abs_tol=1e-5,
            label=f"ref B={B} K={K}",
        )


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestGatherQmmCombineCorrectness:
    """Parametric forward comparison against decomposed reference."""

    @pytest.mark.parametrize("B", [1, 4, 16])
    @pytest.mark.parametrize("K", [1, 2, 4])
    @pytest.mark.parametrize("D_in,D_out", [(512, 512), (1024, 1024)])
    @pytest.mark.parametrize("E", [8, 16])
    @pytest.mark.parametrize("streaming", [True, False])
    def test_forward(self, B, K, D_in, D_out, E, streaming):
        mx.random.seed(100)
        act = mx.random.normal((B, K, D_in))
        weights = mx.random.normal((E, D_in, D_out)) * 0.01
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E

        result = gather_qmm_combine(
            act, weights, gate, indices, streaming=streaming,
        )
        ref = _ref_gather_mm_combine(act, weights, gate, indices)
        mx.eval(result, ref)

        assert_numerical_quality(
            np.array(result.tolist()), np.array(ref.tolist()),
            max_abs_tol=5e-4, mean_abs_tol=5e-5,
            label=f"fwd B={B} K={K} D_in={D_in} E={E} streaming={streaming}",
        )


# ---------------------------------------------------------------------------
# K=1 equivalence
# ---------------------------------------------------------------------------


class TestGatherQmmCombineK1Equivalence:
    """K=1 must equal simple gather+matmul."""

    @pytest.mark.parametrize("B", [1, 4, 16])
    @pytest.mark.parametrize("D_in,D_out", [(256, 256), (512, 1024)])
    def test_k1(self, B, D_in, D_out):
        mx.random.seed(55)
        E = 8
        act = mx.random.normal((B, 1, D_in))
        weights = mx.random.normal((E, D_in, D_out)) * 0.01
        gate = mx.ones((B, 1))
        indices = (mx.random.uniform(shape=(B, 1)) * E).astype(mx.uint32) % E

        result = gather_qmm_combine(act, weights, gate, indices, streaming=True)

        # Simple gather + matmul
        a_flat = act.squeeze(axis=1)
        w_gathered = weights[indices.squeeze(axis=1)]
        ref = mx.matmul(mx.expand_dims(a_flat, axis=1), w_gathered).squeeze(axis=1)
        mx.eval(result, ref)

        assert_numerical_quality(
            np.array(result.tolist()), np.array(ref.tolist()),
            max_abs_tol=1e-4, mean_abs_tol=1e-5,
            label=f"k1 B={B} D_in={D_in}",
        )


# ---------------------------------------------------------------------------
# Streaming vs non-streaming equivalence
# ---------------------------------------------------------------------------


class TestGatherQmmCombineStreamingEquivalence:
    """Streaming and non-streaming paths should produce same results."""

    @pytest.mark.parametrize("B", [1, 4, 32])
    @pytest.mark.parametrize("K", [2, 4])
    def test_streaming_vs_nonstreaming(self, B, K):
        mx.random.seed(88)
        E, D_in, D_out = 8, 256, 256
        act = mx.random.normal((B, K, D_in))
        weights = mx.random.normal((E, D_in, D_out)) * 0.01
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E

        r_stream = gather_qmm_combine(
            act, weights, gate, indices, streaming=True,
        )
        r_batch = gather_qmm_combine(
            act, weights, gate, indices, streaming=False,
        )
        mx.eval(r_stream, r_batch)

        assert_numerical_quality(
            np.array(r_stream.tolist()), np.array(r_batch.tolist()),
            max_abs_tol=1e-4, mean_abs_tol=1e-5,
            label=f"stream_eq B={B} K={K}",
        )


# ---------------------------------------------------------------------------
# Golden values
# ---------------------------------------------------------------------------


class TestGatherQmmCombineGolden:
    """Fixed-seed regression snapshots."""

    def test_golden_determinism(self):
        mx.random.seed(9999)
        E, B, K, D_in, D_out = 4, 2, 2, 32, 32
        act = mx.random.normal((B, K, D_in))
        weights = mx.random.normal((E, D_in, D_out)) * 0.01
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1)
        indices = mx.array([[0, 1], [2, 3]], dtype=mx.uint32)

        r1 = gather_qmm_combine(act, weights, gate, indices, streaming=True)
        mx.eval(r1)

        # Same inputs, same result
        r2 = gather_qmm_combine(act, weights, gate, indices, streaming=True)
        mx.eval(r2)

        np.testing.assert_array_equal(
            np.array(r1.tolist()), np.array(r2.tolist()),
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGatherQmmCombineEdgeCases:
    """Edge case handling."""

    def test_all_same_expert(self):
        """All tokens routed to the same expert."""
        mx.random.seed(11)
        E, B, K, D_in, D_out = 8, 4, 2, 64, 64
        act = mx.random.normal((B, K, D_in))
        weights = mx.random.normal((E, D_in, D_out)) * 0.01
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1)
        indices = mx.zeros((B, K), dtype=mx.uint32)  # all expert 0

        result = gather_qmm_combine(act, weights, gate, indices, streaming=True)
        ref = _ref_gather_mm_combine(act, weights, gate, indices)
        mx.eval(result, ref)

        assert_numerical_quality(
            np.array(result.tolist()), np.array(ref.tolist()),
            max_abs_tol=1e-4, mean_abs_tol=1e-5,
            label="all_same_expert",
        )

    def test_gate_one_hot(self):
        """Gate weights are one-hot (only one expert contributes)."""
        mx.random.seed(22)
        E, B, K, D_in, D_out = 8, 4, 4, 64, 64
        act = mx.random.normal((B, K, D_in))
        weights = mx.random.normal((E, D_in, D_out)) * 0.01
        gate = mx.zeros((B, K))
        gate = gate.at[:, 0].add(1.0)  # all weight on first expert
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E

        result = gather_qmm_combine(act, weights, gate, indices, streaming=True)
        ref = _ref_gather_mm_combine(act, weights, gate, indices)
        mx.eval(result, ref)

        assert_numerical_quality(
            np.array(result.tolist()), np.array(ref.tolist()),
            max_abs_tol=1e-4, mean_abs_tol=1e-5,
            label="gate_one_hot",
        )

    def test_bad_act_shape_raises(self):
        with pytest.raises(ValueError, match="act must have shape"):
            gather_qmm_combine(
                mx.zeros((4, 8)),
                mx.zeros((2, 8, 8)),
                mx.zeros((4, 2)),
                mx.zeros((4, 2), dtype=mx.uint32),
            )

    def test_bad_weights_shape_raises(self):
        with pytest.raises(ValueError, match="weights must have shape"):
            gather_qmm_combine(
                mx.zeros((4, 2, 8)),
                mx.zeros((8, 8)),  # 2D instead of 3D
                mx.zeros((4, 2)),
                mx.zeros((4, 2), dtype=mx.uint32),
            )


# ---------------------------------------------------------------------------
# Float16 correctness
# ---------------------------------------------------------------------------


class TestGatherQmmCombineFloat16:
    """Float16 dtype correctness."""

    @pytest.mark.parametrize("streaming", [True, False])
    def test_float16(self, streaming):
        mx.random.seed(33)
        E, B, K, D_in, D_out = 8, 4, 2, 128, 128
        act = mx.random.normal((B, K, D_in)).astype(mx.float16)
        weights = (mx.random.normal((E, D_in, D_out)) * 0.01).astype(mx.float16)
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1).astype(mx.float16)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E

        result = gather_qmm_combine(
            act, weights, gate, indices, streaming=streaming,
        )
        ref = _ref_gather_mm_combine(act, weights, gate, indices)
        mx.eval(result, ref)

        assert_numerical_quality(
            np.array(result.tolist()), np.array(ref.tolist()),
            max_abs_tol=5e-2, mean_abs_tol=5e-3,
            label=f"float16 streaming={streaming}",
        )


# ---------------------------------------------------------------------------
# FP32 output/accumulate mode
# ---------------------------------------------------------------------------


class TestGatherQmmCombineFP32Output:
    """Output dtype override for promotion-sensitive MoE paths."""

    @pytest.mark.parametrize("streaming", [True, False])
    def test_fp32_output_dtype(self, streaming):
        mx.random.seed(1234)
        E, B, K, D_in, D_out = 8, 4, 4, 128, 256
        act = mx.random.normal((B, K, D_in)).astype(mx.float16)
        weights = (mx.random.normal((E, D_in, D_out)) * 0.01).astype(mx.float16)
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1).astype(mx.float32)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E

        result = gather_qmm_combine(
            act,
            weights,
            gate,
            indices,
            streaming=streaming,
            no_fma=True,
            output_dtype=mx.float32,
        )
        ref = _ref_gather_mm_combine(act, weights, gate, indices).astype(mx.float32)
        mx.eval(result, ref)

        assert result.dtype == mx.float32
        assert_numerical_quality(
            np.array(result.tolist()),
            np.array(ref.tolist()),
            max_abs_tol=1e-3,
            mean_abs_tol=1e-4,
            label=f"fp32_output streaming={streaming}",
        )


# ---------------------------------------------------------------------------
# K-vectorized gather+combine paths
# ---------------------------------------------------------------------------


class TestGatherKVectorizedPaths:
    """Correctness for K-vectorized gather_mm/gather_qmm combine APIs."""

    def test_gather_mmk_combine_matches_reference(self):
        mx.random.seed(2026)
        E, B, K, D_in, D_out = 8, 4, 4, 128, 96
        act = mx.random.normal((B, K, D_in)).astype(mx.float16)
        weights = (mx.random.normal((E, D_in, D_out)) * 0.01).astype(mx.float16)
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1).astype(mx.float16)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E

        result = gather_mmk_combine(
            act,
            weights,
            gate,
            indices,
            no_fma=False,
        )
        ref = _ref_gather_mm_combine(act, weights, gate, indices)
        mx.eval(result, ref)

        assert_numerical_quality(
            np.array(result.tolist()),
            np.array(ref.tolist()),
            max_abs_tol=5e-4,
            mean_abs_tol=5e-5,
            label="gather_mmk_combine",
        )

    def test_gather_qmm_combine_vectorized_k_matches_reference(self):
        mx.random.seed(2027)
        E, B, K, D_in, D_out = 8, 4, 4, 128, 96
        act = mx.random.normal((B, K, D_in)).astype(mx.float16)
        weights = (mx.random.normal((E, D_in, D_out)) * 0.01).astype(mx.float16)
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1).astype(mx.float16)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E

        result = gather_qmm_combine(
            act,
            weights,
            gate,
            indices,
            streaming=False,
            vectorized_k=True,
        )
        ref = _ref_gather_mm_combine(act, weights, gate, indices)
        mx.eval(result, ref)

        assert_numerical_quality(
            np.array(result.tolist()),
            np.array(ref.tolist()),
            max_abs_tol=5e-4,
            mean_abs_tol=5e-5,
            label="gather_qmm_combine_vectorized_k",
        )

    @pytest.mark.skipif(
        not hasattr(mx, "gather_qmm"),
        reason="mx.gather_qmm not available in this MLX build",
    )
    def test_gather_qmmk_combine_quantized_matches_reference(self):
        mx.random.seed(2028)
        E, B, K, D_in, D_out = 8, 2, 4, 128, 96
        group_size = 32
        bits = 4
        act = mx.random.normal((B, K, D_in)).astype(mx.float16)
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1).astype(mx.float16)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E
        q_w, q_s, q_b = _quantize_expert_bank(E, D_out, D_in, bits, group_size)

        result = gather_qmmk_combine_quantized(
            act,
            q_w,
            q_s,
            q_b,
            gate,
            indices,
            group_size=group_size,
            bits=bits,
        )
        ref = _ref_gather_qmm_combine_quantized(
            act,
            q_w,
            q_s,
            q_b,
            gate,
            indices,
            group_size=group_size,
            bits=bits,
        )
        mx.eval(result, ref)

        assert_numerical_quality(
            np.array(result.tolist()),
            np.array(ref.tolist()),
            max_abs_tol=5e-3,
            mean_abs_tol=5e-4,
            label="gather_qmmk_combine_quantized",
        )

    @pytest.mark.skipif(
        not hasattr(mx, "gather_qmm"),
        reason="mx.gather_qmm not available in this MLX build",
    )
    def test_gather_qmm_combine_quantized_vectorized_k_matches_reference(self):
        mx.random.seed(2029)
        E, B, K, D_in, D_out = 8, 2, 4, 128, 96
        group_size = 32
        bits = 4
        act = mx.random.normal((B, K, D_in)).astype(mx.float16)
        gate = mx.softmax(mx.random.normal((B, K)), axis=-1).astype(mx.float16)
        indices = (mx.random.uniform(shape=(B, K)) * E).astype(mx.uint32) % E
        q_w, q_s, q_b = _quantize_expert_bank(E, D_out, D_in, bits, group_size)

        result = gather_qmm_combine_quantized(
            act,
            q_w,
            q_s,
            q_b,
            gate,
            indices,
            group_size=group_size,
            bits=bits,
            streaming=False,
            vectorized_k=True,
        )
        ref = _ref_gather_qmm_combine_quantized(
            act,
            q_w,
            q_s,
            q_b,
            gate,
            indices,
            group_size=group_size,
            bits=bits,
        )
        mx.eval(result, ref)

        assert_numerical_quality(
            np.array(result.tolist()),
            np.array(ref.tolist()),
            max_abs_tol=5e-3,
            mean_abs_tol=5e-4,
            label="gather_qmm_combine_quantized_vectorized_k",
        )
