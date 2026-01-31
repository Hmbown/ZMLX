"""Correctness tests for mx.gather_qmm_swiglu (fused gather+QMM+SwiGLU).

Validates the fused op against the naive two-pass approach:
    gate_out = mx.gather_qmm(x, gate_w, gate_scales, gate_biases, ...)
    up_out   = mx.gather_qmm(x, up_w, up_scales, up_biases, ...)
    expected = silu(gate_out) * up_out
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# Skip entire module if the fused op is not available
pytestmark = pytest.mark.skipif(
    not hasattr(mx, "gather_qmm_swiglu"),
    reason="mx.gather_qmm_swiglu not available in this MLX build",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quantize_experts(
    n_experts: int,
    N: int,
    K: int,
    bits: int,
    group_size: int,
    dtype: mx.Dtype = mx.float16,
):
    """Create properly quantized expert weight matrices.

    Returns (w, scales, biases) each with shape (n_experts, ...).
    """
    w_list, s_list, b_list = [], [], []
    for _ in range(n_experts):
        fp = mx.random.normal((N, K)).astype(dtype) * 0.02
        w, s, b = mx.quantize(fp, group_size=group_size, bits=bits)
        w_list.append(w)
        s_list.append(s)
        b_list.append(b)
    return mx.stack(w_list), mx.stack(s_list), mx.stack(b_list)


def _reference_swiglu(
    x: mx.array,
    gate_w: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    up_w: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    lhs_indices: mx.array | None,
    rhs_indices: mx.array | None,
    group_size: int,
    bits: int,
) -> mx.array:
    """Two-pass reference: gather_qmm(gate) + gather_qmm(up) + SwiGLU."""
    gate_out = mx.gather_qmm(
        x, gate_w, gate_scales, gate_biases,
        lhs_indices=lhs_indices, rhs_indices=rhs_indices,
        transpose=True, group_size=group_size, bits=bits,
    )
    up_out = mx.gather_qmm(
        x, up_w, up_scales, up_biases,
        lhs_indices=lhs_indices, rhs_indices=rhs_indices,
        transpose=True, group_size=group_size, bits=bits,
    )
    return nn.silu(gate_out) * up_out


def _run_and_compare(
    n_experts: int,
    M: int,
    K: int,
    N: int,
    bits: int,
    group_size: int,
    dtype: mx.Dtype,
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    """Run fused op vs reference and assert closeness."""
    gate_w, gate_s, gate_b = _quantize_experts(n_experts, N, K, bits, group_size, dtype)
    up_w, up_s, up_b = _quantize_experts(n_experts, N, K, bits, group_size, dtype)

    x = mx.random.normal((1, M, K)).astype(dtype) * 0.1

    # Select a subset of experts
    n_sel = min(2, n_experts)
    lhs_indices = mx.zeros((n_sel,), dtype=mx.uint32)
    rhs_indices = mx.arange(n_sel).astype(mx.uint32)

    result = mx.gather_qmm_swiglu(
        x, gate_w, gate_s, gate_b,
        up_w, up_s, up_b,
        lhs_indices=lhs_indices, rhs_indices=rhs_indices,
        transpose=True, group_size=group_size, bits=bits,
    )
    expected = _reference_swiglu(
        x, gate_w, gate_s, gate_b,
        up_w, up_s, up_b,
        lhs_indices=lhs_indices, rhs_indices=rhs_indices,
        group_size=group_size, bits=bits,
    )
    mx.eval(result, expected)

    a = np.array(result.tolist(), dtype=np.float32)
    b = np.array(expected.tolist(), dtype=np.float32)
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasic:
    """Basic smoke tests."""

    def test_minimal(self):
        """Single token, 2 experts, 4-bit, K=512, N=512."""
        _run_and_compare(
            n_experts=2, M=1, K=512, N=512,
            bits=4, group_size=64, dtype=mx.float16,
        )

    def test_output_shape(self):
        """Verify output shape matches expected (n_sel, M, N)."""
        K, N, n_experts = 512, 1024, 4
        gate_w, gate_s, gate_b = _quantize_experts(n_experts, N, K, 4, 64)
        up_w, up_s, up_b = _quantize_experts(n_experts, N, K, 4, 64)
        x = mx.random.normal((1, 1, K)).astype(mx.float16)
        lhs_indices = mx.zeros((2,), dtype=mx.uint32)
        rhs_indices = mx.array([0, 2], dtype=mx.uint32)

        result = mx.gather_qmm_swiglu(
            x, gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            lhs_indices=lhs_indices, rhs_indices=rhs_indices,
            transpose=True, group_size=64, bits=4,
        )
        mx.eval(result)
        assert result.shape == (2, 1, N), f"Expected (2, 1, {N}), got {result.shape}"

    def test_output_dtype_f16(self):
        """Output dtype matches input dtype (float16)."""
        K, N = 512, 512
        gate_w, gate_s, gate_b = _quantize_experts(2, N, K, 4, 64, mx.float16)
        up_w, up_s, up_b = _quantize_experts(2, N, K, 4, 64, mx.float16)
        x = mx.random.normal((1, 1, K)).astype(mx.float16)
        result = mx.gather_qmm_swiglu(
            x, gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            lhs_indices=mx.array([0]), rhs_indices=mx.array([0]),
            transpose=True, group_size=64, bits=4,
        )
        mx.eval(result)
        assert result.dtype == mx.float16


class TestVariedShapes:
    """Test across different K and N dimensions."""

    @pytest.mark.parametrize("K", [512, 1024, 2048, 4096])
    def test_varied_K(self, K):
        _run_and_compare(
            n_experts=2, M=1, K=K, N=1024,
            bits=4, group_size=64, dtype=mx.float16,
        )

    @pytest.mark.parametrize("N", [512, 1024, 2048])
    def test_varied_N(self, N):
        _run_and_compare(
            n_experts=2, M=1, K=1024, N=N,
            bits=4, group_size=64, dtype=mx.float16,
        )

    @pytest.mark.parametrize("M", [1, 4, 16])
    def test_varied_M(self, M):
        """Multiple tokens per batch."""
        _run_and_compare(
            n_experts=2, M=M, K=512, N=512,
            bits=4, group_size=64, dtype=mx.float16,
        )


class TestQuantizationVariants:
    """Test across different quantization configs."""

    @pytest.mark.parametrize("bits", [4, 8])
    def test_varied_bits(self, bits):
        _run_and_compare(
            n_experts=2, M=1, K=1024, N=1024,
            bits=bits, group_size=64, dtype=mx.float16,
        )

    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_varied_group_size(self, group_size):
        _run_and_compare(
            n_experts=2, M=1, K=1024, N=1024,
            bits=4, group_size=group_size, dtype=mx.float16,
        )


class TestMultiExpert:
    """Test with varied expert counts and selection patterns."""

    @pytest.mark.parametrize("n_experts", [8, 16, 64])
    def test_many_experts(self, n_experts):
        _run_and_compare(
            n_experts=n_experts, M=1, K=512, N=512,
            bits=4, group_size=64, dtype=mx.float16,
        )

    def test_select_noncontiguous_experts(self):
        """Select non-adjacent experts (e.g. experts 1 and 5 out of 8)."""
        K, N, n_experts = 512, 512, 8
        gate_w, gate_s, gate_b = _quantize_experts(n_experts, N, K, 4, 64)
        up_w, up_s, up_b = _quantize_experts(n_experts, N, K, 4, 64)
        x = mx.random.normal((1, 1, K)).astype(mx.float16) * 0.1

        lhs_indices = mx.zeros((2,), dtype=mx.uint32)
        rhs_indices = mx.array([1, 5], dtype=mx.uint32)

        result = mx.gather_qmm_swiglu(
            x, gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            lhs_indices=lhs_indices, rhs_indices=rhs_indices,
            transpose=True, group_size=64, bits=4,
        )
        expected = _reference_swiglu(
            x, gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            lhs_indices=lhs_indices, rhs_indices=rhs_indices,
            group_size=64, bits=4,
        )
        mx.eval(result, expected)
        a = np.array(result.tolist(), dtype=np.float32)
        b = np.array(expected.tolist(), dtype=np.float32)
        np.testing.assert_allclose(a, b, atol=1e-2, rtol=1e-2)


class TestDtype:
    """Test across different compute dtypes."""

    def test_float16(self):
        _run_and_compare(
            n_experts=2, M=1, K=512, N=512,
            bits=4, group_size=64, dtype=mx.float16,
        )

    def test_bfloat16(self):
        _run_and_compare(
            n_experts=2, M=1, K=512, N=512,
            bits=4, group_size=64, dtype=mx.bfloat16,
        )


class TestNoIndices:
    """Test with None indices (batch-dimension broadcasting)."""

    def test_no_lhs_indices(self):
        """When lhs_indices is None, x is broadcast."""
        K, N = 512, 512
        gate_w, gate_s, gate_b = _quantize_experts(2, N, K, 4, 64)
        up_w, up_s, up_b = _quantize_experts(2, N, K, 4, 64)
        x = mx.random.normal((2, 1, K)).astype(mx.float16) * 0.1

        result = mx.gather_qmm_swiglu(
            x, gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            transpose=True, group_size=64, bits=4,
        )
        expected = _reference_swiglu(
            x, gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            lhs_indices=None, rhs_indices=None,
            group_size=64, bits=4,
        )
        mx.eval(result, expected)
        a = np.array(result.tolist(), dtype=np.float32)
        b = np.array(expected.tolist(), dtype=np.float32)
        np.testing.assert_allclose(a, b, atol=1e-2, rtol=1e-2)


class TestQwen3Shapes:
    """Shapes matching real Qwen3-30B-A3B expert dimensions."""

    def test_qwen3_expert_shape(self):
        """Qwen3-30B-A3B: K=2048, N=1024 (intermediate_size // 2)."""
        _run_and_compare(
            n_experts=8, M=1, K=2048, N=1024,
            bits=4, group_size=64, dtype=mx.float16,
        )

    def test_qwen3_larger_intermediate(self):
        """Qwen3-30B-A3B layer variant: K=2048, N=2048."""
        _run_and_compare(
            n_experts=8, M=1, K=2048, N=2048,
            bits=4, group_size=64, dtype=mx.float16,
        )
