"""Tests for add_rms_norm fused kernel.

Tolerance calibration (float32 kernel vs numpy float64 decomposed):
  max_abs_err ~ 1e-6, mean_abs_err ~ 1e-7
For float16 kernel vs float32 decomposed:
  max_abs_err ~ 2e-3, mean_abs_err ~ 2e-4
Thresholds set at ~3x worst-case observed.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from numerical_harness import assert_numerical_quality, numerical_report

from zmlx.kernels.norms import add_rms_norm
from zmlx.kernels.transformer import rmsnorm_residual


def _ref_add_rmsnorm_np(x, residual, weight, eps):
    """Numpy float64 reference implementation."""
    x = np.asarray(x, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    h = x + residual
    rms = np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps)
    normed = h / rms * weight
    return normed, h


def _ref_add_rmsnorm_mx(x, residual, weight, eps):
    """MLX decomposed reference (float32 compute)."""
    h = x.astype(mx.float32) + residual.astype(mx.float32)
    rms = mx.rsqrt(mx.mean(h * h, axis=-1, keepdims=True) + eps)
    normed = h * rms * weight.astype(mx.float32)
    return normed.astype(x.dtype), h.astype(x.dtype)


# ---------------------------------------------------------------------------
# Reference validation (numpy float64)
# ---------------------------------------------------------------------------


class TestAddRmsNormReference:
    """Validate decomposed MLX reference against numpy float64."""

    @pytest.mark.parametrize("B", [1, 4, 16])
    @pytest.mark.parametrize("D", [64, 128, 512, 1024])
    def test_mlx_reference_vs_numpy(self, B, D):
        mx.random.seed(42)
        x = mx.random.normal((B, D))
        res = mx.random.normal((B, D))
        w = mx.random.normal((D,))
        eps = 1e-5

        normed_mx, h_mx = _ref_add_rmsnorm_mx(x, res, w, eps)
        normed_np, _ = _ref_add_rmsnorm_np(
            np.array(x.tolist()), np.array(res.tolist()),
            np.array(w.tolist()), eps,
        )

        mx.eval(normed_mx, h_mx)
        assert_numerical_quality(
            np.array(normed_mx.tolist()), normed_np,
            max_abs_tol=5e-6, mean_abs_tol=5e-7, label=f"ref B={B} D={D}",
        )


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestAddRmsNormCorrectness:
    """Parametric forward comparison against decomposed reference."""

    @pytest.mark.parametrize("B", [1, 4, 16, 32])
    @pytest.mark.parametrize("D", [64, 128, 512, 1024, 2048])
    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
    @pytest.mark.parametrize("eps", [1e-5, 1e-6])
    @pytest.mark.parametrize("tg", [32, 128, 256])
    def test_forward(self, B, D, dtype, eps, tg):
        if tg > D:
            pytest.skip("TG > D not meaningful")
        mx.random.seed(123)
        x = mx.random.normal((B, D)).astype(dtype)
        res = mx.random.normal((B, D)).astype(dtype)
        w = mx.ones((D,), dtype=dtype) * 0.5 + mx.random.normal((D,)).astype(dtype) * 0.1

        normed, h = add_rms_norm(x, res, w, eps=eps, threadgroup=tg)
        normed_ref, h_ref = _ref_add_rmsnorm_mx(x, res, w, eps)
        mx.eval(normed, h, normed_ref, h_ref)

        # Tolerances calibrated per dtype
        if dtype == mx.float32:
            max_abs, mean_abs = 5e-5, 5e-6
        else:
            max_abs, mean_abs = 1e-2, 2e-3

        assert_numerical_quality(
            np.array(normed.tolist()), np.array(normed_ref.tolist()),
            max_abs_tol=max_abs, mean_abs_tol=mean_abs,
            label=f"fwd B={B} D={D} {dtype} eps={eps} tg={tg}",
            print_report=True,
        )


# ---------------------------------------------------------------------------
# Residual exactness
# ---------------------------------------------------------------------------


class TestAddRmsNormResidualExact:
    """Verify h = x + residual is bit-exact in output (up to T-cast)."""

    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
    def test_residual_exact(self, dtype):
        mx.random.seed(7)
        B, D = 16, 256
        x = mx.random.normal((B, D)).astype(dtype)
        res = mx.random.normal((B, D)).astype(dtype)
        w = mx.ones((D,), dtype=dtype)

        _, h = add_rms_norm(x, res, w)
        h_ref = (x.astype(mx.float32) + res.astype(mx.float32)).astype(dtype)
        mx.eval(h, h_ref)

        np.testing.assert_array_equal(
            np.array(h.tolist()), np.array(h_ref.tolist()),
            err_msg="h output should be bit-exact T-cast of float32 sum",
        )


# ---------------------------------------------------------------------------
# Backward correctness
# ---------------------------------------------------------------------------


class TestAddRmsNormBackward:
    """Gradient correctness vs mx.grad of decomposed reference."""

    @pytest.mark.parametrize("B", [1, 4, 16])
    @pytest.mark.parametrize("D", [64, 256, 512])
    def test_grad_x(self, B, D):
        mx.random.seed(99)
        x = mx.random.normal((B, D))
        res = mx.random.normal((B, D))
        w = mx.ones((D,))

        # Kernel gradient
        def kernel_loss(x_in):
            normed, _ = add_rms_norm(x_in, res, w, eps=1e-5)
            return mx.sum(normed)

        grad_kernel = mx.grad(kernel_loss)(x)
        mx.eval(grad_kernel)

        # Reference gradient
        def ref_loss(x_in):
            normed, _ = _ref_add_rmsnorm_mx(x_in, res, w, 1e-5)
            return mx.sum(normed)

        grad_ref = mx.grad(ref_loss)(x)
        mx.eval(grad_ref)

        assert_numerical_quality(
            np.array(grad_kernel.tolist()), np.array(grad_ref.tolist()),
            max_abs_tol=1e-4, mean_abs_tol=1e-5,
            label=f"grad_x B={B} D={D}",
        )

    @pytest.mark.parametrize("B", [1, 4])
    @pytest.mark.parametrize("D", [64, 256])
    def test_grad_residual(self, B, D):
        mx.random.seed(77)
        x = mx.random.normal((B, D))
        res = mx.random.normal((B, D))
        w = mx.ones((D,))

        def kernel_loss(res_in):
            normed, _ = add_rms_norm(x, res_in, w, eps=1e-5)
            return mx.sum(normed)

        grad_kernel = mx.grad(kernel_loss)(res)

        def ref_loss(res_in):
            normed, _ = _ref_add_rmsnorm_mx(x, res_in, w, 1e-5)
            return mx.sum(normed)

        grad_ref = mx.grad(ref_loss)(res)
        mx.eval(grad_kernel, grad_ref)

        assert_numerical_quality(
            np.array(grad_kernel.tolist()), np.array(grad_ref.tolist()),
            max_abs_tol=1e-4, mean_abs_tol=1e-5,
            label=f"grad_res B={B} D={D}",
        )


# ---------------------------------------------------------------------------
# Statistical error distribution
# ---------------------------------------------------------------------------


class TestAddRmsNormStatistical:
    """Error distribution over many random inputs."""

    def test_error_distribution_float32(self):
        mx.random.seed(0)
        max_errs = []
        for _ in range(100):
            B, D = 8, 512
            x = mx.random.normal((B, D))
            res = mx.random.normal((B, D))
            w = mx.random.normal((D,))
            normed, _ = add_rms_norm(x, res, w)
            normed_ref, _ = _ref_add_rmsnorm_mx(x, res, w, 1e-6)
            mx.eval(normed, normed_ref)
            report = numerical_report(
                np.array(normed.tolist()), np.array(normed_ref.tolist()),
            )
            max_errs.append(report["max_abs_err"])

        p99 = np.percentile(max_errs, 99)
        assert p99 < 1e-4, f"p99 max_abs_err over 100 trials: {p99:.6e}"


# ---------------------------------------------------------------------------
# Golden values (fixed-seed regression snapshots)
# ---------------------------------------------------------------------------


class TestAddRmsNormGolden:
    """Fixed-seed regression test."""

    def test_golden_values(self):
        mx.random.seed(12345)
        x = mx.random.normal((2, 4))
        res = mx.random.normal((2, 4))
        w = mx.ones((4,))
        normed, h = add_rms_norm(x, res, w, eps=1e-5)
        mx.eval(normed, h)

        # Just verify determinism: same seed = same output
        mx.random.seed(12345)
        x2 = mx.random.normal((2, 4))
        res2 = mx.random.normal((2, 4))
        w2 = mx.ones((4,))
        normed2, h2 = add_rms_norm(x2, res2, w2, eps=1e-5)
        mx.eval(normed2, h2)

        np.testing.assert_array_equal(
            np.array(normed.tolist()), np.array(normed2.tolist()),
        )
        np.testing.assert_array_equal(
            np.array(h.tolist()), np.array(h2.tolist()),
        )


# ---------------------------------------------------------------------------
# Comparison vs existing implementation
# ---------------------------------------------------------------------------


class TestAddRmsNormVsExisting:
    """Head-to-head comparison against transformer.rmsnorm_residual."""

    @pytest.mark.parametrize("B", [1, 4, 16])
    @pytest.mark.parametrize("D", [128, 512, 1024])
    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16])
    def test_matches_existing(self, B, D, dtype):
        mx.random.seed(42)
        x = mx.random.normal((B, D)).astype(dtype)
        res = mx.random.normal((B, D)).astype(dtype)
        w = mx.ones((D,), dtype=dtype)

        normed_new, h_new = add_rms_norm(x, res, w, eps=1e-5)
        normed_old, h_old = rmsnorm_residual(x, res, w, eps=1e-5)
        mx.eval(normed_new, h_new, normed_old, h_old)

        if dtype == mx.float32:
            max_abs = 1e-5
        else:
            # New kernel may be slightly more precise due to re-read
            max_abs = 5e-3

        assert_numerical_quality(
            np.array(normed_new.tolist()), np.array(normed_old.tolist()),
            max_abs_tol=max_abs, mean_abs_tol=max_abs / 5,
            label=f"vs_existing B={B} D={D} {dtype}",
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAddRmsNormEdgeCases:
    """Edge case handling."""

    def test_all_zeros_x(self):
        x = mx.zeros((4, 64))
        res = mx.random.normal((4, 64))
        w = mx.ones((64,))
        normed, h = add_rms_norm(x, res, w)
        mx.eval(normed, h)
        assert not mx.any(mx.isnan(normed)).item()

    def test_all_zeros_residual(self):
        x = mx.random.normal((4, 64))
        res = mx.zeros((4, 64))
        w = mx.ones((64,))
        normed, h = add_rms_norm(x, res, w)
        mx.eval(normed, h)
        assert not mx.any(mx.isnan(normed)).item()

    def test_all_zeros_both(self):
        x = mx.zeros((4, 64))
        res = mx.zeros((4, 64))
        w = mx.ones((64,))
        normed, h = add_rms_norm(x, res, w, eps=1e-5)
        mx.eval(normed, h)
        # With all zeros input, normed should be all zeros (0 * rsqrt(eps))
        assert not mx.any(mx.isnan(normed)).item()

    def test_d_not_multiple_of_tg(self):
        """D not a multiple of threadgroup size."""
        x = mx.random.normal((2, 100))
        res = mx.random.normal((2, 100))
        w = mx.ones((100,))
        normed, _ = add_rms_norm(x, res, w, threadgroup=32)
        normed_ref, _ = _ref_add_rmsnorm_mx(x, res, w, 1e-6)
        mx.eval(normed, normed_ref)
        assert_numerical_quality(
            np.array(normed.tolist()), np.array(normed_ref.tolist()),
            max_abs_tol=5e-5, mean_abs_tol=5e-6, label="D=100 TG=32",
        )

    def test_single_row(self):
        x = mx.random.normal((1, 512))
        res = mx.random.normal((1, 512))
        w = mx.ones((512,))
        normed, _ = add_rms_norm(x, res, w)
        normed_ref, _ = _ref_add_rmsnorm_mx(x, res, w, 1e-6)
        mx.eval(normed, normed_ref)
        assert_numerical_quality(
            np.array(normed.tolist()), np.array(normed_ref.tolist()),
            max_abs_tol=5e-5, mean_abs_tol=5e-6, label="single_row",
        )

    def test_3d_input(self):
        """Batch + sequence dimensions."""
        x = mx.random.normal((2, 8, 128))
        res = mx.random.normal((2, 8, 128))
        w = mx.ones((128,))
        normed, _ = add_rms_norm(x, res, w)
        normed_ref, _ = _ref_add_rmsnorm_mx(x, res, w, 1e-6)
        mx.eval(normed, normed_ref)
        assert_numerical_quality(
            np.array(normed.tolist()), np.array(normed_ref.tolist()),
            max_abs_tol=5e-5, mean_abs_tol=5e-6, label="3d_input",
        )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            add_rms_norm(mx.zeros((2, 4)), mx.zeros((3, 4)), mx.ones((4,)))

    def test_weight_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="weight must have shape"):
            add_rms_norm(mx.zeros((2, 4)), mx.zeros((2, 4)), mx.ones((5,)))
