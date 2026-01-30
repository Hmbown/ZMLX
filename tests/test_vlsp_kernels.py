"""Tests for VLSP kernels: fused_recurrent_step, depth_gate_sigmoid, grpo_advantage_norm."""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest

# Skip on non-Apple Silicon
pytestmark = pytest.mark.skipif(
    not hasattr(mx, "metal"),
    reason="Metal not available",
)


# ---------------------------------------------------------------------------
# Test 1: depth_gate_sigmoid
# ---------------------------------------------------------------------------


class TestDepthGateSigmoid:
    """Test differentiable depth prediction with STE backward."""

    def test_output_range(self):
        """Output should be in [0, k_max]."""
        from zmlx.kernels.vlsp import depth_gate_sigmoid

        x = mx.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        k_max = 8
        out = depth_gate_sigmoid(x, k_max=k_max)
        mx.eval(out)

        assert out.shape == x.shape
        for v in out.tolist():
            assert 0.0 <= v <= k_max + 0.01, f"Value {v} outside [0, {k_max}]"

    def test_sigmoid_at_zero(self):
        """sigmoid(0) = 0.5, so output should be k_max/2."""
        from zmlx.kernels.vlsp import depth_gate_sigmoid

        x = mx.array([0.0])
        out = depth_gate_sigmoid(x, k_max=8)
        mx.eval(out)
        assert abs(out.item() - 4.0) < 0.01

    def test_monotonic(self):
        """Output should increase with input (sigmoid is monotonic)."""
        from zmlx.kernels.vlsp import depth_gate_sigmoid

        x = mx.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = depth_gate_sigmoid(x, k_max=8)
        mx.eval(out)
        values = out.tolist()
        for i in range(1, len(values)):
            assert values[i] > values[i - 1], f"Not monotonic at index {i}"

    def test_gradient_flows(self):
        """Verify gradient flows via STE."""
        from zmlx.kernels.vlsp import depth_gate_sigmoid

        def loss_fn(x):
            return depth_gate_sigmoid(x, k_max=8).sum()

        x = mx.array([0.0, 1.0, -1.0])
        grad = mx.grad(loss_fn)(x)
        mx.eval(grad)

        # Gradient should be k_max * sig(x) * (1-sig(x))
        for i, xi in enumerate(x.tolist()):
            s = 1.0 / (1.0 + math.exp(-xi))
            expected = 8.0 * s * (1.0 - s)
            assert abs(grad[i].item() - expected) < 0.01, (
                f"Grad mismatch at {i}: got {grad[i].item()}, expected {expected}"
            )

    def test_different_k_max(self):
        """Test with different k_max values."""
        from zmlx.kernels.vlsp import depth_gate_sigmoid

        x = mx.array([0.0])
        for k_max in [2, 4, 8, 16]:
            out = depth_gate_sigmoid(x, k_max=k_max)
            mx.eval(out)
            expected = k_max / 2.0
            assert abs(out.item() - expected) < 0.1, f"k_max={k_max}: got {out.item()}"


# ---------------------------------------------------------------------------
# Test 2: grpo_advantage_norm
# ---------------------------------------------------------------------------


class TestGRPOAdvantageNorm:
    """Test fused GRPO advantage normalization."""

    def test_basic_normalization(self):
        """Output should have zero mean and unit std per group."""
        from zmlx.kernels.vlsp import grpo_advantage_norm

        rewards = mx.array([[1.0, 0.0, 1.0, 0.0]], dtype=mx.float32)
        adv = grpo_advantage_norm(rewards)
        mx.eval(adv)

        mean = adv.mean(axis=-1).item()
        assert abs(mean) < 0.01, f"Mean should be ~0, got {mean}"

    def test_reference_match(self):
        """Compare against reference MLX implementation."""
        from zmlx.kernels.vlsp import grpo_advantage_norm

        rewards = mx.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=mx.float32)

        # ZMLX kernel
        adv_zmlx = grpo_advantage_norm(rewards, eps=1e-6)
        mx.eval(adv_zmlx)

        # Reference
        mean = rewards.mean(axis=-1, keepdims=True)
        std = mx.sqrt(mx.mean((rewards - mean) ** 2, axis=-1, keepdims=True) + 1e-6)
        adv_ref = (rewards - mean) / std
        mx.eval(adv_ref)

        assert mx.allclose(adv_zmlx, adv_ref, atol=1e-4), (
            f"Kernel output doesn't match reference.\n"
            f"Kernel: {adv_zmlx.tolist()}\n"
            f"Ref: {adv_ref.tolist()}"
        )

    def test_uniform_rewards(self):
        """When all rewards are equal, advantages should be ~0."""
        from zmlx.kernels.vlsp import grpo_advantage_norm

        rewards = mx.array([[0.5, 0.5, 0.5, 0.5]], dtype=mx.float32)
        adv = grpo_advantage_norm(rewards)
        mx.eval(adv)

        for v in adv[0].tolist():
            assert abs(v) < 0.1, f"Expected ~0 for uniform rewards, got {v}"

    def test_binary_rewards(self):
        """Test with binary (correct/incorrect) rewards typical of GRPO."""
        from zmlx.kernels.vlsp import grpo_advantage_norm

        # 2 correct, 2 incorrect
        rewards = mx.array([[1.0, 0.0, 1.0, 0.0]], dtype=mx.float32)
        adv = grpo_advantage_norm(rewards)
        mx.eval(adv)

        values = adv[0].tolist()
        # Correct answers should have positive advantage
        assert values[0] > 0 and values[2] > 0
        # Incorrect should have negative
        assert values[1] < 0 and values[3] < 0

    def test_1d_input(self):
        """1D input should be auto-reshaped to (1, G)."""
        from zmlx.kernels.vlsp import grpo_advantage_norm

        rewards = mx.array([1.0, 0.0, 1.0, 0.0], dtype=mx.float32)
        adv = grpo_advantage_norm(rewards)
        mx.eval(adv)
        assert adv.shape == (1, 4)

    def test_batch(self):
        """Multiple groups (batch) should work."""
        from zmlx.kernels.vlsp import grpo_advantage_norm

        rewards = mx.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ], dtype=mx.float32)

        adv = grpo_advantage_norm(rewards)
        mx.eval(adv)
        assert adv.shape == (3, 4)

        # Each row should have mean ~0
        for row_idx in range(3):
            row_mean = adv[row_idx].mean().item()
            assert abs(row_mean) < 0.01, f"Row {row_idx} mean={row_mean}"


# ---------------------------------------------------------------------------
# Test 3: fused_recurrent_step
# ---------------------------------------------------------------------------


class TestFusedRecurrentStep:
    """Test fused RMSNorm + SiLU + gating + residual kernel."""

    def test_basic_output_shape(self):
        """Output should match input shape."""
        from zmlx.kernels.vlsp import fused_recurrent_step

        D = 128
        B = 4
        h = mx.random.normal((B, D))
        w_norm = mx.ones((D,))
        gate = mx.random.normal((B, D))
        alpha = mx.array([1.0])

        out = fused_recurrent_step(h, w_norm, gate, alpha)
        mx.eval(out)
        assert out.shape == (B, D)

    def test_reference_match(self):
        """Compare against reference MLX implementation."""
        from zmlx.kernels.vlsp import fused_recurrent_step

        D = 64
        B = 2
        mx.random.seed(42)
        h = mx.random.normal((B, D)).astype(mx.float32)
        w_norm = mx.random.normal((D,)).astype(mx.float32)
        gate = mx.random.normal((B, D)).astype(mx.float32)
        alpha = mx.array([0.5])
        eps = 1e-6

        # ZMLX kernel
        out_zmlx = fused_recurrent_step(h, w_norm, gate, alpha, eps=eps)
        mx.eval(out_zmlx)

        # Reference
        rms = mx.sqrt(mx.mean(h * h, axis=-1, keepdims=True) + eps)
        h_normed = h / rms * w_norm
        silu_normed = h_normed * mx.sigmoid(h_normed)
        out_ref = h + 0.5 * silu_normed * gate
        mx.eval(out_ref)

        assert mx.allclose(out_zmlx, out_ref, atol=1e-3), (
            f"Max diff: {mx.abs(out_zmlx - out_ref).max().item()}"
        )

    def test_alpha_zero_is_identity(self):
        """With alpha=0, output should equal input h."""
        from zmlx.kernels.vlsp import fused_recurrent_step

        D = 64
        h = mx.random.normal((2, D))
        w_norm = mx.ones((D,))
        gate = mx.random.normal((2, D))
        alpha = mx.array([0.0])

        out = fused_recurrent_step(h, w_norm, gate, alpha)
        mx.eval(out)
        assert mx.allclose(out, h, atol=1e-5)

    def test_single_row(self):
        """Should work with single sample."""
        from zmlx.kernels.vlsp import fused_recurrent_step

        D = 256
        h = mx.random.normal((1, D))
        w_norm = mx.ones((D,))
        gate = mx.ones((1, D))
        alpha = mx.array([1.0])

        out = fused_recurrent_step(h, w_norm, gate, alpha)
        mx.eval(out)
        assert out.shape == (1, D)


# ---------------------------------------------------------------------------
# Test 4: silu_mul_residual
# ---------------------------------------------------------------------------


class TestSiluMulResidual:
    """Test fused silu(gate) * up + residual."""

    def test_basic_correctness(self):
        """Compare against reference."""
        from zmlx.kernels.vlsp import silu_mul_residual

        gate = mx.array([1.0, -1.0, 0.0, 2.0])
        up = mx.array([1.0, 1.0, 1.0, 1.0])
        residual = mx.array([0.5, 0.5, 0.5, 0.5])

        out = silu_mul_residual(gate, up, residual)
        mx.eval(out)

        # Reference
        ref = nn.silu(gate) * up + residual
        mx.eval(ref)

        assert mx.allclose(out, ref, atol=1e-4), (
            f"Kernel: {out.tolist()}\nRef: {ref.tolist()}"
        )

    def test_gradient_flows(self):
        """Verify gradients for all three inputs."""
        from zmlx.kernels.vlsp import silu_mul_residual

        def loss_fn(g, u, r):
            return silu_mul_residual(g, u, r).sum()

        gate = mx.array([1.0, 0.0, -1.0])
        up = mx.array([2.0, 1.0, 0.5])
        residual = mx.array([0.1, 0.2, 0.3])

        grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
        dg, du, dr = grad_fn(gate, up, residual)
        mx.eval(dg, du, dr)

        # d/d(residual) should be all 1s (pass-through)
        assert mx.allclose(dr, mx.ones_like(dr), atol=1e-4), f"d_residual: {dr.tolist()}"

        # d/d(up) should be silu(gate)
        ref_du = nn.silu(gate)
        mx.eval(ref_du)
        assert mx.allclose(du, ref_du, atol=1e-4), f"d_up: {du.tolist()} vs {ref_du.tolist()}"

    def test_2d_input(self):
        """Should work with batched 2D input."""
        from zmlx.kernels.vlsp import silu_mul_residual

        gate = mx.random.normal((4, 128))
        up = mx.random.normal((4, 128))
        residual = mx.random.normal((4, 128))

        out = silu_mul_residual(gate, up, residual)
        mx.eval(out)
        assert out.shape == (4, 128)


# ---------------------------------------------------------------------------
# Integration test: recurrence loop
# ---------------------------------------------------------------------------


class TestRecurrenceLoop:
    """Test that kernels work together in a recurrence loop."""

    def test_variable_depth_recurrence(self):
        """Simulate K-step recurrence with depth gating."""
        from zmlx.kernels.vlsp import depth_gate_sigmoid, fused_recurrent_step

        D = 64
        B = 2
        K_MAX = 4

        h = mx.random.normal((B, D))
        w_norm = mx.ones((D,))
        alpha = mx.array([0.1])

        # Predict depth
        depth_logit = mx.array([0.5, -0.5])
        k_continuous = depth_gate_sigmoid(depth_logit, k_max=K_MAX)
        mx.eval(k_continuous)

        k_discrete = mx.clip(mx.round(k_continuous).astype(mx.int32), 1, K_MAX)
        mx.eval(k_discrete)

        # Run max K steps, masking per-sample
        for step in range(K_MAX):
            gate = mx.random.normal((B, D))
            h_new = fused_recurrent_step(h, w_norm, gate, alpha)
            mx.eval(h_new)

            # Mask: only update samples where step < k_discrete
            mask = mx.array([1.0 if step < k_discrete[b].item() else 0.0 for b in range(B)])
            mask = mask.reshape(B, 1)
            h = h_new * mask + h * (1.0 - mask)
            mx.eval(h)

        assert h.shape == (B, D)
        # h should have changed from initial
        assert not mx.allclose(h, mx.random.normal((B, D)), atol=0.01)
