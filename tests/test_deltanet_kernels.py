"""Tests for fused DeltaNet kernels.

Validates correctness of:
  - fused_conv1d_silu against separate conv1d + silu
  - gated_rmsnorm_silu against separate rmsnorm + silu + multiply
  - fused_input_proj against separate linear projections
"""

import pytest

from zmlx._compat import import_mx

mx = import_mx()
if mx is None:
    pytest.skip("MLX not available", allow_module_level=True)

import mlx.nn as nn
import numpy as np

from zmlx.kernels.deltanet import (
    fused_conv1d_silu,
    fused_input_proj,
    gated_rmsnorm_silu,
)


# ---------------------------------------------------------------------------
# fused_conv1d_silu tests
# ---------------------------------------------------------------------------
class TestFusedConv1dSiLU:
    """Test fused depthwise conv1d + SiLU against reference."""

    @pytest.mark.parametrize("conv_dim", [512, 2048, 8192])
    @pytest.mark.parametrize("kernel_size", [4])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_correctness(self, conv_dim, kernel_size, batch_size):
        """Output matches separate conv1d + silu."""
        mx.random.seed(42)
        # Create conv_input: [B, kernel_size, conv_dim]
        conv_input = mx.random.normal((batch_size, kernel_size, conv_dim))

        # Create depthwise conv weights: [conv_dim, kernel_size, 1]
        # (MLX Conv1d weight shape: [out_channels, kernel_size, in_channels/groups])
        weight_3d = mx.random.normal((conv_dim, kernel_size, 1))

        # Reference: nn.Conv1d + silu
        conv = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=kernel_size,
            groups=conv_dim,
            bias=False,
            padding=0,
        )
        conv.weight = weight_3d
        ref_conv_out = conv(conv_input)  # [B, 1, conv_dim]
        ref_out = nn.silu(ref_conv_out)
        mx.eval(ref_out)

        # Fused kernel
        fused_out, new_state = fused_conv1d_silu(conv_input, weight_3d)
        mx.eval(fused_out, new_state)

        # Compare
        np.testing.assert_allclose(
            np.array(fused_out),
            np.array(ref_out),
            atol=1e-4,
            rtol=1e-3,
        )

        # Check new_state = conv_input[:, 1:, :]
        expected_state = conv_input[:, 1:, :]
        np.testing.assert_allclose(
            np.array(new_state),
            np.array(expected_state),
            atol=1e-6,
        )

    def test_fp16(self):
        """Works with float16 inputs."""
        mx.random.seed(42)
        conv_dim = 512
        kernel_size = 4
        conv_input = mx.random.normal((1, kernel_size, conv_dim)).astype(mx.float16)
        weight = mx.random.normal((conv_dim, kernel_size, 1)).astype(mx.float16)

        fused_out, new_state = fused_conv1d_silu(conv_input, weight)
        mx.eval(fused_out, new_state)

        assert fused_out.dtype == mx.float16
        assert fused_out.shape == (1, 1, conv_dim)
        assert new_state.shape == (1, kernel_size - 1, conv_dim)

    def test_qwen35_dims(self):
        """Correct for Qwen3.5-35B dimensions: conv_dim=8192, kernel_size=4."""
        mx.random.seed(42)
        conv_dim = 8192
        kernel_size = 4

        conv_input = mx.random.normal((1, kernel_size, conv_dim))
        weight = mx.random.normal((conv_dim, kernel_size, 1))

        conv = nn.Conv1d(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=kernel_size, groups=conv_dim, bias=False, padding=0,
        )
        conv.weight = weight
        ref_out = nn.silu(conv(conv_input))
        mx.eval(ref_out)

        fused_out, _ = fused_conv1d_silu(conv_input, weight)
        mx.eval(fused_out)

        np.testing.assert_allclose(
            np.array(fused_out), np.array(ref_out),
            atol=1e-4, rtol=1e-3,
        )


# ---------------------------------------------------------------------------
# gated_rmsnorm_silu tests
# ---------------------------------------------------------------------------
class TestGatedRMSNormSiLU:
    """Test fused gated RMSNorm + SiLU against reference."""

    @pytest.mark.parametrize("d", [64, 128, 256])
    @pytest.mark.parametrize("rows", [1, 32, 64])
    def test_correctness(self, d, rows):
        """Output matches separate rmsnorm + silu + multiply."""
        mx.random.seed(42)
        y = mx.random.normal((rows, d))
        z = mx.random.normal((rows, d))
        weight = mx.ones(d) + mx.random.normal((d,)) * 0.1
        eps = 1e-6

        # Reference: rmsnorm(y) * silu(z)
        ref_normed = mx.fast.rms_norm(y, weight, eps)
        ref_silu = nn.silu(z)
        ref_out = ref_normed * ref_silu
        mx.eval(ref_out)

        # Fused kernel
        fused_out = gated_rmsnorm_silu(y, z, weight, eps=eps)
        mx.eval(fused_out)

        np.testing.assert_allclose(
            np.array(fused_out), np.array(ref_out),
            atol=1e-4, rtol=1e-3,
        )

    def test_4d_shape(self):
        """Works with [B, S, Hv, Dv] shape (DeltaNet output shape)."""
        mx.random.seed(42)
        B, S, Hv, Dv = 1, 1, 32, 128
        y = mx.random.normal((B, S, Hv, Dv))
        z = mx.random.normal((B, S, Hv, Dv))
        weight = mx.ones(Dv) + mx.random.normal((Dv,)) * 0.1
        eps = 1e-6

        # Reference
        ref_normed = mx.fast.rms_norm(y, weight, eps)
        ref_out = ref_normed * nn.silu(z)
        mx.eval(ref_out)

        # Fused
        fused_out = gated_rmsnorm_silu(y, z, weight, eps=eps)
        mx.eval(fused_out)

        np.testing.assert_allclose(
            np.array(fused_out), np.array(ref_out),
            atol=1e-4, rtol=1e-3,
        )

    def test_fp16(self):
        """Works with float16 inputs."""
        mx.random.seed(42)
        y = mx.random.normal((32, 128)).astype(mx.float16)
        z = mx.random.normal((32, 128)).astype(mx.float16)
        weight = mx.ones(128).astype(mx.float16)

        fused_out = gated_rmsnorm_silu(y, z, weight)
        mx.eval(fused_out)
        assert fused_out.dtype == mx.float16

    def test_shape_mismatch_raises(self):
        """Raises if y and z shapes differ."""
        y = mx.zeros((2, 64))
        z = mx.zeros((2, 128))
        weight = mx.ones(64)
        with pytest.raises(ValueError, match="same shape"):
            gated_rmsnorm_silu(y, z, weight)

    def test_weight_shape_mismatch_raises(self):
        """Raises if weight shape doesn't match last dim."""
        y = mx.zeros((2, 64))
        z = mx.zeros((2, 64))
        weight = mx.ones(128)
        with pytest.raises(ValueError, match="weight must have shape"):
            gated_rmsnorm_silu(y, z, weight)


# ---------------------------------------------------------------------------
# fused_input_proj tests
# ---------------------------------------------------------------------------
class TestFusedInputProj:
    """Test fused input projections against separate nn.Linear calls."""

    def test_correctness(self):
        """Output matches 4 separate nn.Linear calls."""
        mx.random.seed(42)
        hidden = 4096
        key_dim = 2048
        value_dim = 4096
        num_v_heads = 32
        B, S = 1, 1

        x = mx.random.normal((B, S, hidden))

        # Create weight matrices (Linear stores [out, in])
        w_qkv = mx.random.normal((key_dim * 2 + value_dim, hidden))
        w_z = mx.random.normal((value_dim, hidden))
        w_b = mx.random.normal((num_v_heads, hidden))
        w_a = mx.random.normal((num_v_heads, hidden))

        # Reference: 4 separate matmuls
        ref_qkv = x @ w_qkv.T
        ref_z = x @ w_z.T
        ref_b = x @ w_b.T
        ref_a = x @ w_a.T
        mx.eval(ref_qkv, ref_z, ref_b, ref_a)

        # Fused
        f_qkv, f_z, f_b, f_a = fused_input_proj(x, w_qkv, w_z, w_b, w_a)
        mx.eval(f_qkv, f_z, f_b, f_a)

        np.testing.assert_allclose(
            np.array(f_qkv), np.array(ref_qkv), atol=1e-3, rtol=1e-3,
        )
        np.testing.assert_allclose(
            np.array(f_z), np.array(ref_z), atol=1e-3, rtol=1e-3,
        )
        np.testing.assert_allclose(
            np.array(f_b), np.array(ref_b), atol=1e-3, rtol=1e-3,
        )
        np.testing.assert_allclose(
            np.array(f_a), np.array(ref_a), atol=1e-3, rtol=1e-3,
        )

    def test_shapes(self):
        """Output shapes match expected dimensions."""
        mx.random.seed(42)
        hidden = 4096
        d_qkv = 8192
        d_z = 4096
        d_b = 32
        d_a = 32
        B, S = 1, 1

        x = mx.random.normal((B, S, hidden))
        w_qkv = mx.random.normal((d_qkv, hidden))
        w_z = mx.random.normal((d_z, hidden))
        w_b = mx.random.normal((d_b, hidden))
        w_a = mx.random.normal((d_a, hidden))

        qkv, z, b, a = fused_input_proj(x, w_qkv, w_z, w_b, w_a)
        mx.eval(qkv, z, b, a)

        assert qkv.shape == (B, S, d_qkv)
        assert z.shape == (B, S, d_z)
        assert b.shape == (B, S, d_b)
        assert a.shape == (B, S, d_a)
