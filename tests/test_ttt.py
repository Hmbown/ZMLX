"""Tests for TTT-Linear layer and fused Metal decode kernel."""

from __future__ import annotations

import pytest

# Skip entire module on non-Apple-Silicon
pytest.importorskip("mlx.core")

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from zmlx.ttt.linear import (
    TTTCache,
    TTTLinear,
    TTTLinearConfig,
    _layernorm_bwd,
    _layernorm_fwd,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _randn(*shape, dtype=mx.float32):
    return mx.random.normal(shape).astype(dtype)


def _allclose(a, b, atol=1e-4, rtol=1e-3):
    a_np = np.array(a.astype(mx.float32))
    b_np = np.array(b.astype(mx.float32))
    return np.allclose(a_np, b_np, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Unit tests: LayerNorm helpers
# ---------------------------------------------------------------------------


class TestLayerNormHelpers:
    def test_layernorm_fwd_matches_mlx(self):
        """_layernorm_fwd matches nn.LayerNorm output."""
        f = 64
        x = _randn(4, 1, f)
        w = mx.ones((1, 1, f))
        b = mx.zeros((1, 1, f))
        w_broad = mx.broadcast_to(w, x.shape)
        b_broad = mx.broadcast_to(b, x.shape)

        out, x_hat, std = _layernorm_fwd(x, w_broad, b_broad)

        # Compare with MLX LayerNorm
        ln = nn.LayerNorm(f)
        ref = ln(x)

        assert _allclose(out, ref, atol=1e-4), (
            f"max diff: {np.max(np.abs(np.array(out) - np.array(ref)))}"
        )

    def test_layernorm_bwd_gradient(self):
        """_layernorm_bwd produces correct gradient direction."""
        f = 64
        x = _randn(2, 1, f)
        w = mx.ones((2, 1, f))
        b = mx.zeros((2, 1, f))

        out, x_hat, std = _layernorm_fwd(x, w, b)
        # Use a simple gradient: gradient of sum(out)
        grad_out = mx.ones_like(out)
        grad_x = _layernorm_bwd(grad_out, x_hat, std, w)

        # Gradient should have same shape
        assert grad_x.shape == x.shape

        # Numerical check: grad should be approximately zero-mean per row
        # (since LN output is shift-invariant)
        grad_mean = mx.mean(grad_x, axis=-1)
        assert _allclose(grad_mean, mx.zeros_like(grad_mean), atol=1e-3)


# ---------------------------------------------------------------------------
# Unit tests: TTTLinearConfig and TTTCache
# ---------------------------------------------------------------------------


class TestTTTConfig:
    def test_defaults(self):
        c = TTTLinearConfig()
        assert c.hidden_size == 2048
        assert c.num_heads == 32
        assert c.head_dim == 64
        assert c.mini_batch_size == 16

    def test_custom(self):
        c = TTTLinearConfig(hidden_size=512, num_heads=8, head_dim=64)
        assert c.hidden_size == 512
        assert c.num_heads == 8


class TestTTTCache:
    def test_creation(self):
        c = TTTLinearConfig(hidden_size=128, num_heads=2, head_dim=64)
        W1 = mx.zeros((c.num_heads, c.head_dim, c.head_dim))
        b1 = mx.zeros((c.num_heads, 1, c.head_dim))
        cache = TTTCache(
            batch_size=1,
            num_heads=c.num_heads,
            head_dim=c.head_dim,
            W1_init=W1,
            b1_init=b1,
        )
        assert cache.W1.shape == (2, 64, 64)
        assert cache.b1.shape == (2, 1, 64)
        assert cache.W1_grad.shape == (2, 64, 64)
        assert cache.seq_offset == 0


# ---------------------------------------------------------------------------
# Integration tests: TTTLinear layer
# ---------------------------------------------------------------------------


class TestTTTLinear:
    @pytest.fixture
    def small_config(self):
        """Small config for fast tests."""
        return TTTLinearConfig(
            hidden_size=128,
            num_heads=2,
            head_dim=64,
            mini_batch_size=4,
            ttt_base_lr=1.0,
        )

    def test_construction(self, small_config):
        layer = TTTLinear(small_config)
        assert layer.W1.shape == (2, 64, 64)
        assert layer.b1.shape == (2, 1, 64)

    def test_prefill_shape(self, small_config):
        """Prefill produces correct output shape."""
        layer = TTTLinear(small_config)
        B, N, D = 1, 8, small_config.hidden_size
        x = _randn(B, N, D)

        out = layer(x)
        assert out.shape == (B, N, D), f"Expected {(B, N, D)}, got {out.shape}"

    def test_decode_shape(self, small_config):
        """Single-token decode produces correct output shape."""
        layer = TTTLinear(small_config)
        B, D = 1, small_config.hidden_size
        cache = layer.create_cache(batch_size=B)

        x = _randn(B, 1, D)
        out = layer(x, cache=cache)
        assert out.shape == (B, 1, D), f"Expected {(B, 1, D)}, got {out.shape}"

    def test_decode_updates_cache(self, small_config):
        """Decode step increments cache seq_offset."""
        layer = TTTLinear(small_config)
        cache = layer.create_cache(batch_size=1)

        x = _randn(1, 1, small_config.hidden_size)
        layer(x, cache=cache)
        assert cache.seq_offset == 1

        layer(x, cache=cache)
        assert cache.seq_offset == 2

    def test_decode_state_updates_at_boundary(self, small_config):
        """W1 state updates at mini-batch boundary."""
        layer = TTTLinear(small_config)
        cache = layer.create_cache(batch_size=1)

        W1_initial = mx.array(cache.W1)  # copy
        x = _randn(1, 1, small_config.hidden_size)

        # Process mini_batch_size tokens
        for _ in range(small_config.mini_batch_size):
            layer(x, cache=cache)

        # After a full mini-batch, W1 should have been updated
        mx.eval(cache.W1)
        mx.eval(W1_initial)
        diff = float(mx.sum(mx.abs(cache.W1 - W1_initial)).item())
        assert diff > 0, "W1 should change after a full mini-batch"

    def test_multiple_decode_steps_deterministic(self, small_config):
        """Running the same sequence produces identical output."""
        layer = TTTLinear(small_config)
        D = small_config.hidden_size

        tokens = [_randn(1, 1, D) for _ in range(6)]

        # Run 1
        cache1 = layer.create_cache(batch_size=1)
        outs1 = [layer(t, cache=cache1) for t in tokens]

        # Run 2
        cache2 = layer.create_cache(batch_size=1)
        outs2 = [layer(t, cache=cache2) for t in tokens]

        for i, (o1, o2) in enumerate(zip(outs1, outs2, strict=True)):
            mx.eval(o1, o2)
            assert _allclose(o1, o2, atol=1e-5), f"Step {i} diverged"

    def test_batch_size_2(self, small_config):
        """Batch size > 1 works for prefill."""
        layer = TTTLinear(small_config)
        B, N, D = 2, 4, small_config.hidden_size
        x = _randn(B, N, D)

        out = layer(x)
        assert out.shape == (B, N, D)


# ---------------------------------------------------------------------------
# Metal kernel tests (skip if no Metal)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"),
    reason="Metal kernels not available",
)
class TestTTTMetalKernel:
    def test_kernel_compiles(self):
        """The fused decode kernel compiles without error."""
        from zmlx.ttt.kernel import _ttt_linear_decode_kernel

        kern = _ttt_linear_decode_kernel(64, 64)
        assert kern is not None

    def test_kernel_vs_reference(self):
        """Fused kernel output matches reference MLX implementation."""
        from zmlx.ttt.kernel import ttt_linear_decode

        F = 64
        B_nh = 2  # 1 batch, 2 heads

        # Random inputs
        xq = _randn(B_nh, F)
        xk = _randn(B_nh, F)
        xv = _randn(B_nh, F)
        ttt_lr = mx.full((B_nh,), 0.01)
        token_idx = mx.array([0.5])
        last_in_mb = mx.array([0], dtype=mx.int32)
        W1 = _randn(B_nh, F, F) * 0.02
        b1 = mx.zeros((B_nh, F))
        W1_grad = mx.zeros((B_nh, F, F))
        b1_grad = mx.zeros((B_nh, F))
        ln_weight = mx.ones((B_nh, F))
        ln_bias = mx.zeros((B_nh, F))

        # Run kernel
        out_k, W1_k, b1_k, W1g_k, b1g_k = ttt_linear_decode(
            xq, xk, xv, ttt_lr, token_idx, last_in_mb,
            W1, b1, W1_grad, b1_grad, ln_weight, ln_bias,
        )
        mx.eval(out_k, W1_k, b1_k, W1g_k, b1g_k)

        # Reference: step by step in MLX
        # Step 1: Z1 = xk @ W1 + b1
        xk_3d = xk[:, None, :]  # [B_nh, 1, F]
        xq_3d = xq[:, None, :]
        xv_3d = xv[:, None, :]
        b1_3d = b1[:, None, :]
        Z1 = (xk_3d @ W1) + b1_3d  # [B_nh, 1, F]

        # Step 2
        l2_target = xv_3d - xk_3d

        # Step 3-4: LN forward
        ln_w_3d = ln_weight[:, None, :]
        ln_b_3d = ln_bias[:, None, :]
        LN_out, Z1_hat, std = _layernorm_fwd(Z1, ln_w_3d, ln_b_3d)

        # Step 5
        dl_dLN = LN_out - l2_target

        # Step 6
        dl_dZ1 = _layernorm_bwd(dl_dLN, Z1_hat, std, ln_w_3d)

        # Step 7
        lr_3d = ttt_lr[:, None, None]
        scaled = lr_3d * dl_dZ1

        # Step 8: outer product [B_nh, F, 1] @ [B_nh, 1, F] = [B_nh, F, F]
        W1_grad_new = W1_grad + (mx.transpose(xk_3d, (0, 2, 1)) @ scaled)
        b1_grad_new = b1_grad + scaled.squeeze(1)

        # Step 9
        ti = float(token_idx[0].item())
        W1_bar = W1 - ti * W1_grad_new
        b1_bar = b1 - ti * b1_grad_new

        # Step 10
        b1_bar_3d = b1_bar[:, None, :]
        Z1_bar = (xq_3d @ W1_bar) + b1_bar_3d

        # Step 11
        LN_bar, _, _ = _layernorm_fwd(Z1_bar, ln_w_3d, ln_b_3d)

        # Step 12
        ref_out = (xq_3d + LN_bar).squeeze(1)

        mx.eval(ref_out, W1_grad_new, b1_grad_new)

        assert _allclose(out_k, ref_out, atol=1e-3, rtol=1e-2), (
            f"Output max diff: {np.max(np.abs(np.array(out_k) - np.array(ref_out)))}"
        )
        assert _allclose(W1g_k, W1_grad_new, atol=1e-3, rtol=1e-2), (
            f"W1_grad max diff: {np.max(np.abs(np.array(W1g_k) - np.array(W1_grad_new)))}"
        )
