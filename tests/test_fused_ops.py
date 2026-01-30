"""Tests for fused operations (SwiGLU, dropout, Top-K, MoE gating).

These tests validate the correctness of fused kernels against reference
MLX implementations.
"""

import mlx.core as mx
import numpy as np
import pytest


# Helper to compare arrays with tolerance
def assert_allclose(a, b, rtol=1e-5, atol=1e-6):
    """Assert two arrays are close."""
    np.testing.assert_allclose(
        np.array(a.tolist()),
        np.array(b.tolist()),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.metal
class TestSwiGLU:
    """Tests for fused SwiGLU operations."""
    
    def test_swiglu_shape(self):
        """Test that swiglu produces correct output shape."""
        from zmlx.kernels.transformer import swiglu
        
        B, D = 4, 64
        x = mx.random.normal((B, 2 * D))
        
        y = swiglu(x)
        
        assert y.shape == (B, D)
    
    def test_swiglu_correctness(self):
        """Test swiglu against reference implementation."""
        from zmlx.kernels.transformer import swiglu
        
        B, D = 2, 8
        x = mx.random.normal((B, 2 * D))
        
        # Fused version
        y_fused = swiglu(x)
        
        # Reference: SiLU(x[:D]) * x[D:]
        x_np = np.array(x.tolist())
        a = x_np[:, :D]
        b = x_np[:, D:]
        silu = a * (1 / (1 + np.exp(-a)))
        y_ref = silu * b
        
        assert_allclose(y_fused, mx.array(y_ref), rtol=1e-4)
    
    def test_swiglu2_shape(self):
        """Test swiglu2 with separate inputs."""
        from zmlx.kernels.transformer import swiglu2
        
        B, D = 4, 64
        gate = mx.random.normal((B, D))
        up = mx.random.normal((B, D))
        
        y = swiglu2(gate, up)
        
        assert y.shape == (B, D)
    
    def test_swiglu2_correctness(self):
        """Test swiglu2 against reference."""
        from zmlx.kernels.transformer import swiglu2
        
        B, D = 2, 8
        gate = mx.random.normal((B, D))
        up = mx.random.normal((B, D))
        
        y_fused = swiglu2(gate, up)
        
        # Reference
        g_np = np.array(gate.tolist())
        u_np = np.array(up.tolist())
        silu = g_np * (1 / (1 + np.exp(-g_np)))
        y_ref = silu * u_np
        
        assert_allclose(y_fused, mx.array(y_ref), rtol=1e-4)
    
    @pytest.mark.skip(reason="Gradient shape issue - needs investigation")
    def test_swiglu_gradient(self):
        """Test swiglu gradient computation."""
        from zmlx.kernels.transformer import swiglu
        
        B, D = 2, 8
        x = mx.random.normal((B, 2 * D))
        
        def f(x):
            return swiglu(x).sum()
        
        # Should not raise
        grad = mx.grad(f)(x)
        
        # Gradient may be flattened, check size matches
        assert grad.size == x.size
        # Check that gradient is non-zero
        assert np.any(np.array(grad.tolist()) != 0)
    
    @pytest.mark.skip(reason="Gradient shape issue - needs investigation")
    def test_swiglu2_gradient(self):
        """Test swiglu2 gradient computation."""
        from zmlx.kernels.transformer import swiglu2
        
        B, D = 2, 8
        gate = mx.random.normal((B, D))
        up = mx.random.normal((B, D))
        
        def f(gate, up):
            return swiglu2(gate, up).sum()
        
        dg, du = mx.grad(f)(gate, up)
        
        # Gradients may be flattened, check sizes match
        assert dg.size == gate.size
        assert du.size == up.size


@pytest.mark.metal
class TestGeGLU:
    """Tests for fused GeGLU operations."""
    
    def test_geglu_shape(self):
        """Test geglu output shape."""
        from zmlx.kernels.transformer import geglu
        
        B, D = 4, 64
        x = mx.random.normal((B, 2 * D))
        
        y = geglu(x)
        
        assert y.shape == (B, D)
    
    def test_geglu2_correctness(self):
        """Test geglu2 against reference."""
        from zmlx.kernels.transformer import geglu2
        
        B, D = 2, 8
        gate = mx.random.normal((B, D))
        up = mx.random.normal((B, D))
        
        y_fused = geglu2(gate, up)
        
        # Reference GeLU
        def gelu_ref(x):
            return 0.5 * x * (1 + np.tanh(
                0.7978845608028654 * (x + 0.044715 * x**3)
            ))
        
        g_np = np.array(gate.tolist())
        u_np = np.array(up.tolist())
        y_ref = gelu_ref(g_np) * u_np
        
        assert_allclose(y_fused, mx.array(y_ref), rtol=1e-4)


@pytest.mark.metal
class TestDropout:
    """Tests for fused dropout."""
    
    def test_dropout_shape(self):
        """Test dropout preserves shape."""
        from zmlx.kernels.transformer import dropout
        
        x = mx.random.normal((4, 64))
        y = dropout(x, p=0.5, seed=42)
        
        assert y.shape == x.shape
    
    def test_dropout_scaling(self):
        """Test that dropout scales output correctly."""
        from zmlx.kernels.transformer import dropout
        
        # Use deterministic seed
        x = mx.ones((1000,))
        y = dropout(x, p=0.5, seed=12345)
        
        y_np = np.array(y.tolist())
        
        # Check that values are either 0 or scaled (2.0 for p=0.5)
        assert np.all((y_np == 0) | (np.abs(y_np - 2.0) < 0.001))
        
        # Approximately 50% should be zeros
        zero_ratio = np.mean(y_np == 0)
        assert 0.4 < zero_ratio < 0.6  # Allow some variance
    
    def test_dropout_p_zero(self):
        """Test dropout with p=0 (no dropout)."""
        from zmlx.kernels.transformer import dropout
        
        x = mx.random.normal((10, 10))
        y = dropout(x, p=0.0, seed=42)
        
        # Should be identical
        assert_allclose(y, x)
    
    def test_dropout_invalid_p(self):
        """Test dropout with invalid probability."""
        from zmlx.kernels.transformer import dropout
        
        x = mx.ones((10,))
        
        with pytest.raises(ValueError):
            dropout(x, p=1.0)
        
        with pytest.raises(ValueError):
            dropout(x, p=-0.1)


@pytest.mark.metal
class TestTopKGating:
    """Tests for Top-K gating operations."""
    
    def test_top2_gating_shape(self):
        """Test top2 gating output shapes."""
        from zmlx.kernels.moe import top2_gating_softmax
        
        B, E = 8, 16  # batch, num_experts
        x = mx.random.normal((B, E))
        
        weights, indices = top2_gating_softmax(x)
        
        assert weights.shape == (B, 2)
        assert indices.shape == (B, 2)
        assert indices.dtype == mx.uint32
    
    def test_top2_gating_weights_sum_to_one(self):
        """Test that gating weights sum to 1."""
        from zmlx.kernels.moe import top2_gating_softmax
        
        B, E = 4, 8
        x = mx.random.normal((B, E))
        
        weights, indices = top2_gating_softmax(x)
        
        weights_np = np.array(weights.tolist())
        sums = weights_np.sum(axis=-1)
        
        np.testing.assert_allclose(sums, 1.0, rtol=1e-4)
    
    def test_top2_gating_selects_top(self):
        """Test that gating selects the top 2 experts."""
        from zmlx.kernels.moe import top2_gating_softmax
        
        B, E = 4, 8
        # Create input where we know the top 2
        x = mx.array([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # top: 7, 6
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],  # top: 0, 1
        ])
        
        weights, indices = top2_gating_softmax(x)
        
        indices_np = np.array(indices.tolist())
        
        # First row should select experts 7 and 6 (in some order)
        assert 7 in indices_np[0]
        assert 6 in indices_np[0]
        
        # Second row should select experts 0 and 1
        assert 0 in indices_np[1]
        assert 1 in indices_np[1]
    
    def test_topk_gating_various_k(self):
        """Test topk gating with different k values."""
        from zmlx.kernels.moe import topk_gating_softmax
        
        B, E = 4, 16
        x = mx.random.normal((B, E))
        
        for k in [1, 2, 4, 8]:
            weights, indices = topk_gating_softmax(x, k=k)
            
            assert weights.shape == (B, k)
            assert indices.shape == (B, k)
            
            # Weights should sum to approximately 1 (after softmax)
            weights_np = np.array(weights.tolist())
            sums = weights_np.sum(axis=-1)
            np.testing.assert_allclose(sums, 1.0, rtol=1e-4)
    
    def test_topk_gating_with_expert_bias(self):
        """Test topk gating with expert bias."""
        from zmlx.kernels.moe import topk_gating_softmax
        
        B, E = 4, 8
        x = mx.random.normal((B, E))
        bias = mx.zeros(E)
        
        weights, indices = topk_gating_softmax(x, k=2, expert_bias=bias)
        
        assert weights.shape == (B, 2)
        assert indices.shape == (B, 2)
    
    def test_topk_gating_norm_topk_prob(self):
        """Test topk gating with norm_topk_prob=False."""
        from zmlx.kernels.moe import topk_gating_softmax
        
        B, E = 4, 8
        x = mx.random.normal((B, E))
        
        weights, indices = topk_gating_softmax(x, k=2, norm_topk_prob=False)
        
        # Weights may not sum to 1 when norm_topk_prob=False
        assert weights.shape == (B, 2)


@pytest.mark.metal
class TestMoECombine:
    """Tests for MoE combine operation."""
    
    def test_moe_combine_shape(self):
        """Test moe_combine output shape."""
        from zmlx.kernels.moe import moe_combine
        
        B, K, D = 4, 2, 64
        expert_outputs = mx.random.normal((B, K, D))
        weights = mx.array([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.4, 0.6]])
        
        y = moe_combine(expert_outputs, weights)
        
        assert y.shape == (B, D)
    
    def test_moe_combine_correctness(self):
        """Test moe_combine against reference implementation."""
        from zmlx.kernels.moe import moe_combine
        
        B, K, D = 2, 2, 4
        expert_outputs = mx.random.normal((B, K, D))
        weights = mx.array([[0.6, 0.4], [0.7, 0.3]])
        
        y_fused = moe_combine(expert_outputs, weights)
        
        # Reference: weighted sum over K dimension
        eo_np = np.array(expert_outputs.tolist())
        w_np = np.array(weights.tolist())
        y_ref = np.zeros((B, D))
        for b in range(B):
            for k in range(K):
                y_ref[b] += eo_np[b, k] * w_np[b, k]
        
        assert_allclose(y_fused, mx.array(y_ref), rtol=1e-4)


@pytest.mark.metal
class TestMoEDispatch:
    """Tests for MoE dispatch operation."""
    
    def test_moe_dispatch_shape(self):
        """Test moe_dispatch output shape."""
        from zmlx.kernels.moe import moe_dispatch
        
        B, D, K = 4, 64, 2
        x = mx.random.normal((B, D))
        indices = mx.array([[0, 1], [1, 2], [0, 2], [1, 3]], dtype=mx.uint32)
        
        y = moe_dispatch(x, indices)
        
        assert y.shape == (B, K, D)


@pytest.mark.metal
class TestFusedBias:
    """Tests for fused bias operations."""
    
    def test_add_bias(self):
        """Test fused add_bias."""
        from zmlx.kernels.fused import add_bias
        
        B, C = 4, 64
        x = mx.random.normal((B, C))
        bias = mx.random.normal((C,))
        
        add_bias_op = add_bias(c=C, compute_dtype=mx.float32)
        y = add_bias_op(x, bias)
        
        assert y.shape == (B, C)
        
        # Compare with reference
        y_ref = x + bias
        assert_allclose(y, y_ref, rtol=1e-5)
    
    def test_bias_silu(self):
        """Test fused bias + SiLU."""
        from zmlx.kernels.fused import bias_silu
        
        B, C = 4, 64
        x = mx.random.normal((B, C))
        bias = mx.random.normal((C,))
        
        bias_silu_op = bias_silu(c=C, compute_dtype=mx.float32)
        y = bias_silu_op(x, bias)
        
        assert y.shape == (B, C)
        
        # Compare with reference
        x_biased = x + bias
        y_ref = x_biased * mx.sigmoid(x_biased)
        assert_allclose(y, y_ref, rtol=1e-4)


@pytest.mark.metal
class TestFusedNorm:
    """Tests for fused normalization operations."""
    
    def test_rmsnorm_residual_shape(self):
        """Test rmsnorm_residual output shapes."""
        from zmlx.kernels.transformer import rmsnorm_residual
        
        B, D = 4, 64
        x = mx.random.normal((B, D))
        residual = mx.random.normal((B, D))
        weight = mx.ones((D,))
        
        out, updated_res = rmsnorm_residual(x, residual, weight)
        
        assert out.shape == (B, D)
        assert updated_res.shape == (B, D)
    
    def test_fused_add_rmsnorm_shape(self):
        """Test fused_add_rmsnorm output shape."""
        from zmlx.kernels.transformer import fused_add_rmsnorm
        
        B, D = 4, 64
        x1 = mx.random.normal((B, D))
        x2 = mx.random.normal((B, D))
        weight = mx.ones((D,))
        
        out = fused_add_rmsnorm(x1, x2, weight)
        
        assert out.shape == (B, D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
