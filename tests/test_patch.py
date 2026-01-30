"""Tests for zmlx.patch — model patching infrastructure."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class TinyRMSNormModel(nn.Module):
    """Minimal model with RMSNorm for testing."""

    def __init__(self, dims: int = 64):
        super().__init__()
        self.norm = nn.RMSNorm(dims)
        self.linear = nn.Linear(dims, dims)

    def __call__(self, x):
        return self.linear(self.norm(x))


class TinyLayerNormModel(nn.Module):
    """Minimal model with LayerNorm for testing."""

    def __init__(self, dims: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dims)
        self.linear = nn.Linear(dims, dims)

    def __call__(self, x):
        return self.linear(self.norm(x))


class TinySwiGLUMLP(nn.Module):
    """Minimal SwiGLU MLP (Llama-style)."""

    def __init__(self, dims: int = 64, hidden: int = 128):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)

    def __call__(self, x):
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TinyGeGLUMLP(nn.Module):
    """Minimal GeGLU MLP (Gemma-style)."""

    def __init__(self, dims: int = 64, hidden: int = 128):
        super().__init__()
        self.gate_proj = nn.Linear(dims, hidden, bias=False)
        self.up_proj = nn.Linear(dims, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dims, bias=False)
        self.hidden_act = "gelu"

    def __call__(self, x):
        gate = nn.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TinyTransformerBlock(nn.Module):
    """Minimal Llama-like transformer block."""

    def __init__(self, dims: int = 64, hidden: int = 128):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(dims)
        self.post_attention_layernorm = nn.RMSNorm(dims)
        self.self_attn = nn.Linear(dims, dims)  # Placeholder
        self.mlp = TinySwiGLUMLP(dims, hidden)

    def __call__(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class TinyManualAttention(nn.Module):
    """Minimal attention-like module with explicit softmax call."""

    def __init__(self, dims: int = 32):
        super().__init__()
        self.scale = dims**-0.5
        self.q_proj = nn.Linear(dims, dims, bias=False)
        self.k_proj = nn.Linear(dims, dims, bias=False)
        self.softmax = nn.Softmax()

    def __call__(self, x):
        return self.softmax(x)


class TinyModel(nn.Module):
    """Multi-layer model for end-to-end patching tests."""

    def __init__(self, dims: int = 64, hidden: int = 128, n_layers: int = 2):
        super().__init__()
        self.layers = [TinyTransformerBlock(dims, hidden) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(dims)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# --- Pattern listing ---


def test_list_patterns():
    from zmlx.patch import list_patterns

    names = list_patterns()
    assert "rmsnorm" in names
    assert "layernorm" in names
    assert "swiglu_mlp" in names
    assert "geglu_mlp" in names


# --- RMSNorm pattern ---


def test_rmsnorm_patch():
    from zmlx.patch import patch
    from zmlx.patch._modules import ZMLXRMSNorm

    model = TinyRMSNormModel(dims=64)
    x = mx.random.normal((2, 64))

    out_before = model(x)
    mx.eval(out_before)

    patch(model, patterns=["rmsnorm"])

    assert isinstance(model.norm, ZMLXRMSNorm)
    out_after = model(x)
    mx.eval(out_after)

    assert mx.allclose(out_before, out_after, atol=1e-4).item()


def test_rmsnorm_preserves_weights():
    from zmlx.patch import patch
    from zmlx.patch._modules import ZMLXRMSNorm

    model = TinyRMSNormModel(dims=32)
    original_weight = model.norm.weight
    mx.eval(original_weight)

    patch(model, patterns=["rmsnorm"])

    assert isinstance(model.norm, ZMLXRMSNorm)
    assert mx.array_equal(model.norm.weight, original_weight).item()


# --- LayerNorm pattern ---


def test_layernorm_patch():
    from zmlx.patch import patch
    from zmlx.patch._modules import ZMLXLayerNorm

    model = TinyLayerNormModel(dims=64)
    x = mx.random.normal((2, 64))

    out_before = model(x)
    mx.eval(out_before)

    patch(model, patterns=["layernorm"])

    assert isinstance(model.norm, ZMLXLayerNorm)
    out_after = model(x)
    mx.eval(out_after)

    assert mx.allclose(out_before, out_after, atol=1e-4).item()


# --- SwiGLU MLP pattern ---


def test_swiglu_mlp_patch():
    from zmlx.patch import patch

    model = TinySwiGLUMLP(dims=64, hidden=128)
    x = mx.random.normal((2, 64))

    out_before = model(x)
    mx.eval(out_before)

    patch(model, patterns=["swiglu_mlp"])

    out_after = model(x)
    mx.eval(out_after)

    assert mx.allclose(out_before, out_after, atol=1e-3).item()


# --- GeGLU MLP pattern ---


def test_geglu_mlp_patch():
    from zmlx.patch import patch

    model = TinyGeGLUMLP(dims=64, hidden=128)
    x = mx.random.normal((2, 64))

    out_before = model(x)
    mx.eval(out_before)

    patch(model, patterns=["geglu_mlp"])

    out_after = model(x)
    mx.eval(out_after)

    assert mx.allclose(out_before, out_after, atol=1e-3).item()


# --- Full model test ---


def test_patch_full_model():
    """Default patch() applies FUSED_ACTIVATIONS only — norms are NOT replaced."""
    from zmlx.patch import patch
    from zmlx.patch._modules import ZMLXRMSNorm

    model = TinyModel(dims=64, hidden=128, n_layers=2)
    x = mx.random.normal((2, 64))

    out_before = model(x)
    mx.eval(out_before)

    result_model = patch(model, verbose=False)

    assert result_model is model  # Modified in place
    assert hasattr(model, "_zmlx_patch_result")

    # Default is FUSED_ACTIVATIONS — RMSNorms should NOT be replaced
    assert not isinstance(model.norm, ZMLXRMSNorm)

    out_after = model(x)
    mx.eval(out_after)

    assert mx.allclose(out_before, out_after, atol=1e-3).item()


def test_patch_all_patterns():
    """Explicit ALL_PATTERNS applies all 7 patterns including norms."""
    from zmlx.patch import ALL_PATTERNS, patch
    from zmlx.patch._modules import ZMLXRMSNorm

    model = TinyModel(dims=64, hidden=128, n_layers=2)
    x = mx.random.normal((2, 64))

    out_before = model(x)
    mx.eval(out_before)

    patch(model, patterns=ALL_PATTERNS, verbose=False)

    # ALL_PATTERNS includes rmsnorm — RMSNorms SHOULD be replaced
    assert isinstance(model.norm, ZMLXRMSNorm)

    out_after = model(x)
    mx.eval(out_after)

    assert mx.allclose(out_before, out_after, atol=1e-3).item()


def test_patch_result_counts():
    from zmlx.patch import patch

    model = TinyModel(dims=64, hidden=128, n_layers=2)
    patch(model, patterns=["rmsnorm"])

    result = model._zmlx_patch_result
    # 2 layers * 2 norms each + 1 final norm = 5 RMSNorms
    assert result.pattern_counts.get("rmsnorm", 0) == 5


# --- Exclude patterns ---


def test_exclude_patterns():
    from zmlx.patch import patch
    from zmlx.patch._modules import ZMLXRMSNorm

    model = TinyModel(dims=64, hidden=128, n_layers=2)
    patch(model, exclude=["rmsnorm"])

    # RMSNorm should NOT be replaced
    assert not isinstance(model.norm, ZMLXRMSNorm)


# --- Selective patterns ---


def test_selective_patterns():
    from zmlx.patch import patch
    from zmlx.patch._modules import ZMLXRMSNorm

    model = TinyModel(dims=64, hidden=128, n_layers=2)
    patch(model, patterns=["rmsnorm"])

    assert isinstance(model.norm, ZMLXRMSNorm)
    result = model._zmlx_patch_result
    # Only rmsnorm should have been applied
    assert "swiglu_mlp" not in result.pattern_counts


def test_residual_norm_patch():
    from zmlx.patch import patch

    model = TinyTransformerBlock(dims=32, hidden=64)
    x = mx.random.normal((2, 32))

    out_before = model(x)
    mx.eval(out_before)

    patch(model, patterns=["residual_norm"])

    out_after = model(x)
    mx.eval(out_after)

    assert mx.allclose(out_before, out_after, atol=1e-3).item()


def test_softmax_patch():
    from zmlx.patch import patch

    model = TinyManualAttention(dims=32)
    x = mx.random.normal((4, 32))

    out_before = model(x)
    mx.eval(out_before)

    patch(model, patterns=["softmax"])

    out_after = model(x)
    mx.eval(out_after)

    assert mx.allclose(out_before, out_after, atol=1e-3).item()


# --- Gradient preservation ---


def test_rmsnorm_gradient():
    """Verify that gradients flow through patched RMSNorm."""
    from zmlx.patch import patch

    model = TinyRMSNormModel(dims=32)
    x = mx.random.normal((2, 32))

    # Gradient before patching
    loss_fn = nn.losses.mse_loss

    def forward_before(model, x):
        return loss_fn(model(x), mx.zeros_like(x))

    grad_fn = mx.grad(forward_before)
    # Just verify it doesn't crash
    _grads_before = grad_fn(model, x)
    mx.eval(_grads_before)

    patch(model, patterns=["rmsnorm"])

    def forward_after(model, x):
        return loss_fn(model(x), mx.zeros_like(x))

    grad_fn2 = mx.grad(forward_after)
    _grads_after = grad_fn2(model, x)
    mx.eval(_grads_after)
    # If we get here without error, gradients flow through the patched model


# --- Mode parameter ---


def test_patch_mode_inference():
    """mode='inference' applies FUSED_ACTIVATIONS (no norm replacement)."""
    from zmlx.patch import patch
    from zmlx.patch._modules import ZMLXRMSNorm

    model = TinyModel(dims=64, hidden=128, n_layers=2)
    x = mx.random.normal((2, 64))

    out_before = model(x)
    mx.eval(out_before)

    patch(model, mode="inference")

    # Inference mode should NOT replace norms
    assert not isinstance(model.norm, ZMLXRMSNorm)

    out_after = model(x)
    mx.eval(out_after)
    assert mx.allclose(out_before, out_after, atol=1e-3).item()


def test_patch_mode_training():
    """mode='training' applies TRAINING_RECOMMENDED (includes norm replacement)."""
    from zmlx.patch import patch
    from zmlx.patch._modules import ZMLXRMSNorm

    model = TinyModel(dims=64, hidden=128, n_layers=2)
    x = mx.random.normal((2, 64))

    out_before = model(x)
    mx.eval(out_before)

    patch(model, mode="training")

    # Training mode SHOULD replace norms
    assert isinstance(model.norm, ZMLXRMSNorm)

    out_after = model(x)
    mx.eval(out_after)
    assert mx.allclose(out_before, out_after, atol=1e-3).item()


def test_patch_mode_invalid():
    """Invalid mode raises ValueError."""
    import pytest

    from zmlx.patch import patch

    model = TinyModel(dims=64, hidden=128, n_layers=2)
    with pytest.raises(ValueError, match="Unknown mode"):
        patch(model, mode="bogus")


def test_patch_patterns_overrides_mode():
    """Explicit patterns= takes precedence over mode=."""
    from zmlx.patch import patch
    from zmlx.patch._modules import ZMLXRMSNorm

    model = TinyModel(dims=64, hidden=128, n_layers=2)

    # Even with mode="training", explicit patterns=["swiglu_mlp"] should win
    patch(model, mode="training", patterns=["swiglu_mlp"])

    # Norms should NOT be replaced because explicit patterns didn't include them
    assert not isinstance(model.norm, ZMLXRMSNorm)
