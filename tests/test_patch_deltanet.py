"""Tests for the DeltaNet patch pattern.

Validates that:
  - Pattern matches GatedDeltaNet modules from qwen3_5
  - Patched decode output is token-identical to original
  - Prefill falls through to original forward pass
  - Pattern does not match unrelated modules
"""

import pytest

from zmlx._compat import import_mx

mx = import_mx()
if mx is None:
    pytest.skip("MLX not available", allow_module_level=True)

import mlx.nn as nn
import numpy as np

try:
    from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs
    from mlx_lm.models.cache import ArraysCache

    HAS_QWEN35 = True
except ImportError:
    HAS_QWEN35 = False

from zmlx.patch import patch, unpatch
from zmlx.patch._registry import get_pattern


def _make_tiny_config():
    """Create a minimal TextModelArgs for testing."""
    return TextModelArgs(
        model_type="qwen3_5",
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
    )


class _Container(nn.Module):
    """Wrapper so the traversal can find GatedDeltaNet as a child."""

    def __init__(self, m):
        super().__init__()
        self.linear_attn = m

    def __call__(self, x, **kw):
        return self.linear_attn(x, **kw)


def _patch_module(module):
    """Wrap in container, patch, return (container, patch_result)."""
    container = _Container(module)
    patch(container, patterns=["deltanet"])
    pr = container._zmlx_patch_result
    assert pr.patched_count == 1, f"Expected 1 patched module, got {pr.patched_count}"
    return container


@pytest.mark.skipif(not HAS_QWEN35, reason="mlx_lm qwen3_5 model not available")
class TestDeltaNetPatternMatch:
    """Test pattern detection."""

    def test_matches_gated_deltanet(self):
        """Pattern matches GatedDeltaNet modules."""
        cfg = _make_tiny_config()
        module = GatedDeltaNet(cfg)
        mx.eval(module.parameters())

        pat = get_pattern("deltanet")
        assert pat.matches(module, "linear_attn", parent=None)

    def test_no_match_on_linear(self):
        """Pattern does not match plain nn.Linear."""
        module = nn.Linear(64, 64)
        pat = get_pattern("deltanet")
        assert not pat.matches(module, "linear", parent=None)

    def test_no_match_on_mlp(self):
        """Pattern does not match a SwiGLU MLP."""

        class TinyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(64, 128, bias=False)
                self.up_proj = nn.Linear(64, 128, bias=False)
                self.down_proj = nn.Linear(128, 64, bias=False)

            def __call__(self, x):
                return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

        pat = get_pattern("deltanet")
        assert not pat.matches(TinyMLP(), "mlp", parent=None)


@pytest.mark.skipif(not HAS_QWEN35, reason="mlx_lm qwen3_5 model not available")
class TestDeltaNetPatchCorrectness:
    """Test that patched decode matches original."""

    def _make_module_and_cache(self):
        cfg = _make_tiny_config()
        module = GatedDeltaNet(cfg)
        mx.eval(module.parameters())
        cache = ArraysCache(size=2)
        return module, cache

    def test_decode_correctness(self):
        """Patched decode output matches original."""
        mx.random.seed(42)
        module, cache = self._make_module_and_cache()

        # Run original forward on a few tokens to build up state
        for _ in range(3):
            x = mx.random.normal((1, 1, 128))
            ref_out = module(x, cache=cache)
            mx.eval(ref_out, *[c for c in cache if c is not None])

        # Save cache state for comparison
        cache_orig = ArraysCache(size=2)
        cache_orig[0] = cache[0]
        cache_orig[1] = cache[1]
        mx.eval(*[c for c in cache_orig if c is not None])

        cache_patched = ArraysCache(size=2)
        cache_patched[0] = cache[0]
        cache_patched[1] = cache[1]
        mx.eval(*[c for c in cache_patched if c is not None])

        # Next token — original
        x_test = mx.random.normal((1, 1, 128))
        ref_out = module(x_test, cache=cache_orig)
        mx.eval(ref_out)

        # Patch (wraps in container) and run same token
        _patch_module(module)
        patched_out = module(x_test, cache=cache_patched)
        mx.eval(patched_out)

        np.testing.assert_allclose(
            np.array(patched_out),
            np.array(ref_out),
            atol=1e-3,
            rtol=1e-3,
        )

    def test_prefill_falls_through(self):
        """Prefill (S>1) uses original forward pass."""
        mx.random.seed(42)
        module, cache_orig = self._make_module_and_cache()
        cache_patched = ArraysCache(size=2)

        x = mx.random.normal((1, 4, 128))

        ref_out = module(x, cache=cache_orig)
        mx.eval(ref_out, *[c for c in cache_orig if c is not None])

        _patch_module(module)
        patched_out = module(x, cache=cache_patched)
        mx.eval(patched_out, *[c for c in cache_patched if c is not None])

        np.testing.assert_allclose(
            np.array(patched_out),
            np.array(ref_out),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_multi_token_decode_sequence(self):
        """Patched decode stays correct over multiple steps."""
        from mlx.utils import tree_flatten

        mx.random.seed(42)
        module_ref, cache_ref = self._make_module_and_cache()
        module_patched, cache_patched = self._make_module_and_cache()

        # Copy weights from ref to patched (need flattened key paths)
        flat_weights = tree_flatten(module_ref.parameters())
        module_patched.load_weights(flat_weights)
        mx.eval(module_patched.parameters())

        _patch_module(module_patched)

        # Run 10 decode steps
        for step in range(10):
            x = mx.random.normal((1, 1, 128))

            ref_out = module_ref(x, cache=cache_ref)
            mx.eval(ref_out, *[c for c in cache_ref if c is not None])

            patched_out = module_patched(x, cache=cache_patched)
            mx.eval(patched_out, *[c for c in cache_patched if c is not None])

            np.testing.assert_allclose(
                np.array(patched_out),
                np.array(ref_out),
                atol=1e-3,
                rtol=1e-3,
                err_msg=f"Mismatch at decode step {step}",
            )

    def test_unpatch_restores_original(self):
        """unpatch() restores original forward pass."""
        mx.random.seed(42)
        module, _ = self._make_module_and_cache()

        container = _patch_module(module)
        assert hasattr(module, "_zmlx_original_call")

        unpatch(container)

        # Should still work after unpatch
        cache = ArraysCache(size=2)
        x = mx.random.normal((1, 1, 128))
        out = module(x, cache=cache)
        mx.eval(out)
        assert out.shape == (1, 1, 128)


@pytest.mark.skipif(not HAS_QWEN35, reason="mlx_lm qwen3_5 model not available")
class TestDeltaNetPatchCounts:
    """Test that patch counts are correct."""

    def test_patch_count(self):
        """Patching a GatedDeltaNet reports 1 module patched."""
        cfg = _make_tiny_config()
        module = GatedDeltaNet(cfg)
        mx.eval(module.parameters())

        container = _Container(module)
        patch(container, patterns=["deltanet"])
        pr = container._zmlx_patch_result
        assert pr.patched_count == 1
        assert pr.pattern_counts.get("deltanet", 0) == 1
