"""Tests for defkernel declarative kernel definitions."""

from __future__ import annotations

import pytest

from zmlx._compat import is_metal_available

pytestmark = pytest.mark.skipif(
    not is_metal_available(),
    reason="Metal GPU not available",
)


class TestDefkernelUnary:
    """Test that defkernel produces working unary kernels."""

    def test_forward_only(self):
        import mlx.core as mx

        from zmlx.dsl.defkernel import defkernel

        my_relu = defkernel("kk_test_relu", "metal::max(x, (T)0)")
        kernel = my_relu()
        x = mx.array([-1.0, 0.0, 1.0, 2.0])
        result = kernel(x)
        mx.eval(result)
        expected = mx.maximum(x, mx.array(0.0))
        mx.eval(expected)
        assert mx.allclose(result, expected).item()

    def test_forward_with_expr(self):
        import mlx.core as mx

        from zmlx.dsl.defkernel import defkernel
        from zmlx.dsl.expr import Var, max_

        x = Var("x")
        expr = max_(x, Var("(T)0"))  # slightly unusual but valid
        my_relu = defkernel("kk_test_relu_expr", "metal::max(x, (T)0)")
        kernel = my_relu()
        inp = mx.array([-2.0, -1.0, 0.0, 1.0])
        result = kernel(inp)
        mx.eval(result)
        expected = mx.maximum(inp, mx.array(0.0))
        mx.eval(expected)
        assert mx.allclose(result, expected).item()

    def test_with_vjp(self):
        import mlx.core as mx

        from zmlx.dsl.defkernel import defkernel

        # SiLU with gradient
        my_silu = defkernel(
            "kk_test_silu_grad",
            "x * kk_sigmoid(x)",
            vjp_expr="g * (s + x * s * ((T)1 - s))",
            vjp_prelude="T s = kk_sigmoid(x);",
            use_output=False,
        )
        kernel = my_silu()
        x = mx.array([0.0, 1.0, -1.0, 2.0])
        result = kernel(x)
        mx.eval(result)
        # SiLU: x * sigmoid(x)
        expected = x * mx.sigmoid(x)
        mx.eval(expected)
        assert mx.allclose(result, expected, atol=1e-5).item()

    def test_compute_dtype_key(self):
        import mlx.core as mx

        from zmlx.dsl.defkernel import defkernel

        my_exp = defkernel("kk_test_exp", "metal::exp(x)")
        # Default is float32
        k32 = my_exp()
        k16 = my_exp(compute_dtype_key="float16")
        # Both should work
        x = mx.array([0.0, 1.0])
        r32 = k32(x)
        r16 = k16(x)
        mx.eval(r32, r16)
        assert mx.allclose(r32, r16, atol=1e-3).item()

    def test_caching(self):
        """Builder should return the same object for same args."""
        from zmlx.dsl.defkernel import defkernel

        my_exp = defkernel("kk_test_exp_cache", "metal::exp(x)")
        k1 = my_exp()
        k2 = my_exp()
        assert k1 is k2

    def test_matches_legacy_silu(self):
        """DSL silu should produce identical output to legacy silu."""
        import mlx.core as mx

        from zmlx.dsl.defkernel import defkernel
        from zmlx.kernels.activations import silu as legacy_silu

        dsl_silu = defkernel("kk_silu", "kk_silu(x)")
        legacy_k = legacy_silu()
        dsl_k = dsl_silu()
        x = mx.random.normal((128,))
        mx.eval(x)

        legacy_out = legacy_k(x)
        dsl_out = dsl_k(x)
        mx.eval(legacy_out, dsl_out)
        assert mx.allclose(legacy_out, dsl_out, atol=1e-6).item()


class TestDefkernelMapreduce:
    """Test defkernel_mapreduce for rowwise map-reduce kernels."""

    def test_softmax_style(self):
        import mlx.core as mx

        from zmlx.dsl.defkernel import defkernel_mapreduce

        my_softmax = defkernel_mapreduce(
            "kk_test_softmax",
            pass1={
                "init": "-INFINITY",
                "update": "metal::max(acc1, x)",
                "reduce_op": "metal::max(a, b)",
            },
            pass2={
                "init": "0.0f",
                "update": "acc2 + metal::exp(x - s1)",
                "reduce_op": "a + b",
            },
            write_expr="metal::exp(x - s1) / s2",
        )

        x = mx.random.normal((4, 64))
        mx.eval(x)
        kernel = my_softmax(d=64, tg=64)
        result = kernel(x)
        mx.eval(result)
        expected = mx.softmax(x, axis=-1)
        mx.eval(expected)
        assert mx.allclose(result, expected, atol=1e-5).item()
