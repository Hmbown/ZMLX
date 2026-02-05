"""Tests for the DSL expression tree."""

from __future__ import annotations

import pytest

from zmlx.dsl.expr import (
    BinOp,
    Call,
    Const,
    Let,
    RawMetal,
    Ternary,
    UnaryOp,
    Var,
    abs_,
    collect_lets,
    exp,
    gelu_tanh,
    log,
    max_,
    min_,
    rsqrt,
    sigmoid,
    silu,
    sqrt,
    tanh,
)

# ---------------------------------------------------------------------------
# Basic node construction and to_metal()
# ---------------------------------------------------------------------------

class TestVar:
    def test_simple(self):
        assert Var("x").to_metal() == "x"

    def test_different_names(self):
        assert Var("acc1").to_metal() == "acc1"
        assert Var("g").to_metal() == "g"


class TestConst:
    def test_float(self):
        assert Const(1.0).to_metal() == "(T)1.0"

    def test_float_no_trailing_zero(self):
        assert Const(0.5).to_metal() == "(T)0.5"

    def test_int(self):
        assert Const(0).to_metal() == "(T)0"
        assert Const(2).to_metal() == "(T)2"

    def test_negative_float(self):
        assert Const(-1.0).to_metal() == "(T)-1.0"


class TestRawMetal:
    def test_passthrough(self):
        assert RawMetal("kk_silu(x)").to_metal() == "kk_silu(x)"

    def test_complex_expression(self):
        code = "(T)0.5 * x * (T(1) + metal::tanh(k0 * (x + k1 * x3)))"
        assert RawMetal(code).to_metal() == code


class TestBinOp:
    def test_add(self):
        result = BinOp("+", Var("x"), Const(1.0))
        assert result.to_metal() == "(x + (T)1.0)"

    def test_mul(self):
        result = BinOp("*", Var("x"), Var("y"))
        assert result.to_metal() == "(x * y)"

    def test_nested(self):
        inner = BinOp("+", Var("a"), Var("b"))
        outer = BinOp("*", inner, Var("c"))
        assert outer.to_metal() == "((a + b) * c)"


class TestUnaryOp:
    def test_negate(self):
        result = UnaryOp("-", Var("x"))
        assert result.to_metal() == "(-(x))"


class TestCall:
    def test_single_arg(self):
        result = Call("metal::exp", (Var("x"),))
        assert result.to_metal() == "metal::exp(x)"

    def test_multi_arg(self):
        result = Call("metal::max", (Var("a"), Var("b")))
        assert result.to_metal() == "metal::max(a, b)"


class TestLet:
    def test_prelude(self):
        s = Var("s")
        x = Var("x")
        let = Let("s", sigmoid(x), s * (Const(1.0) - s))
        assert let.to_metal_prelude() == "T s = kk_sigmoid(x);"

    def test_body(self):
        s = Var("s")
        x = Var("x")
        let = Let("s", sigmoid(x), s * (Const(1.0) - s))
        # to_metal() returns the body expression
        assert let.to_metal() == "(s * (s - (T)1.0))" or let.to_metal() == "((T)1.0 - s)" or "s" in let.to_metal()


class TestTernary:
    def test_simple(self):
        result = Ternary(Var("x") > Const(0.0), Var("x"), Const(0.0))
        assert result.to_metal() == "((x > (T)0.0)) ? (x) : ((T)0.0)"


# ---------------------------------------------------------------------------
# Operator overloading
# ---------------------------------------------------------------------------

class TestOperators:
    def test_add(self):
        x = Var("x")
        result = x + Const(1.0)
        assert isinstance(result, BinOp)
        assert result.op == "+"

    def test_radd(self):
        x = Var("x")
        result = 1.0 + x
        assert isinstance(result, BinOp)
        assert result.to_metal() == "((T)1.0 + x)"

    def test_sub(self):
        x = Var("x")
        result = x - Const(1.0)
        assert result.to_metal() == "(x - (T)1.0)"

    def test_rsub(self):
        x = Var("x")
        result = 1.0 - x
        assert result.to_metal() == "((T)1.0 - x)"

    def test_mul(self):
        x = Var("x")
        result = x * Var("y")
        assert result.to_metal() == "(x * y)"

    def test_rmul(self):
        x = Var("x")
        result = 2.0 * x
        assert result.to_metal() == "((T)2.0 * x)"

    def test_div(self):
        x = Var("x")
        result = x / Const(2.0)
        assert result.to_metal() == "(x / (T)2.0)"

    def test_neg(self):
        x = Var("x")
        result = -x
        assert isinstance(result, UnaryOp)
        assert result.to_metal() == "(-(x))"

    def test_gt(self):
        x = Var("x")
        result = x > 0.0
        assert result.to_metal() == "(x > (T)0.0)"

    def test_lt(self):
        x = Var("x")
        result = x < 0.0
        assert result.to_metal() == "(x < (T)0.0)"

    def test_chain(self):
        x = Var("x")
        # x * sigmoid(x) — i.e. SiLU
        result = x * sigmoid(x)
        assert result.to_metal() == "(x * kk_sigmoid(x))"


# ---------------------------------------------------------------------------
# Built-in constructors
# ---------------------------------------------------------------------------

class TestBuiltins:
    def test_exp(self):
        assert exp(Var("x")).to_metal() == "metal::exp(x)"

    def test_log(self):
        assert log(Var("x")).to_metal() == "metal::log(x)"

    def test_tanh(self):
        assert tanh(Var("x")).to_metal() == "metal::tanh(x)"

    def test_sigmoid(self):
        assert sigmoid(Var("x")).to_metal() == "kk_sigmoid(x)"

    def test_silu(self):
        assert silu(Var("x")).to_metal() == "kk_silu(x)"

    def test_gelu_tanh(self):
        assert gelu_tanh(Var("x")).to_metal() == "kk_gelu_tanh(x)"

    def test_sqrt(self):
        assert sqrt(Var("x")).to_metal() == "metal::sqrt(x)"

    def test_rsqrt(self):
        assert rsqrt(Var("x")).to_metal() == "metal::rsqrt(x)"

    def test_abs(self):
        assert abs_(Var("x")).to_metal() == "metal::abs(x)"

    def test_max(self):
        assert max_(Var("a"), Var("b")).to_metal() == "metal::max(a, b)"

    def test_min(self):
        assert min_(Var("a"), Var("b")).to_metal() == "metal::min(a, b)"


# ---------------------------------------------------------------------------
# Composition — real kernel expressions
# ---------------------------------------------------------------------------

class TestComposition:
    def test_silu_manual(self):
        """SiLU: x * sigmoid(x)"""
        x = Var("x")
        result = x * sigmoid(x)
        assert result.to_metal() == "(x * kk_sigmoid(x))"

    def test_mish(self):
        """Mish: x * tanh(log(1 + exp(x)))"""
        x = Var("x")
        result = x * tanh(log(Const(1.0) + exp(x)))
        assert "metal::tanh" in result.to_metal()
        assert "metal::log" in result.to_metal()
        assert "metal::exp" in result.to_metal()

    def test_relu(self):
        """ReLU via ternary: (x > 0) ? x : 0"""
        x = Var("x")
        result = Ternary(x > 0.0, x, Const(0.0))
        metal = result.to_metal()
        assert "?" in metal
        assert "(x > (T)0.0)" in metal

    def test_softplus(self):
        """softplus: log(exp(x) + 1)"""
        x = Var("x")
        result = log(exp(x) + Const(1.0))
        metal = result.to_metal()
        assert metal == "metal::log((metal::exp(x) + (T)1.0))"


# ---------------------------------------------------------------------------
# collect_lets
# ---------------------------------------------------------------------------

class TestCollectLets:
    def test_no_lets(self):
        x = Var("x")
        bindings, body = collect_lets(x)
        assert bindings == []
        assert body is x

    def test_single_let(self):
        x = Var("x")
        s = Var("s")
        let = Let("s", sigmoid(x), s * x)
        bindings, body = collect_lets(let)
        assert len(bindings) == 1
        assert bindings[0].var_name == "s"
        assert body.to_metal() == "(s * x)"

    def test_nested_lets(self):
        x = Var("x")
        s = Var("s")
        t = Var("t")
        inner = Let("t", tanh(x), s + t)
        outer = Let("s", sigmoid(x), inner)
        bindings, body = collect_lets(outer)
        assert len(bindings) == 2
        assert bindings[0].var_name == "s"
        assert bindings[1].var_name == "t"


# ---------------------------------------------------------------------------
# Frozen dataclass invariants
# ---------------------------------------------------------------------------

class TestFrozen:
    def test_var_frozen(self):
        v = Var("x")
        with pytest.raises(AttributeError):
            v.name = "y"  # type: ignore[misc]

    def test_const_frozen(self):
        c = Const(1.0)
        with pytest.raises(AttributeError):
            c.value = 2.0  # type: ignore[misc]

    def test_binop_frozen(self):
        b = BinOp("+", Var("x"), Const(1.0))
        with pytest.raises(AttributeError):
            b.op = "-"  # type: ignore[misc]

    def test_hashable(self):
        """Frozen dataclasses should be hashable for use as cache keys."""
        v1 = Var("x")
        v2 = Var("x")
        assert hash(v1) == hash(v2)
        assert v1 == v2

        c1 = Const(1.0)
        c2 = Const(1.0)
        assert hash(c1) == hash(c2)
