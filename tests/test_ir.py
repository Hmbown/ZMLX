"""Tests for zmlx.ir — Kernel IR byte-identity and serialization."""

from __future__ import annotations

import json

from zmlx.codegen import (
    elementwise_binary_source,
    elementwise_unary_source,
    rowwise_mapreduce_source,
    rowwise_parallel_reduction_source,
    rowwise_reduction_source,
)
from zmlx.ir import (
    ElementwiseBinaryIR,
    ElementwiseUnaryIR,
    PatternKind,
    RawIR,
    RowwiseMapReduceIR,
    RowwiseParallelReductionIR,
    RowwiseReductionIR,
    VJPSpec,
    ir_from_json,
    ir_to_json,
)

# ---------------------------------------------------------------------------
# Byte-identity tests: IR.to_source() == codegen function
# ---------------------------------------------------------------------------


def test_unary_byte_identity():
    ir = ElementwiseUnaryIR(expr="metal::exp(x)")
    expected = elementwise_unary_source(expr="metal::exp(x)")
    assert ir.to_source() == expected


def test_unary_custom_names():
    ir = ElementwiseUnaryIR(expr="x * x", inp="input", out="output")
    expected = elementwise_unary_source(expr="x * x", inp="input", out="output")
    assert ir.to_source() == expected


def test_binary_byte_identity():
    ir = ElementwiseBinaryIR(expr="a + b")
    expected = elementwise_binary_source(expr="a + b")
    assert ir.to_source() == expected


def test_binary_custom_names():
    ir = ElementwiseBinaryIR(expr="a * b", lhs="left", rhs="right", out="result")
    expected = elementwise_binary_source(expr="a * b", lhs="left", rhs="right", out="result")
    assert ir.to_source() == expected


def test_rowwise_reduction_byte_identity():
    ir = RowwiseReductionIR(
        reduce_expr="acc + v",
        init_expr="(T)0",
        finalize_expr="acc",
        d=128,
    )
    expected = rowwise_reduction_source(
        reduce_expr="acc + v",
        init_expr="(T)0",
        finalize_expr="acc",
        d=128,
    )
    assert ir.to_source() == expected


def test_parallel_reduction_byte_identity():
    ir = RowwiseParallelReductionIR(
        d=256,
        tg=128,
        init_expr="0.0f",
        update_expr="acc + x",
        reduce_op="a + b",
        finalize_expr="s",
    )
    expected = rowwise_parallel_reduction_source(
        d=256,
        tg=128,
        init_expr="0.0f",
        update_expr="acc + x",
        reduce_op="a + b",
        finalize_expr="s",
    )
    assert ir.to_source() == expected


def test_mapreduce_byte_identity():
    """Softmax-equivalent pattern."""
    ir = RowwiseMapReduceIR(
        d=512,
        tg=256,
        pass1_init="-INFINITY",
        pass1_update="metal::max(acc1, x)",
        pass1_reduce_op="metal::max(a, b)",
        pass2_init="0.0f",
        pass2_update="acc2 + metal::exp(x - s1)",
        pass2_reduce_op="a + b",
        write_expr="metal::exp(x - s1) / s2",
    )
    expected = rowwise_mapreduce_source(
        d=512,
        tg=256,
        pass1_init="-INFINITY",
        pass1_update="metal::max(acc1, x)",
        pass1_reduce_op="metal::max(a, b)",
        pass2_init="0.0f",
        pass2_update="acc2 + metal::exp(x - s1)",
        pass2_reduce_op="a + b",
        write_expr="metal::exp(x - s1) / s2",
    )
    assert ir.to_source() == expected


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_pattern_kinds():
    assert ElementwiseUnaryIR(expr="x").kind == PatternKind.ELEMENTWISE_UNARY
    assert ElementwiseBinaryIR(expr="a + b").kind == PatternKind.ELEMENTWISE_BINARY
    assert RowwiseReductionIR(
        reduce_expr="", init_expr="", finalize_expr="", d=1
    ).kind == PatternKind.ROWWISE_REDUCTION
    assert RowwiseParallelReductionIR(
        d=1, tg=1, init_expr="", update_expr="", reduce_op="", finalize_expr=""
    ).kind == PatternKind.ROWWISE_PARALLEL_REDUCTION
    assert RowwiseMapReduceIR(
        d=1, tg=1,
        pass1_init="", pass1_update="", pass1_reduce_op="",
        pass2_init="", pass2_update="", pass2_reduce_op="",
        write_expr="",
    ).kind == PatternKind.ROWWISE_MAPREDUCE
    assert RawIR(
        source="", input_names_list=(), output_names_list=()
    ).kind == PatternKind.RAW


def test_input_output_names():
    ir = ElementwiseUnaryIR(expr="x", inp="a", out="b")
    assert ir.input_names == ("a",)
    assert ir.output_names == ("b",)

    ir2 = ElementwiseBinaryIR(expr="a+b", lhs="l", rhs="r", out="o")
    assert ir2.input_names == ("l", "r")
    assert ir2.output_names == ("o",)


# ---------------------------------------------------------------------------
# VJPSpec
# ---------------------------------------------------------------------------


def test_vjpspec():
    vjp = VJPSpec(
        bwd_source="dinp[elem] = cotan[elem] * output[elem];",
        bwd_input_names=("output", "cotan"),
        bwd_output_names=("dinp",),
        use_output=True,
    )
    assert vjp.use_output is True
    assert len(vjp.bwd_input_names) == 2


def test_ir_with_vjp():
    vjp = VJPSpec(
        bwd_source="dinp[elem] = cotan[elem];",
        bwd_input_names=("cotan",),
        bwd_output_names=("dinp",),
    )
    ir = ElementwiseUnaryIR(expr="x", vjp=vjp)
    assert ir.vjp is not None
    assert ir.vjp.bwd_source == "dinp[elem] = cotan[elem];"


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


def test_json_roundtrip_unary():
    ir = ElementwiseUnaryIR(
        expr="metal::exp(x)",
        tags=frozenset({"activation"}),
    )
    s = ir_to_json(ir)
    parsed = json.loads(s)
    assert parsed["_kind"] == "elementwise_unary"
    assert parsed["expr"] == "metal::exp(x)"

    ir2 = ir_from_json(s)
    assert ir2.to_source() == ir.to_source()
    assert ir2.kind == ir.kind


def test_json_roundtrip_mapreduce():
    ir = RowwiseMapReduceIR(
        d=64,
        tg=32,
        pass1_init="-INFINITY",
        pass1_update="metal::max(acc1, x)",
        pass1_reduce_op="metal::max(a, b)",
        pass2_init="0.0f",
        pass2_update="acc2 + metal::exp(x - s1)",
        pass2_reduce_op="a + b",
        write_expr="metal::exp(x - s1) / s2",
    )
    s = ir_to_json(ir)
    ir2 = ir_from_json(s)
    assert ir2.to_source() == ir.to_source()


def test_json_roundtrip_with_vjp():
    vjp = VJPSpec(
        bwd_source="dinp[elem] = cotan[elem];",
        bwd_input_names=("cotan",),
        bwd_output_names=("dinp",),
        use_output=False,
        prelude="// backward prelude",
    )
    ir = ElementwiseUnaryIR(expr="x * x", vjp=vjp)
    s = ir_to_json(ir)
    ir2 = ir_from_json(s)
    assert ir2.vjp is not None
    assert ir2.vjp.bwd_source == vjp.bwd_source
    assert ir2.vjp.use_output is False
    assert ir2.vjp.prelude == "// backward prelude"


# ---------------------------------------------------------------------------
# Frozen / hashable
# ---------------------------------------------------------------------------


def test_frozen_hashable():
    ir1 = ElementwiseUnaryIR(expr="x")
    ir2 = ElementwiseUnaryIR(expr="x")
    assert hash(ir1) == hash(ir2)
    assert ir1 == ir2

    s = {ir1, ir2}
    assert len(s) == 1
