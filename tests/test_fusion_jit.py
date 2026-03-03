from __future__ import annotations

import mlx.core as mx
import pytest

from zmlx.fusion import build_aot_artifact, jit
from zmlx.fusion.graph import TensorMeta
from zmlx.fusion.passes import run_fusion
from zmlx.fusion.runtime import (
    compile_broadcast_mul_reduce_two_pass,
    compile_quantized_shared_input_swiglu,
)
from zmlx.fusion.trace import quantized_matmul, reduce_sum, symbolic_trace


def test_fusion_jit_trace_and_fuse_path():
    @jit
    def fused_silu_mul(x, y):
        return x * mx.sigmoid(x) * y

    initial = fused_silu_mul.fusion_stats()
    assert initial["traced"] is False
    assert initial["compiles"] == 0

    x = mx.random.normal((256,)).astype(mx.float32)
    y = mx.random.normal((256,)).astype(mx.float32)

    out = fused_silu_mul(x, y)
    mx.eval(out)

    stats = fused_silu_mul.fusion_stats()
    assert stats["traced"] is True
    assert stats["fusable"] is True
    assert stats["expression"] is not None
    assert stats["compiles"] == 1
    assert stats["fallbacks"] == 0


def test_fusion_jit_numerical_parity():
    @jit
    def fused_silu_mul(x, y):
        return x * mx.sigmoid(x) * y

    x = mx.random.normal((1024,)).astype(mx.float32)
    y = mx.random.normal((1024,)).astype(mx.float32)

    out = fused_silu_mul(x, y)
    expected = x * mx.sigmoid(x) * y
    mx.eval(out, expected)

    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()


def test_fusion_jit_cache_hit_skips_recompile():
    @jit
    def fused_silu_mul(x, y):
        return x * mx.sigmoid(x) * y

    x = mx.random.normal((128,)).astype(mx.float32)
    y = mx.random.normal((128,)).astype(mx.float32)

    out1 = fused_silu_mul(x, y)
    mx.eval(out1)
    after_first = fused_silu_mul.fusion_stats()
    assert after_first["compiles"] == 1
    assert after_first["cache_hits"] == 0

    out2 = fused_silu_mul(x, y)
    mx.eval(out2)
    after_second = fused_silu_mul.fusion_stats()

    assert after_second["compiles"] == 1
    assert after_second["cache_hits"] == 1
    assert after_second["fallbacks"] == 0


def test_fusion_jit_fallback_for_unsupported_pattern():
    @jit
    def unsupported(x, y):
        return mx.sin(x) * y

    x = mx.random.normal((64,)).astype(mx.float32)
    y = mx.random.normal((64,)).astype(mx.float32)

    out = unsupported(x, y)
    expected = mx.sin(x) * y
    mx.eval(out, expected)

    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()
    stats = unsupported.fusion_stats()
    assert stats["traced"] is True
    assert stats["fusable"] is False
    assert stats["unsupported_reason"] is not None
    assert stats["compiles"] == 0
    assert stats["fallbacks"] >= 1


def test_phase2_pattern_match_and_descriptor():
    trace = symbolic_trace(
        lambda x, w: reduce_sum(x * w, axis=-1),
        TensorMeta(shape=(4, 8), dtype="float32"),
        TensorMeta(shape=(4, 1), dtype="float32"),
        input_names=("x", "w"),
    )
    fused_ops = run_fusion(trace.graph).fused_ops
    fused = [op for op in fused_ops if op.output_id == trace.output_ids[0]]
    assert len(fused) == 1
    op = fused[0]
    assert op.kind == "broadcast_mul_reduce"
    assert op.attrs["reduce_op"] == "sum"
    assert op.attrs["axis"] == 1
    assert op.input_ids == trace.input_ids


def test_phase2_numerical_parity_vs_unfused_mlx():
    @jit
    def fused_mul_reduce(x, w):
        return mx.sum(x * w, axis=-1)

    x = mx.random.normal((6, 32)).astype(mx.float16)
    w = mx.random.normal((6, 1)).astype(mx.float16)

    out = fused_mul_reduce(x, w)
    expected = mx.sum(x * w, axis=-1)
    mx.eval(out, expected)

    assert out.shape == (6,)
    assert out.dtype == expected.dtype
    assert mx.allclose(out, expected, rtol=2e-2, atol=2e-2).item()
    stats = fused_mul_reduce.fusion_stats()
    assert stats["fusable"] is True
    assert "broadcast_mul_reduce" in str(stats["expression"])


def test_phase2_edge_axis_fallback_unsupported():
    @jit
    def unsupported_axis(x):
        return mx.sum(x, axis=0)

    x = mx.random.normal((5, 7)).astype(mx.float32)
    out = unsupported_axis(x)
    expected = mx.sum(x, axis=0)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    stats = unsupported_axis.fusion_stats()
    assert stats["fusable"] is False
    assert stats["fallbacks"] >= 1


def test_phase2_edge_singleton_dims_and_dtype():
    @jit
    def fused_mean_reduce(x, w):
        return mx.mean(x * w, axis=-1)

    x = mx.random.normal((9, 1)).astype(mx.float16)
    w = mx.ones((9, 1), dtype=mx.float16)
    out = fused_mean_reduce(x, w)
    expected = mx.mean(x * w, axis=-1)
    mx.eval(out, expected)

    assert out.shape == (9,)
    assert out.dtype == expected.dtype
    assert mx.allclose(out, expected, rtol=2e-3, atol=2e-3).item()


def test_phase2_guardrail_fallback_for_unsupported_broadcast():
    @jit
    def fused_mul_reduce(x, w):
        return mx.sum(x * w, axis=-1)

    x = mx.random.normal((2, 3, 16)).astype(mx.float32)
    # Broadcast shape currently unsupported by phase-2 runtime path.
    w = mx.random.normal((1, 3, 1)).astype(mx.float32)
    out = fused_mul_reduce(x, w)
    expected = mx.sum(x * w, axis=-1)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()
    assert fused_mul_reduce.fusion_stats()["fallbacks"] >= 1


def test_phase2_runtime_guardrail_rejects_unsupported_broadcast_shape():
    descriptor = {
        "kind": "broadcast_mul_reduce",
        "reduce_op": "sum",
        "axis": -1,
        "keepdims": False,
    }
    lhs = mx.random.normal((2, 3, 16)).astype(mx.float16)
    rhs = mx.random.normal((1, 3, 1)).astype(mx.float16)
    with pytest.raises(ValueError, match="Unsupported broadcast shape"):
        compile_broadcast_mul_reduce_two_pass(descriptor, [lhs, rhs])


def test_phase2_runtime_guardrail_rejects_non_last_axis_reduction():
    descriptor = {
        "kind": "broadcast_mul_reduce",
        "reduce_op": "sum",
        "axis": 0,
        "keepdims": False,
    }
    lhs = mx.random.normal((3, 16)).astype(mx.float16)
    rhs = mx.random.normal((3, 16)).astype(mx.float16)
    with pytest.raises(ValueError, match="Only last-axis reductions are supported"):
        compile_broadcast_mul_reduce_two_pass(descriptor, [lhs, rhs])


def test_phase3_pattern_match_and_descriptor():
    trace = symbolic_trace(
        lambda x, gw, gs, gb, uw, us, ub: (
            quantized_matmul(
                x,
                gw,
                scales=gs,
                biases=gb,
                group_size=64,
                bits=4,
                mode="affine",
                transpose=True,
            ).silu()
            * quantized_matmul(
                x,
                uw,
                scales=us,
                biases=ub,
                group_size=64,
                bits=4,
                mode="affine",
                transpose=True,
            )
        ),
        TensorMeta(shape=(1, 64), dtype="float16"),
        TensorMeta(shape=(32, 8), dtype="uint32"),
        TensorMeta(shape=(32, 1), dtype="float32"),
        TensorMeta(shape=(32, 1), dtype="float32"),
        TensorMeta(shape=(32, 8), dtype="uint32"),
        TensorMeta(shape=(32, 1), dtype="float32"),
        TensorMeta(shape=(32, 1), dtype="float32"),
        input_names=("x", "gw", "gs", "gb", "uw", "us", "ub"),
    )
    fused_ops = run_fusion(trace.graph).fused_ops
    fused = [op for op in fused_ops if op.output_id == trace.output_ids[0]]
    assert len(fused) == 1
    op = fused[0]
    assert op.kind == "quantized_shared_input_swiglu"
    assert op.attrs["mode"] == "affine"
    assert op.attrs["bits"] == 4
    assert op.attrs["transpose"] is True


def test_phase3_fused_quantized_pair_parity():
    @jit
    def fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, up_s, up_b):
        gate = mx.quantized_matmul(
            x,
            gate_w,
            scales=gate_s,
            biases=gate_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        up = mx.quantized_matmul(
            x,
            up_w,
            scales=up_s,
            biases=up_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        return (gate * mx.sigmoid(gate)) * up

    mx.random.seed(7)
    x = mx.random.normal((1, 64)).astype(mx.float16)
    gate_w_fp = mx.random.normal((32, 64)).astype(mx.float32)
    up_w_fp = mx.random.normal((32, 64)).astype(mx.float32)
    gate_w, gate_s, gate_b = mx.quantize(gate_w_fp, group_size=64, bits=4, mode="affine")
    up_w, up_s, up_b = mx.quantize(up_w_fp, group_size=64, bits=4, mode="affine")

    out = fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, up_s, up_b)
    expected_gate = mx.quantized_matmul(
        x,
        gate_w,
        scales=gate_s,
        biases=gate_b,
        group_size=64,
        bits=4,
        mode="affine",
        transpose=True,
    )
    expected = (expected_gate * mx.sigmoid(expected_gate)) * mx.quantized_matmul(
        x,
        up_w,
        scales=up_s,
        biases=up_b,
        group_size=64,
        bits=4,
        mode="affine",
        transpose=True,
    )
    mx.eval(out, expected)

    assert mx.allclose(out, expected, rtol=3e-3, atol=3e-3).item()
    stats = fused_quant_pair.fusion_stats()
    assert stats["fusable"] is True
    assert "quantized_shared_input_swiglu" in str(stats["expression"])


def test_phase3_guardrail_fallback_on_m_gt_1():
    @jit
    def fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, up_s, up_b):
        gate = mx.quantized_matmul(
            x,
            gate_w,
            scales=gate_s,
            biases=gate_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        up = mx.quantized_matmul(
            x,
            up_w,
            scales=up_s,
            biases=up_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        return (gate * mx.sigmoid(gate)) * up

    x = mx.random.normal((2, 64)).astype(mx.float16)
    gate_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    up_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    gate_w, gate_s, gate_b = mx.quantize(gate_w_fp, group_size=64, bits=4, mode="affine")
    up_w, up_s, up_b = mx.quantize(up_w_fp, group_size=64, bits=4, mode="affine")

    out = fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, up_s, up_b)
    expected_gate = mx.quantized_matmul(
        x,
        gate_w,
        scales=gate_s,
        biases=gate_b,
        group_size=64,
        bits=4,
        mode="affine",
        transpose=True,
    )
    expected = (expected_gate * mx.sigmoid(expected_gate)) * mx.quantized_matmul(
        x,
        up_w,
        scales=up_s,
        biases=up_b,
        group_size=64,
        bits=4,
        mode="affine",
        transpose=True,
    )
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()
    assert fused_quant_pair.fusion_stats()["fallbacks"] >= 1


def test_phase3_runtime_guardrail_rejects_unsupported_quantized_bits():
    descriptor = {
        "kind": "quantized_shared_input_swiglu",
        "mode": "affine",
        "transpose": True,
        "bits": 2,
        "group_size": 64,
    }
    x = mx.zeros((1, 64), dtype=mx.float16)
    gate_w = mx.zeros((16, 8), dtype=mx.uint32)
    gate_s = mx.ones((16, 1), dtype=mx.float32)
    gate_b = mx.zeros((16, 1), dtype=mx.float32)
    up_w = mx.zeros((16, 8), dtype=mx.uint32)
    up_s = mx.ones((16, 1), dtype=mx.float32)
    up_b = mx.zeros((16, 1), dtype=mx.float32)
    with pytest.raises(ValueError, match="Only 4-bit/8-bit quantized fusion is supported"):
        compile_quantized_shared_input_swiglu(
            descriptor,
            [x, gate_w, gate_s, gate_b, up_w, up_s, up_b],
        )


def test_phase3_runtime_guardrail_rejects_non_affine_mode():
    descriptor = {
        "kind": "quantized_shared_input_swiglu",
        "mode": "symmetric",
        "transpose": True,
        "bits": 4,
        "group_size": 64,
    }
    x = mx.zeros((1, 64), dtype=mx.float16)
    gate_w = mx.zeros((16, 8), dtype=mx.uint32)
    gate_s = mx.ones((16, 1), dtype=mx.float32)
    gate_b = mx.zeros((16, 1), dtype=mx.float32)
    up_w = mx.zeros((16, 8), dtype=mx.uint32)
    up_s = mx.ones((16, 1), dtype=mx.float32)
    up_b = mx.zeros((16, 1), dtype=mx.float32)
    with pytest.raises(
        ValueError, match="Only affine \\+ transpose=True quantized fusion is supported"
    ):
        compile_quantized_shared_input_swiglu(
            descriptor,
            [x, gate_w, gate_s, gate_b, up_w, up_s, up_b],
        )


def test_phase3_runtime_guardrail_rejects_transpose_false():
    descriptor = {
        "kind": "quantized_shared_input_swiglu",
        "mode": "affine",
        "transpose": False,
        "bits": 4,
        "group_size": 64,
    }
    x = mx.zeros((1, 64), dtype=mx.float16)
    gate_w = mx.zeros((16, 8), dtype=mx.uint32)
    gate_s = mx.ones((16, 1), dtype=mx.float32)
    gate_b = mx.zeros((16, 1), dtype=mx.float32)
    up_w = mx.zeros((16, 8), dtype=mx.uint32)
    up_s = mx.ones((16, 1), dtype=mx.float32)
    up_b = mx.zeros((16, 1), dtype=mx.float32)
    with pytest.raises(
        ValueError, match="Only affine \\+ transpose=True quantized fusion is supported"
    ):
        compile_quantized_shared_input_swiglu(
            descriptor,
            [x, gate_w, gate_s, gate_b, up_w, up_s, up_b],
        )


def test_phase3_runtime_guardrail_rejects_packed_k_mismatch():
    descriptor = {
        "kind": "quantized_shared_input_swiglu",
        "mode": "affine",
        "transpose": True,
        "bits": 4,
        "group_size": 64,
    }
    x = mx.zeros((1, 64), dtype=mx.float16)
    gate_w = mx.zeros((16, 7), dtype=mx.uint32)
    gate_s = mx.ones((16, 1), dtype=mx.float32)
    gate_b = mx.zeros((16, 1), dtype=mx.float32)
    up_w = mx.zeros((16, 7), dtype=mx.uint32)
    up_s = mx.ones((16, 1), dtype=mx.float32)
    up_b = mx.zeros((16, 1), dtype=mx.float32)
    with pytest.raises(ValueError, match="Packed weight shape does not match x K dimension"):
        compile_quantized_shared_input_swiglu(
            descriptor,
            [x, gate_w, gate_s, gate_b, up_w, up_s, up_b],
        )


def test_phase3_runtime_guardrail_rejects_unsupported_quantized_scales_shape():
    descriptor = {
        "kind": "quantized_shared_input_swiglu",
        "mode": "affine",
        "transpose": True,
        "bits": 4,
        "group_size": 64,
    }
    x = mx.zeros((1, 64), dtype=mx.float16)
    gate_w = mx.zeros((16, 8), dtype=mx.uint32)
    gate_s = mx.ones((16, 2), dtype=mx.float32)
    gate_b = mx.zeros((16, 2), dtype=mx.float32)
    up_w = mx.zeros((16, 8), dtype=mx.uint32)
    up_s = mx.ones((16, 2), dtype=mx.float32)
    up_b = mx.zeros((16, 2), dtype=mx.float32)
    with pytest.raises(ValueError, match="Unsupported scales shape for quantized fusion"):
        compile_quantized_shared_input_swiglu(
            descriptor,
            [x, gate_w, gate_s, gate_b, up_w, up_s, up_b],
        )


def test_phase3_runtime_guardrail_rejects_gate_up_scales_mismatch():
    descriptor = {
        "kind": "quantized_shared_input_swiglu",
        "mode": "affine",
        "transpose": True,
        "bits": 4,
        "group_size": 64,
    }
    x = mx.zeros((1, 64), dtype=mx.float16)
    gate_w = mx.zeros((16, 8), dtype=mx.uint32)
    gate_s = mx.ones((16, 1), dtype=mx.float32)
    gate_b = mx.zeros((16, 1), dtype=mx.float32)
    up_w = mx.zeros((16, 8), dtype=mx.uint32)
    up_s = mx.ones((16, 2), dtype=mx.float32)
    up_b = mx.zeros((16, 1), dtype=mx.float32)
    with pytest.raises(ValueError, match="gate_scales and up_scales must match"):
        compile_quantized_shared_input_swiglu(
            descriptor,
            [x, gate_w, gate_s, gate_b, up_w, up_s, up_b],
        )


def test_phase3_runtime_guardrail_rejects_gate_up_biases_mismatch():
    descriptor = {
        "kind": "quantized_shared_input_swiglu",
        "mode": "affine",
        "transpose": True,
        "bits": 4,
        "group_size": 64,
    }
    x = mx.zeros((1, 64), dtype=mx.float16)
    gate_w = mx.zeros((16, 8), dtype=mx.uint32)
    gate_s = mx.ones((16, 1), dtype=mx.float32)
    gate_b = mx.zeros((16, 1), dtype=mx.float32)
    up_w = mx.zeros((16, 8), dtype=mx.uint32)
    up_s = mx.ones((16, 1), dtype=mx.float32)
    up_b = mx.zeros((16, 2), dtype=mx.float32)
    with pytest.raises(ValueError, match="gate_biases and up_biases must match"):
        compile_quantized_shared_input_swiglu(
            descriptor,
            [x, gate_w, gate_s, gate_b, up_w, up_s, up_b],
        )


def test_phase3_guardrail_scales_mismatch_records_fallback_in_jit_integration_path():
    @jit
    def fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, up_s, up_b):
        gate = mx.quantized_matmul(
            x,
            gate_w,
            scales=gate_s,
            biases=gate_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        up = mx.quantized_matmul(
            x,
            up_w,
            scales=up_s,
            biases=up_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        return (gate * mx.sigmoid(gate)) * up

    x = mx.random.normal((1, 64)).astype(mx.float16)
    gate_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    up_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    gate_w, gate_s, gate_b = mx.quantize(gate_w_fp, group_size=64, bits=4, mode="affine")
    up_w, up_s, up_b = mx.quantize(up_w_fp, group_size=64, bits=4, mode="affine")
    bad_up_s = mx.concatenate([up_s, up_s], axis=-1)

    with pytest.raises(ValueError, match="Scales and biases should have the same shape"):
        fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, bad_up_s, up_b)

    stats = fused_quant_pair.fusion_stats()
    assert stats["fallbacks"] >= 1
    assert "gate_scales and up_scales must match" in str(stats["unsupported_reason"])


def test_phase3_guardrail_biases_mismatch_records_fallback_in_jit_integration_path():
    @jit
    def fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, up_s, up_b):
        gate = mx.quantized_matmul(
            x,
            gate_w,
            scales=gate_s,
            biases=gate_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        up = mx.quantized_matmul(
            x,
            up_w,
            scales=up_s,
            biases=up_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        return (gate * mx.sigmoid(gate)) * up

    x = mx.random.normal((1, 64)).astype(mx.float16)
    gate_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    up_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    gate_w, gate_s, gate_b = mx.quantize(gate_w_fp, group_size=64, bits=4, mode="affine")
    up_w, up_s, up_b = mx.quantize(up_w_fp, group_size=64, bits=4, mode="affine")
    bad_up_b = mx.concatenate([up_b, up_b], axis=-1)

    with pytest.raises(ValueError, match="Scales and biases should have the same shape"):
        fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, up_s, bad_up_b)

    stats = fused_quant_pair.fusion_stats()
    assert stats["fallbacks"] >= 1
    assert "gate_biases and up_biases must match" in str(stats["unsupported_reason"])


def test_phase3_moe_like_projection_flow():
    @jit
    def moe_like_combine(expert_outputs, gates):
        return mx.sum(expert_outputs * gates, axis=-1)

    expert_outputs = mx.random.normal((3, 24, 8)).astype(mx.float16)
    gates = mx.softmax(mx.random.normal((3, 24, 1)).astype(mx.float16), axis=-2)

    out = moe_like_combine(expert_outputs, gates)
    expected = mx.sum(expert_outputs * gates, axis=-1)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=2e-3, atol=2e-3).item()


def test_phase4_aot_export_scaffold_for_reduction():
    descriptor = {
        "kind": "broadcast_mul_reduce",
        "reduce_op": "sum",
        "axis": -1,
        "keepdims": False,
        "no_fma": True,
    }
    lhs = mx.random.normal((4, 16)).astype(mx.float16)
    rhs = mx.random.normal((4, 1)).astype(mx.float16)
    compiled = compile_broadcast_mul_reduce_two_pass(descriptor, [lhs, rhs])
    artifact = build_aot_artifact(descriptor=descriptor, compiled=compiled)
    assert "pass1" in artifact.sources
    assert "pass2" in artifact.sources
    assert artifact.descriptor["kind"] == "broadcast_mul_reduce"


def test_phase3_guardrail_x_rank_3_deterministic_fallback():
    @jit
    def fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, up_s, up_b):
        gate = mx.quantized_matmul(
            x,
            gate_w,
            scales=gate_s,
            biases=gate_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        up = mx.quantized_matmul(
            x,
            up_w,
            scales=up_s,
            biases=up_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        return (gate * mx.sigmoid(gate)) * up

    x_rank3 = mx.random.normal((1, 1, 64)).astype(mx.float16)
    gate_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    up_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    gate_w, gate_s, gate_b = mx.quantize(gate_w_fp, group_size=64, bits=4, mode="affine")
    up_w, up_s, up_b = mx.quantize(up_w_fp, group_size=64, bits=4, mode="affine")

    out = fused_quant_pair(x_rank3, gate_w, gate_s, gate_b, up_w, up_s, up_b)
    expected_gate = mx.quantized_matmul(
        x_rank3,
        gate_w,
        scales=gate_s,
        biases=gate_b,
        group_size=64,
        bits=4,
        mode="affine",
        transpose=True,
    )
    expected = (expected_gate * mx.sigmoid(expected_gate)) * mx.quantized_matmul(
        x_rank3,
        up_w,
        scales=up_s,
        biases=up_b,
        group_size=64,
        bits=4,
        mode="affine",
        transpose=True,
    )
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    stats = fused_quant_pair.fusion_stats()
    assert stats["fallbacks"] >= 1
    assert stats["unsupported_reason"] is not None


def test_phase3_guardrail_m_gt_1_deterministic_unsupported_reason():
    @jit
    def fused_quant_pair(x, gate_w, gate_s, gate_b, up_w, up_s, up_b):
        gate = mx.quantized_matmul(
            x,
            gate_w,
            scales=gate_s,
            biases=gate_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        up = mx.quantized_matmul(
            x,
            up_w,
            scales=up_s,
            biases=up_b,
            group_size=64,
            bits=4,
            mode="affine",
            transpose=True,
        )
        return (gate * mx.sigmoid(gate)) * up

    x_m2 = mx.random.normal((2, 64)).astype(mx.float16)
    gate_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    up_w_fp = mx.random.normal((16, 64)).astype(mx.float32)
    gate_w, gate_s, gate_b = mx.quantize(gate_w_fp, group_size=64, bits=4, mode="affine")
    up_w, up_s, up_b = mx.quantize(up_w_fp, group_size=64, bits=4, mode="affine")

    out = fused_quant_pair(x_m2, gate_w, gate_s, gate_b, up_w, up_s, up_b)
    expected_gate = mx.quantized_matmul(
        x_m2,
        gate_w,
        scales=gate_s,
        biases=gate_b,
        group_size=64,
        bits=4,
        mode="affine",
        transpose=True,
    )
    expected = (expected_gate * mx.sigmoid(expected_gate)) * mx.quantized_matmul(
        x_m2,
        up_w,
        scales=up_s,
        biases=up_b,
        group_size=64,
        bits=4,
        mode="affine",
        transpose=True,
    )
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    stats = fused_quant_pair.fusion_stats()
    assert stats["fallbacks"] >= 1
    assert stats["unsupported_reason"] is not None


def test_phase2_guardrail_broadcast_3d_deterministic_fallback():
    @jit
    def fused_mul_reduce(x, w):
        return mx.sum(x * w, axis=-1)

    x_3d = mx.random.normal((2, 3, 16)).astype(mx.float32)
    w_broadcast_3d = mx.random.normal((1, 3, 1)).astype(mx.float32)

    out = fused_mul_reduce(x_3d, w_broadcast_3d)
    expected = mx.sum(x_3d * w_broadcast_3d, axis=-1)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    stats = fused_mul_reduce.fusion_stats()
    assert stats["fallbacks"] >= 1
    assert stats["unsupported_reason"] is not None


def test_phase2_guardrail_broadcast_full_shape_fallback():
    @jit
    def fused_mul_reduce(x, w):
        return mx.sum(x * w, axis=-1)

    x = mx.random.normal((4, 16)).astype(mx.float32)
    w_full = mx.random.normal((4, 16)).astype(mx.float32)

    out = fused_mul_reduce(x, w_full)
    expected = mx.sum(x * w_full, axis=-1)
    mx.eval(out, expected)
    assert mx.allclose(out, expected, rtol=1e-5, atol=1e-5).item()

    stats = fused_mul_reduce.fusion_stats()
    assert stats["fusable"] is False
    assert stats["fallbacks"] >= 1
