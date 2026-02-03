import mlx.core as mx

from zmlx.kernels import quant


def _baseline_swiglu(
    x: mx.array,
    gate_w: mx.array,
    gate_scales: mx.array,
    gate_biases: mx.array,
    up_w: mx.array,
    up_scales: mx.array,
    up_biases: mx.array,
    *,
    group_size: int,
    bits: int,
) -> mx.array:
    gate = mx.quantized_matmul(
        x,
        gate_w,
        scales=gate_scales,
        biases=gate_biases,
        group_size=group_size,
        bits=bits,
        mode="affine",
        transpose=True,
    )
    up = mx.quantized_matmul(
        x,
        up_w,
        scales=up_scales,
        biases=up_biases,
        group_size=group_size,
        bits=bits,
        mode="affine",
        transpose=True,
    )
    return gate * mx.sigmoid(gate) * up


def test_progressive_swiglu_matches_when_refined():
    mx.random.seed(0)
    bits = 4
    group_size = 64
    B = 1
    K = 64
    N = 32

    x = mx.random.normal((B, K)).astype(mx.float16)
    w_gate = mx.random.normal((N, K)).astype(mx.float32)
    w_up = mx.random.normal((N, K)).astype(mx.float32)

    gate_w, gate_scales, gate_biases = mx.quantize(w_gate, group_size=group_size, bits=bits, mode="affine")
    up_w, up_scales, up_biases = mx.quantize(w_up, group_size=group_size, bits=bits, mode="affine")

    baseline = _baseline_swiglu(
        x,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
    )
    progressive = quant.fused_quantized_swiglu_gemv_progressive(
        x,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
        epsilon=0.0,
    )
    progressive_pg = quant.fused_quantized_swiglu_gemv_progressive(
        x,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
        epsilon=0.0,
        per_group=True,
    )
    progressive_tg = quant.fused_quantized_swiglu_gemv_progressive(
        x,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
        epsilon=0.0,
        threadgroup=32,
    )
    mx.eval(baseline, progressive)
    assert mx.allclose(baseline, progressive, atol=1e-4)
    mx.eval(baseline, progressive_pg)
    assert mx.allclose(baseline, progressive_pg, atol=1e-4)
    mx.eval(baseline, progressive_tg)
    assert mx.allclose(baseline, progressive_tg, atol=1e-4)


def test_progressive_swiglu_tg_skip_refine_matches_non_tg():
    mx.random.seed(1)
    bits = 4
    group_size = 64
    B = 1
    K = 128
    N = 32

    x = mx.random.normal((B, K)).astype(mx.float16)
    w_gate = mx.random.normal((N, K)).astype(mx.float32)
    w_up = mx.random.normal((N, K)).astype(mx.float32)

    gate_w, gate_scales, gate_biases = mx.quantize(w_gate, group_size=group_size, bits=bits, mode="affine")
    up_w, up_scales, up_biases = mx.quantize(w_up, group_size=group_size, bits=bits, mode="affine")

    # Use a very large epsilon to ensure refinement is skipped.
    eps = 1.0e9
    non_tg = quant.fused_quantized_swiglu_gemv_progressive(
        x,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
        epsilon=eps,
    )
    tg = quant.fused_quantized_swiglu_gemv_progressive(
        x,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
        group_size=group_size,
        bits=bits,
        epsilon=eps,
        threadgroup=32,
    )
    mx.eval(non_tg, tg)
    assert mx.allclose(non_tg, tg, atol=1e-4)
