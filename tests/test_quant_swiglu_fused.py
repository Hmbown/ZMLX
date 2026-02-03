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


def test_fused_quantized_swiglu_gemv_matches():
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
    fused = quant.fused_quantized_swiglu_gemv(
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
    mx.eval(baseline, fused)
    assert mx.allclose(baseline, fused, atol=1e-4)
