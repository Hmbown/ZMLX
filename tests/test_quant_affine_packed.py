import mlx.core as mx

from zmlx.kernels import quant


def _gelu_tanh(x: mx.array) -> mx.array:
    k0 = 0.7978845608028654
    k1 = 0.044715
    return 0.5 * x * (1.0 + mx.tanh(k0 * (x + k1 * x * x * x)))


def _run_case(bits: int, group_size: int = 64) -> None:
    mx.random.seed(0)
    w = mx.random.normal((2, group_size)).astype(mx.float32)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits, mode="affine")

    deq_ref = mx.dequantize(q, scales, biases, group_size=group_size, bits=bits, mode="affine")
    deq = quant.dequantize_affine_packed(q, scales, biases, bits=bits, group_size=group_size)
    mx.eval(deq, deq_ref)
    assert mx.allclose(deq, deq_ref, atol=1e-4)

    silu_ref = deq_ref * mx.sigmoid(deq_ref)
    silu = quant.dequantize_affine_packed_silu(q, scales, biases, bits=bits, group_size=group_size)
    mx.eval(silu, silu_ref)
    assert mx.allclose(silu, silu_ref, atol=1e-4)

    gelu_ref = _gelu_tanh(deq_ref)
    gelu = quant.dequantize_affine_packed_gelu(q, scales, biases, bits=bits, group_size=group_size)
    mx.eval(gelu, gelu_ref)
    assert mx.allclose(gelu, gelu_ref, atol=1e-4)


def test_affine_packed_bits4():
    _run_case(bits=4)


def test_affine_packed_bits8():
    _run_case(bits=8)
