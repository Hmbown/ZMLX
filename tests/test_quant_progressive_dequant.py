import mlx.core as mx

from zmlx.kernels import quant


def test_progressive_dequant_reconstructs():
    mx.random.seed(0)
    group_size = 64
    w = mx.random.normal((2, group_size)).astype(mx.float32)
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=4, mode="affine")

    full = mx.dequantize(q, scales, biases, group_size=group_size, bits=4, mode="affine")
    hi2 = quant.dequantize_affine_packed_hi2(q, scales, biases, group_size=group_size)
    lo2 = quant.dequantize_affine_packed_lo2_delta(q, scales, biases, group_size=group_size)
    recon = hi2 + lo2

    mx.eval(full, recon)
    assert mx.allclose(full, recon, atol=1e-4)
