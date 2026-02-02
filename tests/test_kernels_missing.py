from __future__ import annotations

import math
import platform

import mlx.core as mx
import numpy as np
import pytest

from zmlx.kernels import bits, fused_moe, image, indexing, optimizers, quant

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Apple Silicon only",
)


def _resize_bilinear_ref(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    n, in_h, in_w, c = x.shape
    out = np.zeros((n, out_h, out_w, c), dtype=np.float32)
    scale_h = in_h / out_h
    scale_w = in_w / out_w
    for b in range(n):
        for y in range(out_h):
            for x_out in range(out_w):
                in_y = (y + 0.5) * scale_h - 0.5
                in_x = (x_out + 0.5) * scale_w - 0.5
                y0 = math.floor(in_y)
                x0 = math.floor(in_x)
                y1 = y0 + 1
                x1 = x0 + 1
                wy = in_y - y0
                wx = in_x - x0
                y0 = min(max(y0, 0), in_h - 1)
                y1 = min(max(y1, 0), in_h - 1)
                x0 = min(max(x0, 0), in_w - 1)
                x1 = min(max(x1, 0), in_w - 1)
                v00 = x[b, y0, x0]
                v01 = x[b, y0, x1]
                v10 = x[b, y1, x0]
                v11 = x[b, y1, x1]
                out[b, y, x_out] = (
                    v00 * (1.0 - wy) * (1.0 - wx)
                    + v01 * (1.0 - wy) * wx
                    + v10 * wy * (1.0 - wx)
                    + v11 * wy * wx
                )
    return out


def _depthwise_conv3x3_ref(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    n, h, w_in, c = x.shape
    out = np.zeros_like(x, dtype=np.float32)
    for b in range(n):
        for y in range(h):
            for x_out in range(w_in):
                for ch in range(c):
                    acc = 0.0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            iy = y + dy
                            ix = x_out + dx
                            if 0 <= iy < h and 0 <= ix < w_in:
                                acc += x[b, iy, ix, ch] * w[dy + 1, dx + 1, ch]
                    out[b, y, x_out, ch] = acc
    return out


def _fp8_ref(bits: np.ndarray, scale: float, variant: str) -> np.ndarray:
    out = np.empty(bits.shape, dtype=np.float32)
    for i, b in enumerate(bits):
        sign = -1.0 if (b & 0x80) else 1.0
        if variant == "e4m3":
            exp_bits = (b >> 3) & 0x0F
            mant_bits = b & 0x07
            if exp_bits == 0:
                val = sign * (2.0 ** -6.0) * (mant_bits / 8.0)
            elif exp_bits == 15 and mant_bits == 7:
                val = np.nan
            else:
                val = sign * (2.0 ** (exp_bits - 7.0)) * (1.0 + mant_bits / 8.0)
        else:
            exp_bits = (b >> 2) & 0x1F
            mant_bits = b & 0x03
            if exp_bits == 0:
                val = sign * (2.0 ** -14.0) * (mant_bits / 4.0)
            elif exp_bits == 31:
                val = sign * np.inf if mant_bits == 0 else np.nan
            else:
                val = sign * (2.0 ** (exp_bits - 15.0)) * (1.0 + mant_bits / 4.0)
        out[i] = val * scale
    return out


def test_pack_bits_matches_reference() -> None:
    x = mx.array(
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        dtype=mx.uint8,
    )
    result = bits.pack_bits(x)
    mx.eval(result)
    result_np = np.array(result.tolist(), dtype=np.uint8)
    x_np = np.array(x.tolist(), dtype=np.uint8)
    expected = np.zeros((x_np.size // 8,), dtype=np.uint8)
    for byte in range(expected.size):
        res = 0
        base = byte * 8
        for i in range(8):
            if x_np[base + i]:
                res |= 1 << i
        expected[byte] = res
    np.testing.assert_array_equal(result_np, expected)


def test_unpack_bits_matches_reference() -> None:
    packed = mx.array([0b10110010, 0b01011100], dtype=mx.uint8)
    result = bits.unpack_bits(packed)
    mx.eval(result)
    result_np = np.array(result.tolist(), dtype=np.uint8)
    packed_np = np.array(packed.tolist(), dtype=np.uint8)
    expected = np.zeros((packed_np.size * 8,), dtype=np.uint8)
    for gid in range(expected.size):
        byte_idx = gid // 8
        bit_idx = gid % 8
        val = packed_np[byte_idx]
        expected[gid] = 1 if (val & (1 << bit_idx)) else 0
    np.testing.assert_array_equal(result_np, expected)


def test_resize_bilinear_matches_reference() -> None:
    x = mx.array(
        [
            [[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]],
        ],
        dtype=mx.float32,
    )
    result = image.resize_bilinear(x, (3, 4))
    mx.eval(result)
    expected = _resize_bilinear_ref(np.array(x.tolist(), dtype=np.float32), 3, 4)
    np.testing.assert_allclose(
        np.array(result.tolist(), dtype=np.float32),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_depthwise_conv3x3_matches_reference() -> None:
    mx.random.seed(0)
    x = mx.random.normal((1, 4, 4, 2)).astype(mx.float32)
    w = mx.random.normal((3, 3, 2)).astype(mx.float32)
    result = image.depthwise_conv_3x3(x, w)
    mx.eval(result)
    expected = _depthwise_conv3x3_ref(
        np.array(x.tolist(), dtype=np.float32),
        np.array(w.tolist(), dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.array(result.tolist(), dtype=np.float32),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_fused_gather_add_matches_reference() -> None:
    mx.random.seed(1)
    src = mx.random.normal((4, 3)).astype(mx.float32)
    indices = mx.array([2, 0], dtype=mx.uint32)
    other = mx.random.normal((2, 3)).astype(mx.float32)
    result = indexing.fused_gather_add(src, indices, other)
    ref = src[indices] + other
    mx.eval(result, ref)
    assert mx.allclose(result, ref, atol=1e-6, rtol=1e-6).item()


def test_fused_scatter_add_matches_reference() -> None:
    mx.random.seed(2)
    indices = mx.array([0, 2, 4], dtype=mx.uint32)
    updates = mx.random.normal((3, 3)).astype(mx.float32)
    result = indexing.fused_scatter_add(indices, updates, (5, 3))
    mx.eval(result)
    indices_np = np.array(indices.tolist(), dtype=np.int64)
    updates_np = np.array(updates.tolist(), dtype=np.float32)
    expected = np.zeros((5, 3), dtype=np.float32)
    for i, idx in enumerate(indices_np):
        expected[idx] += updates_np[i]
    np.testing.assert_allclose(
        np.array(result.tolist(), dtype=np.float32),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.xfail(reason="Metal template instantiation issue on MLX 0.30.x", strict=False)
def test_adamw_step_matches_reference() -> None:
    p = mx.array([1.0, -2.0, 3.0, -4.0], dtype=mx.float32)
    g = mx.array([0.1, -0.2, 0.3, -0.4], dtype=mx.float32)
    m = mx.zeros_like(p)
    v = mx.zeros_like(p)
    lr = mx.array([0.1], dtype=mx.float32)
    step = mx.array([1.0], dtype=mx.float32)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    wd = 0.01

    new_p, new_m, new_v = optimizers.adamw_step(
        p,
        g,
        m,
        v,
        lr,
        step,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        wd=wd,
    )

    step_val = float(step.item())
    m_ref = beta1 * m + (1.0 - beta1) * g
    v_ref = beta2 * v + (1.0 - beta2) * g * g
    m_hat = m_ref / (1.0 - beta1 ** step_val)
    v_hat = v_ref / (1.0 - beta2 ** step_val)
    p_ref = p - lr * (m_hat / (mx.sqrt(v_hat) + eps) + wd * p)

    mx.eval(new_p, new_m, new_v, p_ref, m_ref, v_ref)
    assert mx.allclose(new_p, p_ref, atol=1e-6, rtol=1e-6).item()
    assert mx.allclose(new_m, m_ref, atol=1e-6, rtol=1e-6).item()
    assert mx.allclose(new_v, v_ref, atol=1e-6, rtol=1e-6).item()


def test_dequantize_int8_matches_reference() -> None:
    x = mx.array([-8, -1, 0, 3, 7], dtype=mx.int8)
    scale = mx.array([0.25], dtype=mx.float32)
    result = quant.dequantize_int8(x, scale)
    ref = x.astype(mx.float32) * scale
    mx.eval(result, ref)
    assert mx.allclose(result, ref, atol=1e-6, rtol=1e-6).item()


def test_dequantize_silu_int8_matches_reference() -> None:
    x = mx.array([-8, -1, 0, 3, 7], dtype=mx.int8)
    scale = mx.array([0.1], dtype=mx.float32)
    result = quant.dequantize_silu_int8(x, scale)
    ref = x.astype(mx.float32) * scale
    ref = ref * mx.sigmoid(ref)
    mx.eval(result, ref)
    assert mx.allclose(result, ref, atol=1e-6, rtol=1e-6).item()


def test_dequantize_int4_matches_reference() -> None:
    packed = mx.array([0x21, 0xBA], dtype=mx.uint8)
    scale = mx.array([0.5], dtype=mx.float32)
    result = quant.dequantize_int4(packed, scale)
    mx.eval(result)
    packed_np = np.array(packed.tolist(), dtype=np.uint8)
    scale_val = float(scale.item())
    expected = np.zeros((packed_np.size * 2,), dtype=np.float32)
    for i, byte in enumerate(packed_np):
        expected[2 * i] = (byte & 0x0F) * scale_val
        expected[2 * i + 1] = (byte >> 4) * scale_val
    np.testing.assert_allclose(
        np.array(result.tolist(), dtype=np.float32),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("bits", [8, 4])
def test_dequantize_blockwise_matches_reference(bits: int) -> None:
    block_size = 4
    if bits == 8:
        x = mx.array([1, -2, 3, -4, 5, -6, 7, -8], dtype=mx.int8)
        scales = mx.array([0.5, 2.0], dtype=mx.float32)
        result = quant.dequantize_blockwise(x, scales, block_size=block_size, bits=bits)
        ref = x.astype(mx.float32).reshape(2, 4) * scales.reshape(2, 1)
        ref = ref.reshape(-1)
        mx.eval(result, ref)
        assert mx.allclose(result, ref, atol=1e-6, rtol=1e-6).item()
    else:
        x = mx.array([0x21, 0xBA, 0x07, 0xF0], dtype=mx.uint8)
        scales = mx.array([1.0, 0.5], dtype=mx.float32)
        result = quant.dequantize_blockwise(x, scales, block_size=block_size, bits=bits)
        mx.eval(result)
        x_np = np.array(x.tolist(), dtype=np.uint8)
        scales_np = np.array(scales.tolist(), dtype=np.float32)
        out_size = x_np.size * 2
        expected = np.zeros((out_size,), dtype=np.float32)
        for gid in range(out_size):
            block_idx = gid // block_size
            byte_idx = gid // 2
            nibble_idx = gid % 2
            packed = x_np[byte_idx]
            val = (packed & 0x0F) if nibble_idx == 0 else (packed >> 4)
            expected[gid] = val * scales_np[block_idx]
        np.testing.assert_allclose(
            np.array(result.tolist(), dtype=np.float32),
            expected,
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.xfail(reason="Quantized SwiGLU reference mismatch under investigation", strict=False)
def test_fused_swiglu_quant_matches_reference() -> None:
    block_size = 4
    x1 = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.int8)
    x2 = mx.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=mx.int8)
    scales1 = mx.array([0.5, 1.0], dtype=mx.float32)
    scales2 = mx.array([1.5, 0.25], dtype=mx.float32)
    result = quant.fused_swiglu_quant(
        x1,
        scales1,
        x2,
        scales2,
        block_size=block_size,
    )
    v1 = x1.astype(mx.float32).reshape(2, 4) * scales1.reshape(2, 1)
    v2 = x2.astype(mx.float32).reshape(2, 4) * scales2.reshape(2, 1)
    ref = (v1 * mx.sigmoid(v1)) * v2
    ref = ref.reshape(-1)
    mx.eval(result, ref)
    assert mx.allclose(result, ref, atol=1e-6, rtol=1e-6).item()


@pytest.mark.parametrize(
    "variant,bits",
    [
        ("e4m3", [0x2A, 0xAA, 0x35]),
        ("e5m2", [0x29, 0xA9, 0x3C]),
    ],
)
def test_dequantize_fp8_matches_reference(variant: str, bits: list[int]) -> None:
    x = mx.array(bits, dtype=mx.uint8)
    scale = mx.array([0.5], dtype=mx.float32)
    result = quant.dequantize_fp8(x, scale, variant=variant)
    mx.eval(result)
    expected = _fp8_ref(np.array(bits, dtype=np.uint8), float(scale.item()), variant)
    np.testing.assert_allclose(
        np.array(result.tolist(), dtype=np.float32),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.xfail(reason="NF4 address space qualifier issue on MLX 0.30.x", strict=False)
def test_dequantize_nf4_matches_reference() -> None:
    packed = mx.array([0x10, 0xF3], dtype=mx.uint8)
    scale = mx.array([0.25], dtype=mx.float32)
    result = quant.dequantize_nf4(packed, scale)
    mx.eval(result)
    table = np.array(quant._NF4_TABLE, dtype=np.float32)
    packed_np = np.array(packed.tolist(), dtype=np.uint8)
    scale_val = float(scale.item())
    expected = np.zeros((packed_np.size * 2,), dtype=np.float32)
    for gid in range(expected.size):
        byte_idx = gid // 2
        nibble_idx = gid % 2
        packed_val = packed_np[byte_idx]
        idx = (packed_val & 0x0F) if nibble_idx == 0 else (packed_val >> 4)
        expected[gid] = table[idx] * scale_val
    np.testing.assert_allclose(
        np.array(result.tolist(), dtype=np.float32),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skipif(
    not fused_moe.has_gather_qmm_swiglu(),
    reason="mx.gather_qmm_swiglu not available in this MLX build",
)
def test_gather_qmm_swiglu_wrapper_matches_reference() -> None:
    mx.random.seed(3)
    n_experts = 2
    k_dim = 512
    n_dim = 512
    group_size = 64
    bits = 4

    def quantize_experts(count: int) -> tuple[mx.array, mx.array, mx.array]:
        weights = []
        scales = []
        biases = []
        for _ in range(count):
            fp = mx.random.normal((n_dim, k_dim)).astype(mx.float16) * 0.02
            w, s, b = mx.quantize(fp, group_size=group_size, bits=bits)
            weights.append(w)
            scales.append(s)
            biases.append(b)
        return mx.stack(weights), mx.stack(scales), mx.stack(biases)

    gate_w, gate_s, gate_b = quantize_experts(n_experts)
    up_w, up_s, up_b = quantize_experts(n_experts)
    x = mx.random.normal((1, 1, k_dim)).astype(mx.float16) * 0.1
    lhs_indices = mx.array([0], dtype=mx.uint32)
    rhs_indices = mx.array([0], dtype=mx.uint32)

    result = fused_moe.gather_qmm_swiglu(
        x,
        gate_w,
        gate_s,
        gate_b,
        up_w,
        up_s,
        up_b,
        lhs_indices=lhs_indices,
        rhs_indices=rhs_indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )

    gate_out = mx.gather_qmm(
        x,
        gate_w,
        gate_s,
        gate_b,
        lhs_indices=lhs_indices,
        rhs_indices=rhs_indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )
    up_out = mx.gather_qmm(
        x,
        up_w,
        up_s,
        up_b,
        lhs_indices=lhs_indices,
        rhs_indices=rhs_indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )
    ref = (gate_out * mx.sigmoid(gate_out)) * up_out

    mx.eval(result, ref)
    assert mx.allclose(result, ref, atol=1e-2, rtol=1e-2).item()
