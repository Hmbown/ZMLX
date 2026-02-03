from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def _dequant_int8_kernel(act_expr: str = "val") -> Any:
    source = f"""
        uint gid = thread_position_in_grid.x;
        float val = (float)inp[gid] * scale[0];
        out[gid] = (T)({act_expr});
    """
    return metal_kernel(
        name=f"kk_dequant_int8_{hash(act_expr) % 10000}",
        input_names=["inp", "scale"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def dequantize_int8(x: Any, scale: Any, *, compute_dtype: Any | None = None) -> Any:
    """Simple int8 dequantization: y = x * scale.
    
    x: int8 array
    scale: float32 scalar (or array of size 1)
    """
    cd = compute_dtype or mx.float32
    k = _dequant_int8_kernel()
    return k(
        x, scale,
        template=[("T", cd)],
        grid=(x.size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[cd],
    )[0]

def dequantize_silu_int8(x: Any, scale: Any, *, compute_dtype: Any | None = None) -> Any:
    """Fused dequantize + SiLU."""
    cd = compute_dtype or mx.float32
    k = _dequant_int8_kernel(act_expr="kk_silu(val)")
    return k(
        x, scale,
        template=[("T", cd)],
        grid=(x.size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[cd],
    )[0]


@cache
def _quantized_swiglu_gemv_kernel(
    m: int,
    n: int,
    k: int,
    *,
    bits: int,
    group_size: int,
) -> Any:
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bits for quantized SwiGLU GEMV: {bits}")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if k % group_size != 0:
        raise ValueError("group_size must evenly divide K")

    elements_per_word = 32 // bits
    if k % elements_per_word != 0:
        raise ValueError("K must be divisible by elements_per_word")

    k_packed = k // elements_per_word
    groups_per_row = k // group_size
    mask = (1 << bits) - 1

    source = f"""
        constexpr uint M = {m};
        constexpr uint N = {n};
        constexpr uint K = {k};
        constexpr uint K_PACKED = {k_packed};
        constexpr uint GROUP = {group_size};
        constexpr uint GROUPS = {groups_per_row};
        constexpr uint ELEMENTS = {elements_per_word};
        constexpr uint MASK = {mask}u;

        uint gid = thread_position_in_grid.x;
        uint row = gid / N;
        uint col = gid % N;

        if (row < M && col < N) {{
            float acc_gate = 0.0f;
            float acc_up = 0.0f;
            uint w_row = col * K_PACKED;
            uint s_row = col * GROUPS;
            uint x_row = row * K;

            for (uint i = 0; i < K; ++i) {{
                uint pack_idx = i / ELEMENTS;
                uint shift = (i % ELEMENTS) * {bits};
                uint q_gate = (gate_w[w_row + pack_idx] >> shift) & MASK;
                uint q_up = (up_w[w_row + pack_idx] >> shift) & MASK;
                uint group_idx = i / GROUP;

                float gate_val = (float)q_gate * (float)gate_scales[s_row + group_idx] + (float)gate_biases[s_row + group_idx];
                float up_val = (float)q_up * (float)up_scales[s_row + group_idx] + (float)up_biases[s_row + group_idx];
                float xv = (float)x[x_row + i];
                acc_gate += gate_val * xv;
                acc_up += up_val * xv;
            }}

            float outv = (acc_gate * kk_sigmoid(acc_gate)) * acc_up;
            out[gid] = (T)outv;
        }}
    """
    return metal_kernel(
        name=f"kk_qswiglu_gemv_M{m}_N{n}_K{k}_b{bits}",
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
        ],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def fused_quantized_swiglu_gemv(
    x: Any,
    gate_w: Any,
    gate_scales: Any,
    gate_biases: Any,
    up_w: Any,
    up_scales: Any,
    up_biases: Any,
    *,
    group_size: int,
    bits: int,
    compute_dtype: Any | None = None,
) -> Any:
    """Reference fused quantized GEMV + SwiGLU for decode-like shapes.

    This is a naive dot-product kernel (one thread per output element).
    Only intended for small-token decode paths where kernel fusion can
    outweigh the lack of tiling.
    """
    if x.ndim != 2:
        raise ValueError("fused_quantized_swiglu_gemv expects 2D input (M, K)")
    if gate_w.shape != up_w.shape:
        raise ValueError("gate_w and up_w must have the same shape")
    if gate_scales.shape != up_scales.shape or gate_biases.shape != up_biases.shape:
        raise ValueError("gate/up scales and biases must match")

    M, K = x.shape
    N, K_packed = gate_w.shape
    elements_per_word = 32 // bits
    if K != K_packed * elements_per_word:
        raise ValueError("Input K does not match packed weight shape")

    cd = compute_dtype or mx.float32
    k = _quantized_swiglu_gemv_kernel(int(M), int(N), int(K), bits=bits, group_size=group_size)
    return k(
        x,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
        template=[("T", cd)],
        grid=(int(M) * int(N), 1, 1),
        output_shapes=[(int(M), int(N))],
        output_dtypes=[cd],
    )[0]

@cache
def _dequant_affine_packed_kernel(
    *,
    bits: int,
    group_size: int,
    act_expr: str = "val",
) -> Any:
    if bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError(f"Unsupported bits for affine packed dequant: {bits}")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")

    elements_per_word = 32 // bits
    mask = (1 << bits) - 1
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint pack_idx = gid / {elements_per_word};
        uint shift = (gid % {elements_per_word}) * {bits};
        uint q = (inp[pack_idx] >> shift) & {mask}u;
        uint group_idx = gid / {group_size};
        float scale = (float)scales[group_idx];
        float bias = (float)biases[group_idx];
        float val = (float)q * scale + bias;
        out[gid] = (T)({act_expr});
    """
    return metal_kernel(
        name=f"kk_dequant_affine_b{bits}_g{group_size}_{hash(act_expr) % 10000}",
        input_names=["inp", "scales", "biases"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def _affine_packed_output_shape(x: Any, bits: int) -> tuple[int, ...]:
    elements_per_word = 32 // bits
    if x.ndim == 0:
        raise ValueError("affine packed dequant expects at least 1D input")
    out_last = x.shape[-1] * elements_per_word
    return (*x.shape[:-1], out_last)


def _affine_packed_grid(x: Any, bits: int) -> int:
    elements_per_word = 32 // bits
    return int(x.size * elements_per_word)


def dequantize_affine_packed(
    x: Any,
    scales: Any,
    biases: Any,
    *,
    bits: int = 4,
    group_size: int = 64,
    compute_dtype: Any | None = None,
) -> Any:
    """Dequantize MLX affine-packed uint32 weights.

    Mirrors ``mx.dequantize`` for affine-packed weights, returning a dense
    float tensor. Supports bits {2,3,4,5,6,8}.
    """
    cd = compute_dtype or mx.float32
    k = _dequant_affine_packed_kernel(bits=bits, group_size=group_size)
    out_shape = _affine_packed_output_shape(x, bits)
    grid = _affine_packed_grid(x, bits)
    return k(
        x,
        scales,
        biases,
        template=[("T", cd)],
        grid=(grid, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[cd],
    )[0]


def dequantize_affine_packed_silu(
    x: Any,
    scales: Any,
    biases: Any,
    *,
    bits: int = 4,
    group_size: int = 64,
    compute_dtype: Any | None = None,
) -> Any:
    """Fused affine-packed dequant + SiLU."""
    cd = compute_dtype or mx.float32
    k = _dequant_affine_packed_kernel(
        bits=bits,
        group_size=group_size,
        act_expr="kk_silu(val)",
    )
    out_shape = _affine_packed_output_shape(x, bits)
    grid = _affine_packed_grid(x, bits)
    return k(
        x,
        scales,
        biases,
        template=[("T", cd)],
        grid=(grid, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[cd],
    )[0]


def dequantize_affine_packed_gelu(
    x: Any,
    scales: Any,
    biases: Any,
    *,
    bits: int = 4,
    group_size: int = 64,
    compute_dtype: Any | None = None,
) -> Any:
    """Fused affine-packed dequant + GELU (tanh approximation)."""
    cd = compute_dtype or mx.float32
    k = _dequant_affine_packed_kernel(
        bits=bits,
        group_size=group_size,
        act_expr="kk_gelu_tanh(val)",
    )
    out_shape = _affine_packed_output_shape(x, bits)
    grid = _affine_packed_grid(x, bits)
    return k(
        x,
        scales,
        biases,
        template=[("T", cd)],
        grid=(grid, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[cd],
    )[0]


@cache
def _dequant_int4_kernel() -> Any:
    source = """
        uint gid = thread_position_in_grid.x;
        uint byte_idx = gid / 2;
        uint nibble_idx = gid % 2;
        
        uint8_t packed = inp[byte_idx];
        int8_t val;
        if (nibble_idx == 0) {
            val = (int8_t)(packed & 0x0F);
        } else {
            val = (int8_t)(packed >> 4);
        }
        // Signed 4-bit range is -8 to 7. 
        // If we want simple unsigned 0-15:
        // val = (nibble_idx == 0) ? (packed & 0x0F) : (packed >> 4);
        
        out[gid] = (T)((float)val * scale[0]);
    """
    return metal_kernel(
        name="kk_dequant_int4",
        input_names=["inp", "scale"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def dequantize_int4(x: Any, scale: Any, *, compute_dtype: Any | None = None) -> Any:
    """Dequantize 4-bit data packed into uint8.
    
    x: uint8 array (size is half of output size)
    scale: float32 scalar
    """
    cd = compute_dtype or mx.float32
    k = _dequant_int4_kernel()
    out_size = x.size * 2
    return k(
        x, scale,
        template=[("T", cd)],
        grid=(out_size, 1, 1),
        output_shapes=[(out_size,)],
        output_dtypes=[cd],
    )[0]

@cache
def _dequant_blockwise_kernel(bits: int = 8, block_size: int = 128, act_expr: str = "val") -> Any:
    if bits == 8:
        load_val = "(float)inp[gid]"
    elif bits == 4:
        # Assuming packed uint8 for 4-bit
        load_val = """
        ((gid % 2 == 0) ? (float)(inp[gid/2] & 0x0F) : (float)(inp[gid/2] >> 4))
        """
    else:
        raise ValueError(f"Unsupported bits: {bits}")

    source = f"""
        uint gid = thread_position_in_grid.x;
        uint block_idx = gid / {block_size};
        float scale = (float)scales[block_idx];
        float val = {load_val} * scale;
        out[gid] = (T)({act_expr});
    """
    return metal_kernel(
        name=f"kk_dequant_b{bits}_bs{block_size}_{hash(act_expr) % 10000}",
        input_names=["inp", "scales"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def dequantize_blockwise(
    x: Any, scales: Any, *, block_size: int = 128, bits: int = 8, compute_dtype: Any | None = None
) -> Any:
    """Block-wise dequantization with scales per block."""
    cd = compute_dtype or mx.float32
    k = _dequant_blockwise_kernel(bits=bits, block_size=block_size)
    grid_size = x.size if bits == 8 else x.size * 2
    return k(
        x, scales,
        template=[("T", cd)],
        grid=(grid_size, 1, 1),
        output_shapes=[(grid_size,)],
        output_dtypes=[cd],
    )[0]


@cache
def _swiglu_quant_kernel(block_size: int = 128) -> Any:
    # Fused SwiGLU: (dequant(x1) * silu(dequant(x1))) * dequant(x2)
    source = f"""
        uint gid = thread_position_in_grid.x;
        uint block_idx = gid / {block_size};
        float s1 = (float)scales1[block_idx];
        float s2 = (float)scales2[block_idx];
        
        float v1 = (float)x1[gid] * s1;
        float v2 = (float)x2[gid] * s2;
        
        float res = (v1 * kk_silu(v1)) * v2;
        out[gid] = (T)res;
    """
    return metal_kernel(
        name=f"kk_swiglu_quant_bs{block_size}",
        input_names=["x1", "scales1", "x2", "scales2"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def fused_swiglu_quant(
    x1: Any, scales1: Any, x2: Any, scales2: Any, *, block_size: int = 128, compute_dtype: Any | None = None
) -> Any:
    """Fused dequantization + SwiGLU for MLP layers."""
    cd = compute_dtype or mx.float32
    k = _swiglu_quant_kernel(block_size=block_size)
    return k(
        x1, scales1, x2, scales2,
        template=[("T", cd)],
        grid=(x1.size, 1, 1),
        output_shapes=[x1.shape],
        output_dtypes=[cd],
    )[0]


@cache
def _dequant_fp8_kernel(variant: str = "e4m3") -> Any:
    """FP8 dequantization via sign/exponent/mantissa bit extraction.

    Supports E4M3 (4-bit exponent, 3-bit mantissa, bias 7) and
    E5M2 (5-bit exponent, 2-bit mantissa, bias 15).  Input is uint8.
    """
    if variant == "e4m3":
        # E4M3: sign(1) | exponent(4) | mantissa(3), bias = 7
        source = """
            uint gid = thread_position_in_grid.x;
            uint8_t bits = inp[gid];

            float sign = (bits & 0x80) ? -1.0f : 1.0f;
            int exp_bits = (int)((bits >> 3) & 0x0F);  // 4-bit exponent
            int mant_bits = (int)(bits & 0x07);         // 3-bit mantissa

            float val;
            if (exp_bits == 0) {
                // Subnormal: (-1)^s * 2^(1-bias) * (0 + mant/8)
                val = sign * exp2(-6.0f) * ((float)mant_bits / 8.0f);
            } else if (exp_bits == 15 && mant_bits == 7) {
                // NaN (E4M3 has no Inf — exp=15,mant=7 is NaN)
                val = NAN;
            } else {
                // Normal: (-1)^s * 2^(exp-bias) * (1 + mant/8)
                val = sign * exp2((float)(exp_bits - 7)) * (1.0f + (float)mant_bits / 8.0f);
            }

            out[gid] = (T)(val * (float)scale[0]);
        """
    elif variant == "e5m2":
        # E5M2: sign(1) | exponent(5) | mantissa(2), bias = 15
        source = """
            uint gid = thread_position_in_grid.x;
            uint8_t bits = inp[gid];

            float sign = (bits & 0x80) ? -1.0f : 1.0f;
            int exp_bits = (int)((bits >> 2) & 0x1F);  // 5-bit exponent
            int mant_bits = (int)(bits & 0x03);         // 2-bit mantissa

            float val;
            if (exp_bits == 0) {
                // Subnormal: (-1)^s * 2^(1-bias) * (0 + mant/4)
                val = sign * exp2(-14.0f) * ((float)mant_bits / 4.0f);
            } else if (exp_bits == 31) {
                // Inf/NaN (follows IEEE-like convention)
                val = (mant_bits == 0) ? sign * INFINITY : NAN;
            } else {
                // Normal: (-1)^s * 2^(exp-bias) * (1 + mant/4)
                val = sign * exp2((float)(exp_bits - 15)) * (1.0f + (float)mant_bits / 4.0f);
            }

            out[gid] = (T)(val * (float)scale[0]);
        """
    else:
        raise ValueError(f"Unknown FP8 variant: {variant!r}. Use 'e4m3' or 'e5m2'.")

    return metal_kernel(
        name=f"kk_dequant_fp8_{variant}",
        input_names=["inp", "scale"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def dequantize_fp8(
    x: Any, scale: Any, *, variant: str = "e4m3", compute_dtype: Any | None = None
) -> Any:
    """Dequantize FP8 data (uint8 encoding) to float.

    Args:
        x: uint8 array containing FP8-encoded values.
        scale: float32 scalar multiplier.
        variant: ``"e4m3"`` (4-bit exp, 3-bit mantissa) or
                 ``"e5m2"`` (5-bit exp, 2-bit mantissa).
        compute_dtype: Output dtype (default: float32).
    """
    cd = compute_dtype or mx.float32
    k = _dequant_fp8_kernel(variant)
    return k(
        x, scale,
        template=[("T", cd)],
        grid=(x.size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[cd],
    )[0]


# ---------------------------------------------------------------------------
# NF4 (NormalFloat 4-bit) dequantization — lookup-table based
# ---------------------------------------------------------------------------
# The 16 NF4 values are the optimal 4-bit quantiles of a standard normal
# distribution, as defined in the QLoRA paper (Dettmers et al., 2023).
_NF4_TABLE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]


@cache
def _dequant_nf4_kernel() -> Any:
    """NF4 lookup-table dequantization kernel.

    Input is uint8 with two 4-bit values packed per byte (low nibble first).
    Uses a constant lookup table embedded in the kernel source.
    """
    lut_str = ", ".join(f"{v:.10f}f" for v in _NF4_TABLE)
    source = f"""
        constant float nf4_lut[16] = {{{lut_str}}};

        uint gid = thread_position_in_grid.x;
        uint byte_idx = gid / 2;
        uint nibble_idx = gid % 2;

        uint8_t packed = inp[byte_idx];
        uint8_t idx = (nibble_idx == 0) ? (packed & 0x0F) : (packed >> 4);

        float val = nf4_lut[idx] * (float)scale[0];
        out[gid] = (T)val;
    """
    return metal_kernel(
        name="kk_dequant_nf4",
        input_names=["inp", "scale"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def dequantize_nf4(x: Any, scale: Any, *, compute_dtype: Any | None = None) -> Any:
    """Dequantize NF4 (NormalFloat 4-bit) packed data.

    Args:
        x: uint8 array with two NF4 values per byte (low nibble first).
        scale: float32 scalar (absmax scale factor).
        compute_dtype: Output dtype (default: float32).

    Returns:
        Dequantized array of shape ``(x.size * 2,)``.
    """
    cd = compute_dtype or mx.float32
    k = _dequant_nf4_kernel()
    out_size = x.size * 2
    return k(
        x, scale,
        template=[("T", cd)],
        grid=(out_size, 1, 1),
        output_shapes=[(out_size,)],
        output_dtypes=[cd],
    )[0]


__all__ = [
    "dequantize_int8",
    "dequantize_silu_int8",
    "dequantize_int4",
    "dequantize_blockwise",
    "dequantize_affine_packed",
    "dequantize_affine_packed_silu",
    "dequantize_affine_packed_gelu",
    "fused_quantized_swiglu_gemv",
    "fused_swiglu_quant",
    "dequantize_fp8",
    "dequantize_nf4",
]
