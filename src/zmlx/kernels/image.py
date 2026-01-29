from __future__ import annotations

from functools import cache
from typing import Any

import mlx.core as mx

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def _resize_bilinear_kernel(oh: int, ow: int, ih: int, iw: int) -> Any:
    source = f"""
        constexpr uint OH = {oh};
        constexpr uint OW = {ow};
        constexpr uint IH = {ih};
        constexpr uint IW = {iw};

        uint gid = thread_position_in_grid.x;
        uint c = gid % C;
        uint x = (gid / C) % OW;
        uint y = (gid / C / OW) % OH;
        uint b = gid / C / OW / OH;

        float scale_h = (float)IH / (float)OH;
        float scale_w = (float)IW / (float)OW;

        float in_y = (y + 0.5f) * scale_h - 0.5f;
        float in_x = (x + 0.5f) * scale_w - 0.5f;

        int y0 = (int)metal::floor(in_y);
        int x0 = (int)metal::floor(in_x);
        int y1 = y0 + 1;
        int x1 = x0 + 1;

        float wy = in_y - y0;
        float wx = in_x - x0;

        y0 = metal::clamp(y0, 0, (int)IH - 1);
        y1 = metal::clamp(y1, 0, (int)IH - 1);
        x0 = metal::clamp(x0, 0, (int)IW - 1);
        x1 = metal::clamp(x1, 0, (int)IW - 1);

        uint b_offset = b * IH * IW * C;
        float v00 = (float)inp[b_offset + (y0 * IW + x0) * C + c];
        float v01 = (float)inp[b_offset + (y0 * IW + x1) * C + c];
        float v10 = (float)inp[b_offset + (y1 * IW + x0) * C + c];
        float v11 = (float)inp[b_offset + (y1 * IW + x1) * C + c];

        float res = v00 * (1.0f - wy) * (1.0f - wx) +
                    v01 * (1.0f - wy) * wx +
                    v10 * wy * (1.0f - wx) +
                    v11 * wy * wx;

        out[gid] = (T)res;
    """
    return metal_kernel(
        name=f"kk_resize_bilinear_{oh}_{ow}",
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def resize_bilinear(x: Any, output_shape: tuple[int, int], *, compute_dtype: Any | None = None) -> Any:
    """Bilinear resize for NHWC images.
    
    x: (N, H, W, C)
    output_shape: (new_H, new_W)
    """
    N, IH, IW, C = x.shape
    OH, OW = output_shape
    cd = compute_dtype or mx.float32
    
    k = _resize_bilinear_kernel(OH, OW, IH, IW)
    return k(
        x,
        template=[("T", cd), ("C", int(C))],
        grid=(N * OH * OW * C, 1, 1),
        output_shapes=[(N, OH, OW, C)],
        output_dtypes=[x.dtype],
    )[0]

@cache
def _depthwise_conv_3x3_kernel(h: int, w: int, c: int) -> Any:
    source = f"""
        constexpr uint H = {h};
        constexpr uint W = {w};
        constexpr uint C = {c};

        uint gid = thread_position_in_grid.x;
        uint channel = gid % C;
        uint x = (gid / C) % W;
        uint y = (gid / C / W) % H;
        uint b = gid / C / W / H;

        float sum = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {{
            for (int dx = -1; dx <= 1; ++dx) {{
                int iy = (int)y + dy;
                int ix = (int)x + dx;
                if (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W) {{
                    float val = (float)inp[b * H * W * C + (iy * W + ix) * C + channel];
                    float weight = (float)weights[((dy + 1) * 3 + (dx + 1)) * C + channel];
                    sum += val * weight;
                }}
            }}
        }}
        out[gid] = (T)sum;
    """
    return metal_kernel(
        name=f"kk_dw_conv3x3_H{h}_W{w}_C{c}",
        input_names=["inp", "weights"],
        output_names=["out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

def depthwise_conv_3x3(x: Any, w: Any, *, compute_dtype: Any | None = None) -> Any:
    """3x3 Depthwise convolution.
    
    x: (N, H, W, C)
    w: (3, 3, C)
    """
    N, H, W, C = x.shape
    cd = compute_dtype or mx.float32
    k = _depthwise_conv_3x3_kernel(H, W, C)
    return k(
        x, w,
        template=[("T", cd)],
        grid=(x.size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]

__all__ = [
    "resize_bilinear",
    "depthwise_conv_3x3",
]
