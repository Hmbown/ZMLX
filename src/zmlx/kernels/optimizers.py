from __future__ import annotations

from functools import cache
from typing import Any

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER


@cache
def _adamw_kernel(
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
) -> Any:
    source = f"""
        uint gid = thread_position_in_grid.x;
        float p_val = (float)p[gid];
        float g_val = (float)g[gid];
        float m_val = (float)m[gid];
        float v_val = (float)v[gid];
        float lr_val = (float)lr[0];
        float step_val = (float)step[0];

        float m_new = {beta1}f * m_val + (1.0f - {beta1}f) * g_val;
        float v_new = {beta2}f * v_val + (1.0f - {beta2}f) * g_val * g_val;

        float m_hat = m_new / (1.0f - pow({beta1}f, step_val));
        float v_hat = v_new / (1.0f - pow({beta2}f, step_val));

        float p_new = p_val - lr_val * (m_hat / (sqrt(v_hat) + {eps}f) + {wd}f * p_val);

        p_out[gid] = (T)p_new;
        m_out[gid] = (T)m_new;
        v_out[gid] = (T)v_new;
    """
    return metal_kernel(
        name=f"kk_adamw_b1_{beta1}_b2_{beta2}_eps_{eps}_wd_{wd}",
        input_names=["p", "g", "m", "v", "lr", "step"],
        output_names=["p_out", "m_out", "v_out"],
        source=source,
        header=DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )


def adamw_step(
    p: Any,
    g: Any,
    m: Any,
    v: Any,
    lr: Any,
    step: Any,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    wd: float = 0.01,
) -> tuple[Any, Any, Any]:
    """Fused AdamW update.
    
    Returns (new_p, new_m, new_v).
    """
    k = _adamw_kernel(beta1, beta2, eps, wd)
    out = k(
        p, g, m, v, lr, step,
        template=[("T", p.dtype)],
        output_shapes=[p.shape, m.shape, v.shape],
        output_dtypes=[p.dtype, m.dtype, v.dtype],
    )
    return out[0], out[1], out[2]
