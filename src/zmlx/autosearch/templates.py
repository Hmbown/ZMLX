"""Parameterized Metal kernel templates for autosearch.

Each template is a function: (config, dimensions) -> (source, grid, threadgroup).
"""

from __future__ import annotations

import math
from typing import Any

from .space import ConfigSpace, Knob

# ---------------------------------------------------------------------------
# Config space definitions
# ---------------------------------------------------------------------------

def swiglu_space(D: int) -> ConfigSpace:
    """Config space for fused SwiGLU elementwise kernel."""
    return ConfigSpace(
        knobs=(
            Knob("vec_width", (1, 2, 4), default=1),
            Knob("elems_per_thread", (1, 2, 4, 8), default=1),
            Knob("tg_size", (32, 64, 128, 256, 512, 1024), default=256),
            Knob("use_fast_sigmoid", (True, False), default=False),
        ),
        constraints=(
            lambda c: c["tg_size"] <= 1024,
            lambda c: D % c["vec_width"] == 0,
        ),
    )


def rmsnorm_space(D: int) -> ConfigSpace:
    """Config space for RMSNorm reduction kernel."""
    return ConfigSpace(
        knobs=(
            Knob("tg_size", (32, 64, 128, 256, 512, 1024), default=256),
            Knob("vec_width", (1, 2, 4), default=1),
            Knob("reduce_method", ("simd_tree", "sequential"), default="simd_tree"),
            Knob("fused_write", (True, False), default=True),
        ),
        constraints=(
            lambda c: c["tg_size"] <= 1024,
            lambda c: D % c["vec_width"] == 0,
            # sequential reduction needs tg_size <= D
            lambda c: c["reduce_method"] != "sequential" or c["tg_size"] <= D,
            # SIMD tree needs tg_size to be multiple of 32
            lambda c: c["reduce_method"] != "simd_tree" or c["tg_size"] % 32 == 0,
        ),
    )


def moe_combine_space(D: int, K: int = 2) -> ConfigSpace:  # noqa: ARG001
    """Config space for MoE combine kernel."""
    _ = K  # K affects template, not constraints
    return ConfigSpace(
        knobs=(
            Knob("vec_width", (1, 2, 4), default=1),
            Knob("tg_size", (32, 64, 128, 256, 512, 1024), default=min(D, 256)),
            Knob("unroll_k", (True, False), default=False),
            Knob("use_fma", (True, False), default=False),
        ),
        constraints=(
            lambda c: c["tg_size"] <= 1024,
            lambda c: D % c["vec_width"] == 0,
        ),
    )


# ---------------------------------------------------------------------------
# Metal source templates
# ---------------------------------------------------------------------------

def swiglu_template(
    config: dict[str, Any], D: int
) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
    """Generate fused SwiGLU Metal source from config.

    Returns (source, grid, threadgroup).
    """
    vw = config["vec_width"]
    ept = config["elems_per_thread"]
    tg = config["tg_size"]
    fast_sig = config["use_fast_sigmoid"]

    N = D  # total elements

    if vw == 1:
        load_g = "float g = (float)gate[idx];"
        load_u = "float u = (float)up[idx];"
        if fast_sig:
            sigmoid = "float sig = kk_sigmoid(g);"
        else:
            sigmoid = "float sig = 1.0f / (1.0f + metal::exp(-g));"
        store = "out[idx] = (T)(g * sig * u);"
    else:
        vt = f"float{vw}"
        load_g = f"{vt} gv; for (int _v = 0; _v < {vw}; ++_v) gv[_v] = (float)gate[idx * {vw} + _v];"
        load_u = f"{vt} uv; for (int _v = 0; _v < {vw}; ++_v) uv[_v] = (float)up[idx * {vw} + _v];"
        if fast_sig:
            sigmoid = f"{vt} sv; for (int _v = 0; _v < {vw}; ++_v) sv[_v] = kk_sigmoid(gv[_v]);"
        else:
            sigmoid = f"{vt} sv; for (int _v = 0; _v < {vw}; ++_v) sv[_v] = 1.0f / (1.0f + metal::exp(-gv[_v]));"
        store = f"for (int _v = 0; _v < {vw}; ++_v) out[idx * {vw} + _v] = (T)(gv[_v] * sv[_v] * uv[_v]);"

    elems_per_dispatch = N // vw
    grid_x = math.ceil(elems_per_dispatch / ept)

    if ept == 1:
        source = f"""
        constexpr uint N = {N};
        uint idx = thread_position_in_grid.x;
        if (idx * {vw} >= N) return;
        {load_g}
        {load_u}
        {sigmoid}
        {store}
"""
    else:
        source = f"""
        constexpr uint N = {N};
        constexpr uint ELEMS = {elems_per_dispatch};
        uint base = thread_position_in_grid.x * {ept};
        for (uint _e = 0; _e < {ept}; ++_e) {{
            uint idx = base + _e;
            if (idx * {vw} >= N) return;
            {load_g}
            {load_u}
            {sigmoid}
            {store}
        }}
"""

    grid = (grid_x, 1, 1)
    threadgroup = (min(grid_x, tg), 1, 1)
    return source, grid, threadgroup


def rmsnorm_template(
    config: dict[str, Any], D: int
) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
    """Generate RMSNorm Metal source from config.

    Returns (source, grid, threadgroup). Grid must be set per-batch externally.
    The returned grid assumes B=1; the harness multiplies grid.x by B.
    """
    tg = config["tg_size"]
    vw = config["vec_width"]
    reduce_method = config["reduce_method"]
    fused = config["fused_write"]

    # Reduction pass
    if vw == 1:
        accum_loop = """
        float sumsq = 0.0f;
        for (uint j = tid; j < D; j += TG) {
            float v = (float)inp[base + j];
            sumsq += v * v;
        }"""
    else:
        accum_loop = f"""
        float sumsq = 0.0f;
        for (uint j = tid * {vw}; j < D; j += TG * {vw}) {{
            for (uint _v = 0; _v < {vw}; ++_v) {{
                float v = (float)inp[base + j + _v];
                sumsq += v * v;
            }}
        }}"""

    if reduce_method == "simd_tree":
        reduce_code = """
        threadgroup float buf[TG];
        KK_SIMD_REDUCE_SUM(buf, sumsq, tid, TG);
        float inv = metal::rsqrt(buf[0] / (float)D + EPS);
        threadgroup_barrier(mem_flags::mem_threadgroup);"""
    else:
        reduce_code = """
        threadgroup float buf[TG];
        buf[tid] = sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint i = 0; i < TG; ++i) total += buf[i];
            buf[0] = metal::rsqrt(total / (float)D + EPS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = buf[0];"""

    # Write pass
    if fused and vw == 1:
        write_loop = """
        for (uint j = tid; j < D; j += TG) {
            float v = (float)inp[base + j];
            float w = (float)weight[j];
            out[base + j] = (T)(v * inv * w);
        }"""
    elif fused and vw > 1:
        write_loop = f"""
        for (uint j = tid * {vw}; j < D; j += TG * {vw}) {{
            for (uint _v = 0; _v < {vw}; ++_v) {{
                float v = (float)inp[base + j + _v];
                float w = (float)weight[j + _v];
                out[base + j + _v] = (T)(v * inv * w);
            }}
        }}"""
    else:
        # Two-pass: compute inv first, then write
        write_loop = """
        for (uint j = tid; j < D; j += TG) {
            out[base + j] = (T)((float)inp[base + j] * inv * (float)weight[j]);
        }"""

    source = f"""
        constexpr uint D = {D};
        constexpr uint TG = {tg};
        constexpr float EPS = 1e-6f;

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;
{accum_loop}
{reduce_code}
{write_loop}
"""

    # Grid for B=1; harness scales by B
    grid = (tg, 1, 1)
    threadgroup = (tg, 1, 1)
    return source, grid, threadgroup


def moe_combine_template(
    config: dict[str, Any], D: int, K: int
) -> tuple[str, tuple[int, int, int], tuple[int, int, int]]:
    """Generate MoE combine Metal source from config.

    Returns (source, grid, threadgroup). Grid.y is set to 1; harness sets it to B.
    """
    vw = config["vec_width"]
    tg = config["tg_size"]
    unroll_k = config["unroll_k"]
    use_fma = config["use_fma"]

    if unroll_k:
        # Unrolled expert loop
        if use_fma:
            inner_lines = []
            for i in range(K):
                inner_lines.append(f"float w{i} = (float)weights[token_idx * {K} + {i}];")
            for i in range(K):
                if i == 0:
                    inner_lines.append(f"float acc = w0 * (float)expert_outputs[(token_idx * {K} + 0) * {D} + d_idx];")
                else:
                    inner_lines.append(
                        f"acc = fma(w{i}, (float)expert_outputs[(token_idx * {K} + {i}) * {D} + d_idx], acc);"
                    )
            inner = "\n            ".join(inner_lines)
        else:
            inner_lines = []
            for i in range(K):
                inner_lines.append(f"float w{i} = (float)weights[token_idx * {K} + {i}];")
            acc_parts = []
            for i in range(K):
                acc_parts.append(
                    f"w{i} * (float)expert_outputs[(token_idx * {K} + {i}) * {D} + d_idx]"
                )
            inner_lines.append(f"float acc = {' + '.join(acc_parts)};")
            inner = "\n            ".join(inner_lines)
    else:
        # Loop over experts
        if use_fma:
            inner = f"""float acc = 0.0f;
            for (uint i = 0; i < {K}; ++i) {{
                float w = (float)weights[token_idx * {K} + i];
                float v = (float)expert_outputs[(token_idx * {K} + i) * {D} + d_idx];
                acc = fma(w, v, acc);
            }}"""
        else:
            inner = f"""float acc = 0.0f;
            for (uint i = 0; i < {K}; ++i) {{
                float w = (float)weights[token_idx * {K} + i];
                float v = (float)expert_outputs[(token_idx * {K} + i) * {D} + d_idx];
                acc += w * v;
            }}"""

    if vw == 1:
        source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_idx = thread_position_in_grid.x;
        if (d_idx >= D) return;

            {inner}
        out[token_idx * D + d_idx] = (T)acc;
"""
    else:
        # Vectorized: each thread handles vw elements
        source = f"""
        constexpr uint D = {D};
        constexpr uint K = {K};
        uint token_idx = thread_position_in_grid.y;
        uint d_base = thread_position_in_grid.x * {vw};
        if (d_base >= D) return;

        for (uint _v = 0; _v < {vw}; ++_v) {{
            uint d_idx = d_base + _v;
            {inner}
            out[token_idx * D + d_idx] = (T)acc;
        }}
"""

    grid_x = math.ceil(D / vw)
    grid = (grid_x, 1, 1)  # .y set to B by harness
    threadgroup = (min(grid_x, tg), 1, 1)
    return source, grid, threadgroup


# ---------------------------------------------------------------------------
# Registry: template_name -> (space_factory, template_fn, target_name)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, tuple[str, ...]] = {
    "swiglu": ("swiglu", "fused_swiglu"),
    "rmsnorm": ("rmsnorm", "rmsnorm"),
    "moe_combine": ("moe_combine", "moe_combine"),
}


def get_space(template_name: str, D: int, K: int = 2) -> ConfigSpace:
    """Return the config space for a given template."""
    if template_name == "swiglu":
        return swiglu_space(D)
    elif template_name == "rmsnorm":
        return rmsnorm_space(D)
    elif template_name == "moe_combine":
        return moe_combine_space(D, K)
    raise ValueError(f"Unknown template: {template_name}")


def get_template_fn(
    template_name: str, D: int, K: int = 2
) -> Any:
    """Return a callable (config) -> (source, grid, threadgroup)."""
    if template_name == "swiglu":
        return lambda config: swiglu_template(config, D)
    elif template_name == "rmsnorm":
        return lambda config: rmsnorm_template(config, D)
    elif template_name == "moe_combine":
        return lambda config: moe_combine_template(config, D, K)
    raise ValueError(f"Unknown template: {template_name}")


def get_target_name(template_name: str) -> str:
    """Return the discover target name for baseline/reference reuse."""
    if template_name in TEMPLATES:
        return TEMPLATES[template_name][1]
    raise ValueError(f"Unknown template: {template_name}")
