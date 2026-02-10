"""Decode RoPE Q/K concat helper for kernel discovery."""

from __future__ import annotations

from typing import Any

from ..types import KernelCandidate
from . import dtype_itemsize, mx_dtype, render_template, template_text

OP_NAME = "rope"
DEFAULT_DTYPE = "float16"
DEFAULT_SHAPE_SUITE = "glm_flash_small"

SHAPE_SUITES: dict[str, list[dict[str, int]]] = {
    "glm_flash_small": [
        {"B": 1, "H_Q": 32, "D_NOPE": 128, "D_ROPE": 64},
        {"B": 2, "H_Q": 32, "D_NOPE": 128, "D_ROPE": 64},
        {"B": 4, "H_Q": 32, "D_NOPE": 128, "D_ROPE": 64},
    ],
    "glm_flash_decode": [
        {"B": 1, "H_Q": 24, "D_NOPE": 128, "D_ROPE": 64},
        {"B": 1, "H_Q": 32, "D_NOPE": 128, "D_ROPE": 64},
        {"B": 1, "H_Q": 40, "D_NOPE": 128, "D_ROPE": 64},
    ],
}

TEMPLATE_PARAM_SPACE: dict[str, tuple[Any, ...]] = {
    "use_fma": (False, True),
}

LAUNCH_PARAM_SPACE: dict[str, tuple[Any, ...]] = {
    "threadgroup_x": (64, 128, 256, 512),
}


def seed_template_params() -> dict[str, Any]:
    return {
        "use_fma": False,
    }


def seed_launch_params() -> dict[str, Any]:
    return {
        "threadgroup_x": 256,
        "launch_kind": "rope_decode_pos",
    }


def normalize_shape(shape: dict[str, Any]) -> dict[str, int]:
    return {
        "B": int(shape["B"]),
        "H_Q": int(shape["H_Q"]),
        "D_NOPE": int(shape["D_NOPE"]),
        "D_ROPE": int(shape["D_ROPE"]),
    }


def shape_signature(shape: dict[str, Any]) -> dict[str, int]:
    s = normalize_shape(shape)
    return {
        "B": s["B"],
        "H_Q": s["H_Q"],
        "D_NOPE": s["D_NOPE"],
        "D_ROPE": s["D_ROPE"],
        "D_OUT": s["D_NOPE"] + s["D_ROPE"],
    }


def bytes_moved(shape: dict[str, Any], dtype_name: str) -> int:
    s = normalize_shape(shape)
    b = s["B"]
    hq = s["H_Q"]
    d_nope = s["D_NOPE"]
    d_rope = s["D_ROPE"]
    d_out = d_nope + d_rope
    item = dtype_itemsize(dtype_name)

    inp = b * hq * d_nope + b * hq * d_rope + b * d_nope + b * d_rope + d_rope
    out = b * hq * d_out + b * d_out
    return int((inp + out) * item)


def arithmetic_intensity(shape: dict[str, Any], dtype_name: str) -> float:
    s = normalize_shape(shape)
    b = s["B"]
    hq = s["H_Q"]
    d_rope = s["D_ROPE"]
    flops = float(b * (hq + 1) * d_rope * 4)
    bbytes = bytes_moved(s, dtype_name)
    return float(flops / max(1.0, float(bbytes)))


def _interleave(even: Any, odd: Any) -> Any:
    import mlx.core as mx

    stacked = mx.stack([even, odd], axis=-1)
    shape = tuple(int(v) for v in even.shape[:-1]) + (int(even.shape[-1]) * 2,)
    return stacked.reshape(shape)


def make_inputs(mx_mod: Any, shape: dict[str, Any], dtype_name: str, seed: int) -> list[Any]:
    s = normalize_shape(shape)
    b = s["B"]
    hq = s["H_Q"]
    d_nope = s["D_NOPE"]
    d_rope = s["D_ROPE"]
    half = d_rope // 2

    mx_mod.random.seed(seed)
    dt = mx_dtype(mx_mod, dtype_name)

    q_nope = mx_mod.random.normal((b, hq, 1, d_nope)).astype(dt)
    q_rope = mx_mod.random.normal((b, hq, 1, d_rope)).astype(dt)
    kv_nope = mx_mod.random.normal((b, 1, 1, d_nope)).astype(dt)
    k_rope = mx_mod.random.normal((b, 1, 1, d_rope)).astype(dt)
    cos = mx_mod.random.normal((half,)).astype(dt)
    sin = mx_mod.random.normal((half,)).astype(dt)
    return [q_nope, q_rope, kv_nope, k_rope, cos, sin]


def reference(mx_mod: Any, inputs: list[Any], shape: dict[str, Any], dtype_name: str) -> list[Any]:
    _ = (mx_mod, shape, dtype_name)
    q_nope, q_rope, kv_nope, k_rope, cos, sin = inputs

    c = cos.reshape(1, 1, 1, -1).astype(mx_mod.float32)
    s = sin.reshape(1, 1, 1, -1).astype(mx_mod.float32)

    q_even = q_rope[..., ::2].astype(mx_mod.float32)
    q_odd = q_rope[..., 1::2].astype(mx_mod.float32)
    q_rot_even = q_even * c - q_odd * s
    q_rot_odd = q_even * s + q_odd * c
    q_rot = _interleave(q_rot_even, q_rot_odd).astype(q_rope.dtype)
    q_out = mx_mod.concatenate([q_nope, q_rot], axis=-1)

    k_even = k_rope[..., ::2].astype(mx_mod.float32)
    k_odd = k_rope[..., 1::2].astype(mx_mod.float32)
    k_rot_even = k_even * c - k_odd * s
    k_rot_odd = k_even * s + k_odd * c
    k_rot = _interleave(k_rot_even, k_rot_odd).astype(k_rope.dtype)
    k_out = mx_mod.concatenate([kv_nope, k_rot], axis=-1)

    return [q_out, k_out]


def output_shapes(shape: dict[str, Any]) -> list[tuple[int, ...]]:
    s = normalize_shape(shape)
    d_out = s["D_NOPE"] + s["D_ROPE"]
    return [
        (s["B"], s["H_Q"], 1, d_out),
        (s["B"], 1, 1, d_out),
    ]


def output_dtypes(mx_mod: Any, dtype_name: str) -> list[Any]:
    dt = mx_dtype(mx_mod, dtype_name)
    return [dt, dt]


def compute_launch(
    launch_params: dict[str, Any],
    shape: dict[str, Any],
    _inputs: list[Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    s = normalize_shape(shape)
    d_out = s["D_NOPE"] + s["D_ROPE"]
    grid_x = s["B"] * (s["H_Q"] + 1) * d_out
    tg = min(int(launch_params["threadgroup_x"]), max(1, grid_x))
    return (max(1, grid_x), 1, 1), (max(1, tg), 1, 1)


def make_candidate(
    *,
    template_params: dict[str, Any],
    launch_params: dict[str, Any],
    shape: dict[str, Any],
    dtype_name: str,
    parent_id: str | None = None,
) -> KernelCandidate:
    s = normalize_shape(shape)
    tg = int(launch_params["threadgroup_x"])
    params = {
        "D_NOPE": s["D_NOPE"],
        "D_ROPE": s["D_ROPE"],
        "H_Q": s["H_Q"],
        "USE_FMA": "true" if bool(template_params["use_fma"]) else "false",
    }
    source = render_template(template_text("rope.metal.tmpl"), params)

    func_name = (
        f"kk_kd_rope_dn{s['D_NOPE']}_dr{s['D_ROPE']}_hq{s['H_Q']}_tg{tg}_"
        f"fma{1 if params['USE_FMA'] == 'true' else 0}"
    )

    launch = dict(launch_params)
    launch["launch_kind"] = "rope_decode_pos"

    b = bytes_moved(s, dtype_name)
    features = {
        "bytes_moved_est": float(b),
        "arithmetic_intensity_est": arithmetic_intensity(s, dtype_name),
        "launch_occupancy_proxy": float(min(1.0, tg / 1024.0)),
    }

    inputs_spec = [
        {
            "name": "q_nope",
            "dtype": dtype_name,
            "shape": [s["B"], s["H_Q"], 1, s["D_NOPE"]],
            "strides": "contiguous",
        },
        {
            "name": "q_rope",
            "dtype": dtype_name,
            "shape": [s["B"], s["H_Q"], 1, s["D_ROPE"]],
            "strides": "contiguous",
        },
        {
            "name": "kv_nope",
            "dtype": dtype_name,
            "shape": [s["B"], 1, 1, s["D_NOPE"]],
            "strides": "contiguous",
        },
        {
            "name": "k_rope",
            "dtype": dtype_name,
            "shape": [s["B"], 1, 1, s["D_ROPE"]],
            "strides": "contiguous",
        },
        {"name": "cos", "dtype": dtype_name, "shape": [s["D_ROPE"] // 2], "strides": "contiguous"},
        {"name": "sin", "dtype": dtype_name, "shape": [s["D_ROPE"] // 2], "strides": "contiguous"},
    ]

    d_out = s["D_NOPE"] + s["D_ROPE"]
    outputs_spec = [
        {
            "name": "q_out",
            "dtype": dtype_name,
            "shape": [s["B"], s["H_Q"], 1, d_out],
            "strides": "contiguous",
        },
        {
            "name": "k_out",
            "dtype": dtype_name,
            "shape": [s["B"], 1, 1, d_out],
            "strides": "contiguous",
        },
    ]

    return KernelCandidate(
        op_name=OP_NAME,
        candidate_id="",
        metal_source=source,
        func_name=func_name,
        inputs_spec=inputs_spec,
        outputs_spec=outputs_spec,
        template_params=dict(template_params),
        launch_params=launch,
        features=features,
        status="new",
        metrics={},
        parent_id=parent_id,
        notes={
            "dtype": dtype_name,
            "shape_signature": shape_signature(s),
            "shape_suite": DEFAULT_SHAPE_SUITE,
        },
    )
