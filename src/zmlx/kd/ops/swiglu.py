"""SwiGLU operator definition for kernel discovery."""

from __future__ import annotations

import math
from typing import Any

from ..types import KernelCandidate
from . import dtype_itemsize, mx_dtype, render_template, template_text

OP_NAME = "swiglu"
DEFAULT_DTYPE = "float16"
DEFAULT_SHAPE_SUITE = "glm_flash_small"

SHAPE_SUITES: dict[str, list[dict[str, int]]] = {
    "glm_flash_small": [
        {"rows": 128, "D": 1536},
        {"rows": 256, "D": 1536},
        {"rows": 512, "D": 1536},
        {"rows": 1024, "D": 1536},
    ],
    "glm_flash_decode": [
        {"rows": 1, "D": 1536},
        {"rows": 2, "D": 1536},
        {"rows": 4, "D": 1536},
    ],
    "qwen30b_decode": [
        {"rows": 1, "D": 2048},
        {"rows": 2, "D": 2048},
        {"rows": 4, "D": 2048},
    ],
    "lfm2_decode": [
        {"rows": 1, "D": 7168},
        {"rows": 2, "D": 7168},
        {"rows": 4, "D": 7168},
    ],
}

TEMPLATE_PARAM_SPACE: dict[str, tuple[Any, ...]] = {
    "vec_width": (1, 2, 4),
    "unroll": (1, 2, 4, 8),
    "fast_sigmoid": (False, True),
}

LAUNCH_PARAM_SPACE: dict[str, tuple[Any, ...]] = {
    "threadgroup_x": (64, 128, 256, 512),
}


def seed_template_params() -> dict[str, Any]:
    return {
        "vec_width": 1,
        "unroll": 1,
        "fast_sigmoid": False,
    }


def seed_launch_params() -> dict[str, Any]:
    return {
        "threadgroup_x": 256,
        "launch_kind": "swiglu_flat",
    }


def normalize_shape(shape: dict[str, Any]) -> dict[str, int]:
    return {
        "rows": int(shape["rows"]),
        "D": int(shape["D"]),
    }


def shape_signature(shape: dict[str, Any]) -> dict[str, int]:
    s = normalize_shape(shape)
    return {
        "rows": s["rows"],
        "D": s["D"],
        "N": s["rows"] * s["D"],
    }


def bytes_moved(shape: dict[str, Any], dtype_name: str) -> int:
    s = normalize_shape(shape)
    n = s["rows"] * s["D"]
    item = dtype_itemsize(dtype_name)
    return int((n + n + n) * item)


def arithmetic_intensity(shape: dict[str, Any], dtype_name: str) -> float:
    s = normalize_shape(shape)
    n = s["rows"] * s["D"]
    flops = n * 8.0
    b = bytes_moved(s, dtype_name)
    return float(flops / max(1.0, float(b)))


def make_inputs(mx_mod: Any, shape: dict[str, Any], dtype_name: str, seed: int) -> list[Any]:
    s = normalize_shape(shape)
    mx_mod.random.seed(seed)
    dt = mx_dtype(mx_mod, dtype_name)
    gate = mx_mod.random.normal((s["rows"], s["D"])).astype(dt)
    up = mx_mod.random.normal((s["rows"], s["D"])).astype(dt)
    return [gate.reshape(-1), up.reshape(-1)]


def reference(mx_mod: Any, inputs: list[Any], shape: dict[str, Any], dtype_name: str) -> list[Any]:
    _ = (shape, dtype_name)
    gate, up = inputs
    g32 = gate.astype(mx_mod.float32)
    u32 = up.astype(mx_mod.float32)
    out = (g32 * mx_mod.sigmoid(g32) * u32).astype(gate.dtype)
    return [out]


def output_shapes(shape: dict[str, Any]) -> list[tuple[int, ...]]:
    s = normalize_shape(shape)
    return [(s["rows"] * s["D"],)]


def output_dtypes(mx_mod: Any, dtype_name: str) -> list[Any]:
    return [mx_dtype(mx_mod, dtype_name)]


def compute_launch(
    launch_params: dict[str, Any],
    shape: dict[str, Any],
    _inputs: list[Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    s = normalize_shape(shape)
    n = s["rows"] * s["D"]
    vec = int(launch_params.get("vec_width", 1))
    unroll = int(launch_params.get("unroll", 1))
    per_thread = max(1, vec * unroll)
    grid_x = int(math.ceil(n / per_thread))
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
    n = s["rows"] * s["D"]
    vec = int(template_params["vec_width"])
    unroll = int(template_params["unroll"])
    tg = int(launch_params["threadgroup_x"])

    params = {
        "N": n,
        "VEC": vec,
        "UNROLL": unroll,
        "FAST_SIGMOID": "true" if bool(template_params["fast_sigmoid"]) else "false",
    }
    source = render_template(template_text("swiglu.metal.tmpl"), params)

    func_name = (
        f"kk_kd_swiglu_n{n}_tg{tg}_v{vec}_u{unroll}_"
        f"fast{1 if params['FAST_SIGMOID'] == 'true' else 0}"
    )

    launch = dict(launch_params)
    launch["launch_kind"] = "swiglu_flat"
    launch["vec_width"] = vec
    launch["unroll"] = unroll

    b = bytes_moved(s, dtype_name)
    features = {
        "bytes_moved_est": float(b),
        "arithmetic_intensity_est": arithmetic_intensity(s, dtype_name),
        "launch_occupancy_proxy": float(min(1.0, tg / 1024.0)),
    }

    inputs_spec = [
        {"name": "gate", "dtype": dtype_name, "shape": [n], "strides": "contiguous"},
        {"name": "up", "dtype": dtype_name, "shape": [n], "strides": "contiguous"},
    ]
    outputs_spec = [
        {"name": "out", "dtype": dtype_name, "shape": [n], "strides": "contiguous"}
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
