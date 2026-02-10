"""RMSNorm operator definition for kernel discovery."""

from __future__ import annotations

from typing import Any

from ..types import KernelCandidate
from . import dtype_itemsize, mx_dtype, render_template, template_text

OP_NAME = "rmsnorm"
DEFAULT_DTYPE = "float16"
DEFAULT_SHAPE_SUITE = "glm_flash_small"

SHAPE_SUITES: dict[str, list[dict[str, int]]] = {
    "glm_flash_small": [
        {"rows": 128, "D": 2048},
        {"rows": 256, "D": 2048},
        {"rows": 512, "D": 2048},
        {"rows": 1024, "D": 2048},
    ],
    "glm_flash_decode": [
        {"rows": 1, "D": 2048},
        {"rows": 2, "D": 2048},
        {"rows": 4, "D": 2048},
    ],
}

TEMPLATE_PARAM_SPACE: dict[str, tuple[Any, ...]] = {
    "vec_width": (1, 2, 4),
    "unroll": (1, 2, 4),
    "use_simd": (True, False),
    "use_threadgroup_cache": (False, True),
    "eps": (1e-6,),
}

LAUNCH_PARAM_SPACE: dict[str, tuple[Any, ...]] = {
    "threadgroup_x": (64, 128, 256, 512),
}


def seed_template_params() -> dict[str, Any]:
    return {
        "vec_width": 1,
        "unroll": 1,
        "use_simd": True,
        "use_threadgroup_cache": False,
        "eps": 1e-6,
    }


def seed_launch_params() -> dict[str, Any]:
    return {
        "threadgroup_x": 256,
        "launch_kind": "rmsnorm_rows_tg",
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
    }


def bytes_moved(shape: dict[str, Any], dtype_name: str) -> int:
    s = normalize_shape(shape)
    item = dtype_itemsize(dtype_name)
    rows = s["rows"]
    d = s["D"]
    return int((rows * d + d + rows * d) * item)


def arithmetic_intensity(shape: dict[str, Any], dtype_name: str) -> float:
    s = normalize_shape(shape)
    rows = s["rows"]
    d = s["D"]
    flops = rows * d * 5.0
    b = bytes_moved(s, dtype_name)
    return float(flops / max(1.0, float(b)))


def make_inputs(mx_mod: Any, shape: dict[str, Any], dtype_name: str, seed: int) -> list[Any]:
    s = normalize_shape(shape)
    rows = s["rows"]
    d = s["D"]
    mx_mod.random.seed(seed)
    dt = mx_dtype(mx_mod, dtype_name)
    x = mx_mod.random.normal((rows, d)).astype(dt)
    w = mx_mod.random.normal((d,)).astype(dt)
    return [x, w]


def reference(mx_mod: Any, inputs: list[Any], shape: dict[str, Any], dtype_name: str) -> list[Any]:
    _ = (shape, dtype_name)
    x, w = inputs
    x32 = x.astype(mx_mod.float32)
    w32 = w.astype(mx_mod.float32)
    inv = mx_mod.rsqrt(mx_mod.mean(x32 * x32, axis=-1, keepdims=True) + 1e-6)
    out = (x32 * inv * w32).astype(x.dtype)
    return [out]


def output_shapes(shape: dict[str, Any]) -> list[tuple[int, ...]]:
    s = normalize_shape(shape)
    return [(s["rows"], s["D"])]


def output_dtypes(mx_mod: Any, dtype_name: str) -> list[Any]:
    return [mx_dtype(mx_mod, dtype_name)]


def compute_launch(
    launch_params: dict[str, Any],
    shape: dict[str, Any],
    _inputs: list[Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    s = normalize_shape(shape)
    tg = int(launch_params["threadgroup_x"])
    return (s["rows"] * tg, 1, 1), (tg, 1, 1)


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
        "D": s["D"],
        "TG": tg,
        "VEC": int(template_params["vec_width"]),
        "UNROLL": int(template_params["unroll"]),
        "EPS": float(template_params["eps"]),
        "USE_SIMD": "true" if bool(template_params["use_simd"]) else "false",
        "USE_TG_CACHE": "true" if bool(template_params["use_threadgroup_cache"]) else "false",
    }
    source = render_template(template_text("rmsnorm.metal.tmpl"), params)
    func_name = (
        f"kk_kd_rmsnorm_d{s['D']}_tg{tg}_v{params['VEC']}_u{params['UNROLL']}_"
        f"simd{1 if params['USE_SIMD'] == 'true' else 0}_"
        f"cache{1 if params['USE_TG_CACHE'] == 'true' else 0}"
    )

    launch = dict(launch_params)
    launch["launch_kind"] = "rmsnorm_rows_tg"

    b = bytes_moved(s, dtype_name)
    features = {
        "bytes_moved_est": float(b),
        "arithmetic_intensity_est": arithmetic_intensity(s, dtype_name),
        "launch_occupancy_proxy": float(min(1.0, tg / 1024.0)),
    }

    inputs_spec = [
        {"name": "inp", "dtype": dtype_name, "shape": [s["rows"], s["D"]], "strides": "contiguous"},
        {"name": "weight", "dtype": dtype_name, "shape": [s["D"]], "strides": "contiguous"},
    ]
    outputs_spec = [
        {"name": "out", "dtype": dtype_name, "shape": [s["rows"], s["D"]], "strides": "contiguous"}
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
