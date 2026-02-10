"""Kernel-discovery operator definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def template_text(filename: str) -> str:
    base = Path(__file__).resolve().parent.parent / "templates"
    return (base / filename).read_text(encoding="utf-8")


def render_template(template: str, params: dict[str, Any]) -> str:
    out = template
    for key, value in params.items():
        out = out.replace(f"{{{{{key}}}}}", str(value))
    return out


def dtype_itemsize(dtype_name: str) -> int:
    return {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }[dtype_name]


def mx_dtype(mx: Any, dtype_name: str) -> Any:
    mapping = {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }
    return mapping[dtype_name]


def dtype_name(dtype: Any) -> str:
    s = str(dtype)
    if "bfloat16" in s:
        return "bfloat16"
    if "float16" in s:
        return "float16"
    if "float32" in s:
        return "float32"
    return s


def _build_ops() -> dict[str, Any]:
    from . import rmsnorm, rmsnorm_residual, rope, swiglu

    return {
        "rmsnorm_residual": rmsnorm_residual,
        "rope": rope,
        "swiglu": swiglu,
        "rmsnorm": rmsnorm,
    }


OPS = _build_ops()


def get_op(op_name: str):
    if op_name not in OPS:
        raise KeyError(f"Unknown op {op_name!r}. Available: {', '.join(sorted(OPS))}")
    return OPS[op_name]
