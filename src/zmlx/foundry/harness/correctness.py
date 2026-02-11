"""Correctness checking: error metrics and tolerance gating.

Adapted from DataFoundry's harness/correctness.py.  Provides both
numpy-only and MLX-accelerated metric computation, plus a simple
tolerance gate (``check_pass``).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import tolerances

# ---------------------------------------------------------------------------
# Numpy metric helpers
# ---------------------------------------------------------------------------


def _max_rel_err_np(y: np.ndarray, y_ref: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_ref), 1e-8)
    return float(np.max(np.abs(y - y_ref) / denom))


def _max_abs_err_np(y: np.ndarray, y_ref: np.ndarray) -> float:
    return float(np.max(np.abs(y - y_ref)))


def _ulp32_np(y: np.ndarray, y_ref: np.ndarray) -> float:
    """ULP (unit in the last place) distance for float32."""
    yi = y.astype(np.float32, copy=False).view(np.int32)
    ri = y_ref.astype(np.float32, copy=False).view(np.int32)
    return float(np.max(np.abs(yi - ri)))


# ---------------------------------------------------------------------------
# Public metric computation
# ---------------------------------------------------------------------------


def compute_metrics_numpy(
    y: np.ndarray, y_ref: np.ndarray, dtype: str
) -> tuple[float, float, float | None]:
    """Compute (max_abs_err, max_rel_err, ulp) from numpy arrays."""
    y32 = y.astype(np.float32, copy=False)
    r32 = y_ref.astype(np.float32, copy=False)
    max_abs = _max_abs_err_np(y32, r32)
    max_rel = _max_rel_err_np(y32, r32)
    ulp = _ulp32_np(y32, r32) if dtype == "float32" else None
    return max_abs, max_rel, ulp


def compute_metrics_mlx(
    mx: Any, y: Any, y_ref: Any, dtype: str
) -> tuple[float, float, float | None]:
    """Compute error metrics on-device via MLX and return python scalars."""
    diff = mx.abs(y.astype(mx.float32) - y_ref.astype(mx.float32))
    max_abs = float(mx.max(diff).item())
    denom = mx.maximum(mx.abs(y_ref.astype(mx.float32)), 1e-8)
    max_rel = float(mx.max(diff / denom).item())
    ulp: float | None = None
    return max_abs, max_rel, ulp


# ---------------------------------------------------------------------------
# Tolerance gate
# ---------------------------------------------------------------------------


def check_pass(
    dtype: str, max_abs: float, max_rel: float
) -> tuple[bool, str]:
    """Return ``(passed, reason)`` where *reason* is empty on pass."""
    tol = tolerances(dtype)
    if max_abs <= tol.max_abs and max_rel <= tol.max_rel:
        return True, ""
    return False, (
        f"exceeds_tolerance("
        f"abs={max_abs:.3g}, rel={max_rel:.3g}, "
        f"tol_abs={tol.max_abs:.3g}, tol_rel={tol.max_rel:.3g})"
    )
