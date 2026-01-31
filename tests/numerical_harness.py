"""Reusable numerical analysis utilities for ZMLX kernel testing.

Goes beyond mx.allclose to provide detailed error statistics and
calibrated tolerance assertions.
"""

from __future__ import annotations

import numpy as np


def numerical_report(actual: np.ndarray, reference: np.ndarray, label: str = "") -> dict:
    """Compute detailed numerical comparison statistics.

    Args:
        actual: Result from the kernel under test (numpy array).
        reference: Ground-truth reference (numpy array).
        label: Optional label for display.

    Returns:
        Dictionary with keys: max_abs_err, mean_abs_err, p99_abs_err,
        max_rel_err, mean_rel_err, p99_rel_err, num_nans, num_infs,
        num_exact, fraction_exact.
    """
    actual = np.asarray(actual, dtype=np.float64).ravel()
    reference = np.asarray(reference, dtype=np.float64).ravel()

    abs_err = np.abs(actual - reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(
            np.abs(reference) > 0, abs_err / np.abs(reference), 0.0
        )

    report = {
        "label": label,
        "max_abs_err": float(np.max(abs_err)) if abs_err.size > 0 else 0.0,
        "mean_abs_err": float(np.mean(abs_err)) if abs_err.size > 0 else 0.0,
        "p99_abs_err": float(np.percentile(abs_err, 99)) if abs_err.size > 0 else 0.0,
        "max_rel_err": float(np.max(rel_err)) if rel_err.size > 0 else 0.0,
        "mean_rel_err": float(np.mean(rel_err)) if rel_err.size > 0 else 0.0,
        "p99_rel_err": float(np.percentile(rel_err, 99)) if rel_err.size > 0 else 0.0,
        "num_nans": int(np.sum(np.isnan(actual))),
        "num_infs": int(np.sum(np.isinf(actual))),
        "num_exact": int(np.sum(actual == reference)),
        "fraction_exact": float(np.mean(actual == reference)) if actual.size > 0 else 0.0,
    }
    return report


def assert_numerical_quality(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    max_abs_tol: float,
    mean_abs_tol: float,
    max_rel_tol: float = 1.0,
    label: str = "",
    print_report: bool = True,
) -> dict:
    """Assert numerical quality with detailed diagnostics on failure.

    Args:
        actual: Result from the kernel under test.
        reference: Ground-truth reference.
        max_abs_tol: Maximum absolute error tolerance.
        mean_abs_tol: Mean absolute error tolerance.
        max_rel_tol: Maximum relative error tolerance (default 1.0 = disabled).
        label: Optional label for display.
        print_report: Whether to print the full report on failure.

    Returns:
        The numerical report dictionary.

    Raises:
        AssertionError: If any tolerance is exceeded.
    """
    report = numerical_report(actual, reference, label=label)

    failures = []
    if report["max_abs_err"] > max_abs_tol:
        failures.append(
            f"max_abs_err {report['max_abs_err']:.6e} > tol {max_abs_tol:.6e}"
        )
    if report["mean_abs_err"] > mean_abs_tol:
        failures.append(
            f"mean_abs_err {report['mean_abs_err']:.6e} > tol {mean_abs_tol:.6e}"
        )
    if report["max_rel_err"] > max_rel_tol:
        failures.append(
            f"max_rel_err {report['max_rel_err']:.6e} > tol {max_rel_tol:.6e}"
        )
    if report["num_nans"] > 0:
        failures.append(f"found {report['num_nans']} NaN values")
    if report["num_infs"] > 0:
        failures.append(f"found {report['num_infs']} Inf values")

    if failures:
        msg_parts = [f"Numerical quality check FAILED{f' [{label}]' if label else ''}:"]
        msg_parts.extend(f"  - {f}" for f in failures)
        if print_report:
            msg_parts.append("Full report:")
            for k, v in report.items():
                if k != "label":
                    msg_parts.append(f"  {k}: {v}")
        raise AssertionError("\n".join(msg_parts))

    return report
