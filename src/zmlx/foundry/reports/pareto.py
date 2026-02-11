"""Pareto front extraction from session attempt data.

Identifies the Pareto-optimal set of kernel variants (non-dominated on
latency and correctness error) and provides helper utilities for finding
the single best kernel by p50 latency.

Adapted from mlx-kernel-lab's ``reports/pareto.py``.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..ndjson import iter_records

# ---------------------------------------------------------------------------
# Attempt loading
# ---------------------------------------------------------------------------

def load_attempts(session_dir: str) -> list[dict[str, Any]]:
    """Load all attempt records from a session directory."""
    p = Path(session_dir)
    out: list[dict[str, Any]] = []

    # Single-file layout
    single = p / "attempts.ndjson"
    if single.exists():
        out.extend(iter_records(single))

    # Worker shards
    for wp in sorted(p.glob("attempts.worker*.ndjson")):
        out.extend(iter_records(wp))

    # Sub-directory layout
    attempts_dir = p / "attempts"
    if attempts_dir.is_dir():
        for fp in sorted(attempts_dir.glob("*.ndjson")):
            out.extend(iter_records(fp))

    return out


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _is_successful(rec: dict[str, Any]) -> bool:
    """Check whether a record represents a successful attempt.

    Supports both DataFoundry layout (build/correctness/bench nested dicts)
    and Foundry layout (result.status == "ok").
    """
    # DataFoundry layout
    if rec.get("build", {}).get("ok") and rec.get("correctness", {}).get("ok") and rec.get("bench", {}).get("ok"):
        return True
    # Foundry layout
    if rec.get("result", {}).get("status") == "ok":
        return True
    return False


def _get_p50(rec: dict[str, Any]) -> float | None:
    """Extract p50 latency from a record (multiple layout conventions)."""
    # DataFoundry: bench.latency_ms.p50
    lat = rec.get("bench", {}).get("latency_ms")
    if isinstance(lat, dict):
        p50 = lat.get("p50")
        if p50 is not None:
            return float(p50)
    # Foundry: result.template_latency_ms
    rlat = rec.get("result", {}).get("template_latency_ms")
    if isinstance(rlat, (int, float)):
        return float(rlat)
    return None


# ---------------------------------------------------------------------------
# Best kernel by p50
# ---------------------------------------------------------------------------

def best_kernel_by_p50(
    attempts: Iterable[dict[str, Any]],
    *,
    op: str | None = None,
) -> dict[str, Any] | None:
    """Return the fastest successful attempt by p50 latency.

    Parameters
    ----------
    attempts : iterable of dicts
        Attempt records.
    op : str, optional
        If given, filter to attempts for this op only.
    """
    best: dict[str, Any] | None = None
    best_p50: float | None = None

    for rec in attempts:
        if op is not None and rec.get("op") != op:
            continue
        if not _is_successful(rec):
            continue
        p50 = _get_p50(rec)
        if p50 is None:
            continue
        if best is None or p50 < best_p50:  # type: ignore[operator]
            best = rec
            best_p50 = p50

    return best


# ---------------------------------------------------------------------------
# Pareto front extraction
# ---------------------------------------------------------------------------

def extract_pareto_front(
    attempts: Iterable[dict[str, Any]],
    *,
    op: str | None = None,
    objectives: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract the Pareto-optimal subset of successful attempts.

    By default, minimizes p50 latency and maximizes correctness quality
    (lower max_abs_err is better).  Custom objectives can be specified.

    Parameters
    ----------
    attempts : iterable of dicts
        Attempt records.
    op : str, optional
        Filter to this op.
    objectives : list of str, optional
        Objective keys to extract from records.  Each objective is
        minimized.  Defaults to ``["p50_latency_ms", "max_abs_err"]``.

    Returns
    -------
    list of dict
        The Pareto-non-dominated records.
    """
    if objectives is None:
        objectives = ["p50_latency_ms", "max_abs_err"]

    # Collect (objective_vector, record) pairs for successful attempts
    candidates: list[tuple[list[float], dict[str, Any]]] = []

    for rec in attempts:
        if op is not None and rec.get("op") != op:
            continue
        if not _is_successful(rec):
            continue

        obj_vec: list[float] = []
        valid = True
        for obj_key in objectives:
            val = _extract_objective(rec, obj_key)
            if val is None:
                valid = False
                break
            obj_vec.append(val)

        if valid:
            candidates.append((obj_vec, rec))

    if not candidates:
        return []

    # Classic Pareto filter: a point is dominated if another point is
    # <= on all objectives and < on at least one.
    n_obj = len(objectives)
    pareto: list[dict[str, Any]] = []

    for i, (vec_i, rec_i) in enumerate(candidates):
        dominated = False
        for j, (vec_j, _) in enumerate(candidates):
            if i == j:
                continue
            # Check if j dominates i (all <= and at least one <)
            all_leq = all(vec_j[k] <= vec_i[k] for k in range(n_obj))
            any_lt = any(vec_j[k] < vec_i[k] for k in range(n_obj))
            if all_leq and any_lt:
                dominated = True
                break
        if not dominated:
            pareto.append(rec_i)

    return pareto


def _extract_objective(rec: dict[str, Any], key: str) -> float | None:
    """Extract a named objective value from a record."""
    if key == "p50_latency_ms":
        return _get_p50(rec)
    if key == "max_abs_err":
        corr = rec.get("correctness", {})
        mae = corr.get("max_abs_err")
        if mae is not None:
            return float(mae)
        # Foundry layout: result.max_abs_err
        mae2 = rec.get("result", {}).get("max_abs_err")
        if mae2 is not None:
            return float(mae2)
        # If correctness is ok but no error metric, use 0.0
        if corr.get("ok") or rec.get("result", {}).get("status") == "ok":
            return 0.0
        return None
    if key == "max_rel_err":
        corr = rec.get("correctness", {})
        mre = corr.get("max_rel_err")
        if mre is not None:
            return float(mre)
        return None
    # Generic: try to find key in top-level, result, or bench
    for loc in [rec, rec.get("result", {}), rec.get("bench", {}), rec.get("correctness", {})]:
        if key in loc:
            try:
                return float(loc[key])
            except (TypeError, ValueError):
                pass
    return None
