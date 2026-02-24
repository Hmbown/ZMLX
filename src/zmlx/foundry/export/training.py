"""JSONL training data export.

Filters a session's attempts down to successful, correct, benchmarked
records and exports them as JSONL with the kernel source inlined.
This is the primary format consumed by downstream training pipelines.

Adapted from mlx-kernel-lab's ``export/training_export.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..ndjson import iter_records


def export_training_jsonl(
    session_dir: str,
    out_dir: str,
    *,
    ops: list[str] | None = None,
    min_p50_ms: float | None = None,
    max_p50_ms: float | None = None,
) -> dict[str, Any]:
    """Export successful attempts as JSONL for training.

    Each output record contains:
    - ``id``: attempt identifier
    - ``op``, ``dtype``, ``shape``, ``spec``: problem specification
    - ``template_id``, ``knobs``: kernel configuration
    - ``source``: Metal kernel source (from ``kernels/<id>.metal``)
    - ``bench``: benchmark metrics

    Parameters
    ----------
    session_dir : str
        Path to the session directory.
    out_dir : str
        Output directory for the JSONL file.
    ops : list of str, optional
        If given, only export attempts for these ops.
    min_p50_ms : float, optional
        Exclude attempts faster than this (filter out measurement noise).
    max_p50_ms : float, optional
        Exclude attempts slower than this (filter out regressions).

    Returns
    -------
    dict
        ``{"out_path": str, "n_records": int, "n_skipped": int}``.
    """
    session_dir_p = Path(session_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    out_path = out_dir_p / f"{session_dir_p.name}_training.jsonl"
    kernels_dir = session_dir_p / "kernels"

    # Collect all attempt records from all layouts
    all_records = _collect_records(session_dir_p)

    n_written = 0
    n_skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            # Op filter
            rec_op = rec.get("op")
            if ops is not None and rec_op not in ops:
                n_skipped += 1
                continue

            # Success filter (support both layouts)
            if not _is_successful(rec):
                n_skipped += 1
                continue

            # Latency filter
            p50 = _get_p50(rec)
            if p50 is not None:
                if min_p50_ms is not None and p50 < min_p50_ms:
                    n_skipped += 1
                    continue
                if max_p50_ms is not None and p50 > max_p50_ms:
                    n_skipped += 1
                    continue

            # Resolve attempt ID (different field names across layouts)
            aid = rec.get("id") or rec.get("attempt_id", "")

            # Load kernel source if available
            src = ""
            if aid:
                src_path = kernels_dir / f"{aid}.metal"
                if src_path.exists():
                    src = src_path.read_text(encoding="utf-8")

            out_rec: dict[str, Any] = {
                "id": aid,
                "op": rec_op,
                "dtype": rec.get("dtype"),
                "shape": rec.get("shape"),
                "spec": rec.get("spec") or rec.get("op_params"),
                "template_id": (
                    rec.get("template_id")
                    or rec.get("kernel", {}).get("template_id")
                ),
                "knobs": (
                    rec.get("knobs")
                    or rec.get("kernel", {}).get("knobs")
                ),
                "source": src,
                "bench": rec.get("bench") or rec.get("result", {}),
            }

            f.write(json.dumps(
                out_rec,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ))
            f.write("\n")
            n_written += 1

    return {
        "out_path": str(out_path),
        "n_records": n_written,
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_records(session_dir: Path) -> list[dict[str, Any]]:
    """Collect all attempt records from a session (all layout conventions)."""
    records: list[dict[str, Any]] = []

    # Single-file layout
    single = session_dir / "attempts.ndjson"
    if single.exists():
        records.extend(iter_records(single))

    # Worker shards
    for wp in sorted(session_dir.glob("attempts.worker*.ndjson")):
        records.extend(iter_records(wp))

    # Sub-directory layout
    attempts_dir = session_dir / "attempts"
    if attempts_dir.is_dir():
        for fp in sorted(attempts_dir.glob("*.ndjson")):
            records.extend(iter_records(fp))

    return records


def _is_successful(rec: dict[str, Any]) -> bool:
    """Check whether a record represents a successful attempt."""
    # DataFoundry layout
    if (
        rec.get("build", {}).get("ok")
        and rec.get("correctness", {}).get("ok")
        and rec.get("bench", {}).get("ok")
    ):
        return True
    # Foundry layout
    if rec.get("result", {}).get("status") == "ok":
        return True
    return False


def _get_p50(rec: dict[str, Any]) -> float | None:
    """Extract p50 latency from a record."""
    lat = rec.get("bench", {}).get("latency_ms")
    if isinstance(lat, dict):
        p50 = lat.get("p50")
        if p50 is not None:
            return float(p50)
    rlat = rec.get("result", {}).get("template_latency_ms")
    if isinstance(rlat, (int, float)):
        return float(rlat)
    return None
