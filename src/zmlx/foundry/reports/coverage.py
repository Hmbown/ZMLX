"""Coverage report generation.

Scans a session's NDJSON attempt logs and produces per-op coverage
summaries (attempts, ok rate, dtypes seen, shape classes seen, best
latency) plus detailed per-combo breakdowns.

Adapted from zmlx-kernel-foundry's ``reports/coverage.py``.  The main
generalization is that this version does not depend on a live op registry;
instead it derives op metadata purely from the attempt records.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..ndjson import iter_records

# ---------------------------------------------------------------------------
# Attempt iteration
# ---------------------------------------------------------------------------

def _iter_attempts(session_dir: Path):
    """Yield attempt records from all NDJSON files in the session."""
    # Single-file layout
    single = session_dir / "attempts.ndjson"
    if single.exists():
        yield from iter_records(single)

    # Per-worker shards
    for p in sorted(session_dir.glob("attempts.worker*.ndjson")):
        yield from iter_records(p)

    # Sub-directory layout (zmlx-kernel-foundry style)
    attempts_dir = session_dir / "attempts"
    if attempts_dir.is_dir():
        for p in sorted(attempts_dir.glob("*.ndjson")):
            yield from iter_records(p)


def _infer_shape_class(op: str, shape: dict[str, int]) -> str:
    """Lightweight shape class string when the op registry is unavailable."""
    b = shape.get("batch", 1)
    s = shape.get("seq", shape.get("tokens", 1))
    h = shape.get("hidden", 0)
    parts = [f"b{b}", f"s{s}", f"h{h}"]
    if not bool(shape.get("contiguous", True)):
        parts.append("strided")
    else:
        parts.append("contig")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Build coverage dict
# ---------------------------------------------------------------------------

def build_coverage(session_dir: Path) -> dict[str, Any]:
    """Compute coverage statistics for a session.

    Returns a dict with keys:

    - ``session_dir``: str
    - ``ops``: per-op summary dicts
    - ``combos``: sorted list of per-combo dicts
    """
    per_op: dict[str, dict[str, Any]] = {}
    by_key: dict[tuple, dict[str, Any]] = defaultdict(
        lambda: {"attempts": 0, "ok": 0, "best_latency_ms": None},
    )

    for att in _iter_attempts(session_dir):
        op = att.get("op")
        if not op:
            continue

        # Lazily initialize per_op entry
        if op not in per_op:
            kc = att.get("kernel_class", "?")
            per_op[op] = {
                "kernel_class": kc,
                "attempts": 0,
                "ok": 0,
                "best_template_latency_ms": None,
                "dtypes_seen": set(),
                "shape_classes_seen": set(),
                "templates_seen": set(),
            }

        dtype = att.get("dtype", "?")
        t_id = att.get("template_id", att.get("kernel", {}).get("template_id", "ref"))
        shp = att.get("shape", {})
        sc = att.get("shape_class") or _infer_shape_class(op, shp)

        per_op[op]["attempts"] += 1
        per_op[op]["dtypes_seen"].add(dtype)
        per_op[op]["shape_classes_seen"].add(sc)
        per_op[op]["templates_seen"].add(t_id)

        # Determine success -- support both Foundry and DataFoundry record layouts
        res = att.get("result", {})
        status = res.get("status")
        correctness_ok = att.get("correctness", {}).get("ok", False)
        bench_ok = att.get("bench", {}).get("ok", False)
        is_ok = (status == "ok") or (correctness_ok and bench_ok)

        key = (op, dtype, sc, t_id)
        by_key[key]["attempts"] += 1

        if is_ok:
            per_op[op]["ok"] += 1
            by_key[key]["ok"] += 1

            # Extract latency (multiple possible record shapes)
            lat = (
                res.get("template_latency_ms")
                or att.get("bench", {}).get("latency_ms", {}).get("p50")
            )
            if isinstance(lat, (int, float)):
                cur = by_key[key]["best_latency_ms"]
                by_key[key]["best_latency_ms"] = lat if cur is None else min(cur, lat)
                cur2 = per_op[op]["best_template_latency_ms"]
                per_op[op]["best_template_latency_ms"] = lat if cur2 is None else min(cur2, lat)

    # Finalize sets -> sorted lists for JSON serialization
    for d in per_op.values():
        d["dtypes_seen"] = sorted(list(d["dtypes_seen"]))
        d["shape_classes_seen"] = sorted(list(d["shape_classes_seen"]))
        d["templates_seen"] = sorted(list(d["templates_seen"]))

    combos: list[dict[str, Any]] = []
    for (op, dtype, sc, t_id), d in by_key.items():
        combos.append({
            "op": op,
            "dtype": dtype,
            "shape_class": sc,
            "template_id": t_id,
            "attempts": d["attempts"],
            "ok": d["ok"],
            "best_latency_ms": d["best_latency_ms"],
        })

    combos.sort(key=lambda x: (x["op"], x["dtype"], x["shape_class"], x["template_id"]))

    return {
        "session_dir": str(session_dir),
        "ops": per_op,
        "combos": combos,
    }


# ---------------------------------------------------------------------------
# Write coverage reports (JSON + Markdown)
# ---------------------------------------------------------------------------

def write_coverage_reports(session_dir: Path) -> tuple[Path, Path]:
    """Generate and write coverage reports to the session directory.

    Returns (markdown_path, json_path).
    """
    cov = build_coverage(session_dir)

    out_json = session_dir / "coverage.json"
    out_md = session_dir / "coverage.md"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cov, f, indent=2, ensure_ascii=False)

    # -- Markdown report ---------------------------------------------------
    lines: list[str] = []
    lines.append(f"# Coverage Report -- {session_dir.name}\n")
    lines.append("## Ops\n")
    lines.append("| op | class | attempts | ok | best_latency_ms | dtypes | shape_classes | templates |")
    lines.append("|---|---|---:|---:|---:|---|---|---|")

    for op, d in sorted(cov["ops"].items()):
        best = d["best_template_latency_ms"]
        best_s = f"{best:.4f}" if isinstance(best, (int, float)) else "-"
        sc_list = d["shape_classes_seen"]
        sc_str = ", ".join(sc_list[:3])
        if len(sc_list) > 3:
            sc_str += "..."
        lines.append(
            f"| {op} | {d['kernel_class']} | {d['attempts']} | {d['ok']}"
            f" | {best_s}"
            f" | {', '.join(d['dtypes_seen'])}"
            f" | {sc_str}"
            f" | {', '.join(d['templates_seen'])} |"
        )

    # Group ops by kernel class
    lines.append("\n## Kernel classes\n")
    by_class: dict[str, list[str]] = defaultdict(list)
    for op, d in cov["ops"].items():
        by_class[d["kernel_class"]].append(op)

    for kc in sorted(by_class.keys()):
        ops = sorted(by_class[kc])
        lines.append(f"### Class {kc}\n")
        lines.append(", ".join(ops))
        lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_md, out_json
