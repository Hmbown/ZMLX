"""Reporting utilities for kernel discovery runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import KernelCandidate


def pareto_frontier(candidates: list[KernelCandidate]) -> list[KernelCandidate]:
    """Latency-error Pareto frontier (min latency, min error)."""
    bench = [c for c in candidates if c.status == "benchmarked"]
    bench.sort(
        key=lambda c: (
            float(c.metrics.get("latency_us", float("inf"))),
            float(c.metrics.get("correctness_max_abs_err", float("inf"))),
        )
    )

    frontier: list[KernelCandidate] = []
    best_err = float("inf")
    for cand in bench:
        err = float(cand.metrics.get("correctness_max_abs_err", float("inf")))
        if err <= best_err:
            frontier.append(cand)
            best_err = err
    return frontier


def knob_notes(candidates: list[KernelCandidate]) -> list[str]:
    bench = [c for c in candidates if c.status == "benchmarked"]
    if len(bench) < 4:
        return ["Not enough benchmarked candidates to infer knob effects."]

    bench.sort(key=lambda c: float(c.metrics.get("latency_us", float("inf"))))
    top = bench[: max(1, len(bench) // 4)]
    rest = bench[max(1, len(bench) // 4) :]
    if not rest:
        return ["Not enough non-top candidates to compare knob effects."]

    notes: list[str] = []
    keys = sorted(top[0].template_params)
    for key in keys:
        top_avg = sum(float(c.template_params.get(key, 0.0)) for c in top) / len(top)
        rest_avg = sum(float(c.template_params.get(key, 0.0)) for c in rest) / len(rest)
        if abs(top_avg - rest_avg) > 1e-6:
            notes.append(f"`{key}`: top average={top_avg:.3f}, rest average={rest_avg:.3f}")

    lkeys = sorted(top[0].launch_params)
    for key in lkeys:
        if key == "launch_kind":
            continue
        top_avg = sum(float(c.launch_params.get(key, 0.0)) for c in top) / len(top)
        rest_avg = sum(float(c.launch_params.get(key, 0.0)) for c in rest) / len(rest)
        if abs(top_avg - rest_avg) > 1e-6:
            notes.append(f"`launch.{key}`: top average={top_avg:.3f}, rest average={rest_avg:.3f}")

    if not notes:
        notes.append("No single knob dominated top-candidate behavior.")
    return notes


def write_markdown_report(
    *,
    out_path: Path,
    op_name: str,
    seed: int,
    budget: int,
    dtype_name: str,
    shape_suite: str,
    candidates: list[KernelCandidate],
) -> None:
    bench = [c for c in candidates if c.status == "benchmarked"]
    bench.sort(key=lambda c: float(c.metrics.get("latency_us", float("inf"))))
    top = bench[:10]
    frontier = pareto_frontier(candidates)

    lines: list[str] = []
    lines.append(f"# Kernel Discovery Report: `{op_name}`")
    lines.append("")
    lines.append(f"- Seed: `{seed}`")
    lines.append(f"- Budget: `{budget}`")
    lines.append(f"- DType: `{dtype_name}`")
    lines.append(f"- Shape suite: `{shape_suite}`")
    lines.append(f"- Candidates total: `{len(candidates)}`")
    lines.append(f"- Benchmarked: `{len(bench)}`")
    lines.append("")

    lines.append("## Top 10 by Latency")
    lines.append("")
    lines.append("| Rank | Candidate | Latency (us) | Speedup vs ref | Max abs err |")
    lines.append("|:--|:--|--:|--:|--:|")
    for i, cand in enumerate(top, start=1):
        lines.append(
            "| "
            f"{i} | `{cand.candidate_id}` | "
            f"{float(cand.metrics.get('latency_us', float('inf'))):.3f} | "
            f"{float(cand.metrics.get('speedup_vs_ref', 0.0)):.3f} | "
            f"{float(cand.metrics.get('correctness_max_abs_err', float('inf'))):.3e} |"
        )
    if not top:
        lines.append("| - | - | - | - | - |")

    lines.append("")
    lines.append("## Pareto Frontier (Latency vs Error)")
    lines.append("")
    lines.append("| Candidate | Latency (us) | Max abs err | Max rel err | Speedup |")
    lines.append("|:--|--:|--:|--:|--:|")
    for cand in frontier:
        lines.append(
            "| "
            f"`{cand.candidate_id}` | "
            f"{float(cand.metrics.get('latency_us', float('inf'))):.3f} | "
            f"{float(cand.metrics.get('correctness_max_abs_err', float('inf'))):.3e} | "
            f"{float(cand.metrics.get('correctness_max_rel_err', float('inf'))):.3e} | "
            f"{float(cand.metrics.get('speedup_vs_ref', 0.0)):.3f} |"
        )
    if not frontier:
        lines.append("| - | - | - | - | - |")

    lines.append("")
    lines.append("## Knob Notes")
    lines.append("")
    for note in knob_notes(candidates):
        lines.append(f"- {note}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_kernels_payload(
    candidates: list[KernelCandidate],
    *,
    runtime_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bench = [c for c in candidates if c.status == "benchmarked"]
    if not bench:
        return {
            "schema_version": "2",
            "entries": [],
            "runtime": runtime_env or {},
        }
    bench = sorted(bench, key=lambda cand: cand.candidate_id)

    def _shape_key(shape_sig: dict[str, Any]) -> str:
        return json.dumps(shape_sig, sort_keys=True, separators=(",", ":"))

    best_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cand in bench:
        dtype_name = str(cand.metrics.get("dtype", cand.notes.get("dtype", "float16")))
        per_shape = cand.metrics.get("per_shape", [])
        if not isinstance(per_shape, list) or not per_shape:
            per_shape = [{"shape": cand.notes.get("shape_signature", {})}]

        for case in per_shape:
            if not isinstance(case, dict):
                continue
            shape_sig = case.get("shape", {})
            if not isinstance(shape_sig, dict):
                continue

            shape_latency = float(case.get("latency_us", cand.metrics.get("latency_us", float("inf"))))
            entry_key = (cand.op_name, dtype_name, _shape_key(shape_sig))
            current = best_by_key.get(entry_key)
            if current is not None:
                cur_latency = float(current["latency_us"])
                cur_err_raw = current.get("correctness_max_abs_err")
                cur_err = float("inf") if cur_err_raw is None else float(cur_err_raw)
                cand_err_raw = case.get("max_abs_err", cand.metrics.get("correctness_max_abs_err"))
                cand_err = float("inf") if cand_err_raw is None else float(cand_err_raw)
                if shape_latency > cur_latency:
                    continue
                if shape_latency == cur_latency:
                    if cand_err > cur_err:
                        continue
                    if cand_err == cur_err and cand.candidate_id >= current["candidate"].candidate_id:
                        continue

            best_by_key[entry_key] = {
                "candidate": cand,
                "shape_signature": shape_sig,
                "dtype": dtype_name,
                "latency_us": shape_latency,
                "speedup_vs_ref": case.get("speedup_vs_ref", cand.metrics.get("speedup_vs_ref")),
                "correctness_max_abs_err": case.get(
                    "max_abs_err", cand.metrics.get("correctness_max_abs_err")
                ),
                "correctness_max_rel_err": case.get(
                    "max_rel_err", cand.metrics.get("correctness_max_rel_err")
                ),
            }

    entries: list[dict[str, Any]] = []
    env = runtime_env or {}
    for op_name, dtype_name, shape_sig_key in sorted(best_by_key):
        _ = shape_sig_key
        selected = best_by_key[(op_name, dtype_name, shape_sig_key)]
        cand = selected["candidate"]
        entries.append(
            {
                "key": {
                    "op_name": op_name,
                    "mlx_version": str(env.get("mlx_version", "unknown")),
                    "device_arch": str(env.get("device_arch", "unknown")),
                    "device_name": str(env.get("device_name", "unknown")),
                    "dtype": dtype_name,
                    "shape_signature": selected["shape_signature"],
                },
                "candidate_id": cand.candidate_id,
                "func_name": cand.func_name,
                "metal_source": cand.metal_source,
                "inputs_spec": cand.inputs_spec,
                "outputs_spec": cand.outputs_spec,
                "template_params": cand.template_params,
                "launch_params": cand.launch_params,
                "source_hash": cand.source_hash,
                "metrics": {
                    "latency_us": selected["latency_us"],
                    "speedup_vs_ref": selected["speedup_vs_ref"],
                    "correctness_max_abs_err": selected["correctness_max_abs_err"],
                    "correctness_max_rel_err": selected["correctness_max_rel_err"],
                },
            }
        )

    return {
        "schema_version": "2",
        "runtime": env,
        "entries": entries,
    }


def write_best_kernels(
    path: Path,
    candidates: list[KernelCandidate],
    *,
    runtime_env: dict[str, Any] | None = None,
) -> None:
    payload = best_kernels_payload(candidates, runtime_env=runtime_env)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_best_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge best-kernel payloads, keeping the best latency per key."""
    merged: dict[str, dict[str, Any]] = {}
    runtime: dict[str, Any] = {}

    def _key_str(entry: dict[str, Any]) -> str:
        key = entry.get("key", {})
        return json.dumps(key, sort_keys=True, separators=(",", ":"))

    for payload in payloads:
        runtime = payload.get("runtime", runtime)

        # v2 shape
        entries = payload.get("entries", [])
        if not entries and "ops" in payload:
            # v1 fallback conversion
            for op_name, op_entry in payload.get("ops", {}).items():
                for shape_sig in op_entry.get("shape_signatures", [{}]):
                    entries.append(
                        {
                            "key": {
                                "op_name": op_name,
                                "mlx_version": "unknown",
                                "device_arch": "unknown",
                                "device_name": "unknown",
                                "dtype": (op_entry.get("supported_dtypes") or ["float16"])[0],
                                "shape_signature": shape_sig,
                            },
                            **{k: v for k, v in op_entry.items() if k != "shape_signatures"},
                        }
                    )

        for entry in entries:
            key = _key_str(entry)
            cur = merged.get(key)
            if cur is None:
                merged[key] = entry
                continue
            cur_lat = float(cur.get("metrics", {}).get("latency_us", float("inf")))
            new_lat = float(entry.get("metrics", {}).get("latency_us", float("inf")))
            if new_lat < cur_lat:
                merged[key] = entry

    return {
        "schema_version": "2",
        "runtime": runtime,
        "entries": list(merged.values()),
    }


def load_ndjson(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records
