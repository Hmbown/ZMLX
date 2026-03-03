#!/usr/bin/env python3
"""Run reproducible model-level transfer probes for MoE combine variants.

This script runs isolated subprocess sweeps through `bench_iso_variant_sweep.py`
for:
1. Qwen fp32 combine variants.
2. GLM fp32 combine variants.

It writes a rollup summary JSON that highlights fidelity and decode outcomes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUITES: dict[str, list[str]] = {
    "qwen3": ["qwen_combine_fp32", "qwen_combine_fp32_no_fma"],
    "glm47": ["glm_combine_fp32", "glm_combine_fp32_no_fma"],
}


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _run_suite(
    *,
    repo_root: Path,
    iso_script: Path,
    suite: str,
    variants: list[str],
    runs: int,
    max_tokens: int,
    output_dir: Path,
    child_prefix: str,
    ledger: str,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir_arg = _safe_relpath(output_dir, repo_root)
    cmd_json = [
        "python",
        "benchmarks/bench_iso_variant_sweep.py",
        "--suite",
        suite,
        "--runs",
        str(runs),
        "--max-tokens",
        str(max_tokens),
        "--prefix",
        child_prefix,
        "--output-dir",
        output_dir_arg,
        "--ledger",
        ledger,
        "--variants",
        *variants,
    ]
    cmd = [
        sys.executable,
        str(iso_script),
        "--suite",
        suite,
        "--runs",
        str(runs),
        "--max-tokens",
        str(max_tokens),
        "--prefix",
        child_prefix,
        "--output-dir",
        str(output_dir),
        "--ledger",
        ledger,
        "--variants",
        *variants,
    ]
    summary_path = output_dir / f"{child_prefix}_summary.json"
    if dry_run:
        return {
            "suite": suite,
            "return_code": 0,
            "command": cmd_json,
            "summary_path": _safe_relpath(summary_path, repo_root),
            "results": [],
            "dry_run": True,
        }

    proc = subprocess.run(cmd, cwd=repo_root)
    out: dict[str, Any] = {
        "suite": suite,
        "return_code": proc.returncode,
        "command": cmd_json,
        "summary_path": _safe_relpath(summary_path, repo_root),
        "results": [],
        "dry_run": False,
    }
    if proc.returncode != 0 or not summary_path.exists():
        out["error"] = "child sweep failed or summary missing"
        return out

    child = json.loads(summary_path.read_text(encoding="utf-8"))
    for entry in child.get("results", []):
        metrics = entry.get("metrics")
        item: dict[str, Any] = {
            "variant": entry.get("variant"),
            "return_code": entry.get("return_code"),
            "capsule": entry.get("capsule"),
        }
        if not metrics:
            item["error"] = entry.get("error", "metrics missing")
            out["results"].append(item)
            continue

        fidelity = metrics.get("fidelity", {})
        matched = int(
            fidelity.get("matched_tokens", fidelity.get("matched", 0)) or 0
        )
        total = int(fidelity.get("total_tokens", fidelity.get("total", 0)) or 0)
        verdict = str(fidelity.get("verdict", "")).upper()
        if verdict in {"PASS", "FAIL"}:
            fidelity_pass = verdict == "PASS"
        else:
            fidelity_pass = bool(total > 0 and matched == total)
        decode_speedup = _parse_float(metrics.get("decode", {}).get("speedup"))
        prefill_change = _parse_float(metrics.get("prefill", {}).get("change"))
        mem_base = _parse_float(metrics.get("memory_gb", {}).get("baseline"))
        mem_patch = _parse_float(metrics.get("memory_gb", {}).get("patched"))
        item["metrics"] = {
            "fidelity_pass": fidelity_pass,
            "matched_tokens": matched,
            "total_tokens": total,
            "verdict": verdict or None,
            "decode_speedup": decode_speedup,
            "prefill_change": prefill_change,
            "baseline_peak_mem_gb": mem_base,
            "patched_peak_mem_gb": mem_patch,
            "delta_peak_mem_gb": (
                (mem_patch - mem_base)
                if mem_base is not None and mem_patch is not None
                else None
            ),
        }
        out["results"].append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated Qwen/GLM moe_combine transfer probes and write rollup."
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Output prefix for child summaries/capsules and rollup "
            "(default: moe_combine_transfer_probe_<UTC timestamp>)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/repro_capsules",
        help="Directory for child summaries/capsules and rollup summary.",
    )
    parser.add_argument(
        "--ledger",
        default="benchmarks/matrix.jsonl",
        help="Ledger passed to child sweeps (set to '' to disable ledger writes).",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip Qwen probe sweep.",
    )
    parser.add_argument(
        "--skip-glm",
        action="store_true",
        help="Skip GLM probe sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned child commands into summary without executing them.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    iso_script = Path(__file__).resolve().parent / "bench_iso_variant_sweep.py"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.prefix or f"moe_combine_transfer_probe_{timestamp}"

    suites: list[str] = []
    if not args.skip_qwen:
        suites.append("qwen3")
    if not args.skip_glm:
        suites.append("glm47")
    if not suites:
        raise SystemExit("Nothing to run: both --skip-qwen and --skip-glm were set.")

    suite_reports: list[dict[str, Any]] = []
    for suite in suites:
        child_prefix = f"{prefix}_{suite}"
        variants = SUITES[suite]
        print(f"\n=== {suite} ===")
        report = _run_suite(
            repo_root=repo_root,
            iso_script=iso_script,
            suite=suite,
            variants=variants,
            runs=int(args.runs),
            max_tokens=int(args.max_tokens),
            output_dir=output_dir,
            child_prefix=child_prefix,
            ledger=str(args.ledger),
            dry_run=bool(args.dry_run),
        )
        suite_reports.append(report)

    rollup = {
        "timestamp_utc": timestamp,
        "runs": int(args.runs),
        "max_tokens": int(args.max_tokens),
        "output_dir": _safe_relpath(output_dir, repo_root),
        "ledger": args.ledger,
        "suites": suite_reports,
    }
    rollup_path = output_dir / f"{prefix}_summary.json"
    rollup_path.write_text(json.dumps(rollup, indent=2), encoding="utf-8")

    print(f"\nWrote rollup: {_safe_relpath(rollup_path, repo_root)}")
    for suite_report in suite_reports:
        suite = suite_report["suite"]
        if suite_report.get("error"):
            print(f"- {suite}: ERROR ({suite_report['error']})")
            continue
        for row in suite_report.get("results", []):
            metrics = row.get("metrics")
            if not metrics:
                print(f"- {suite}/{row.get('variant')}: error={row.get('error')}")
                continue
            fidelity = "PASS" if metrics["fidelity_pass"] else "FAIL"
            speed = metrics["decode_speedup"]
            print(
                f"- {suite}/{row['variant']}: "
                f"fidelity={fidelity} "
                f"({metrics['matched_tokens']}/{metrics['total_tokens']}), "
                f"decode_speedup={speed}"
            )


if __name__ == "__main__":
    main()
