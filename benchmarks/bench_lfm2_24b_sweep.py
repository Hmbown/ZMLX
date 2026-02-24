#!/usr/bin/env python3
"""LFM2-24B-A2B variant sweep: gate mode × combine mode matrix.

Runs each variant in an isolated subprocess to avoid cross-contamination.
Produces repro capsule JSON for each variant.

Usage:
    # Full matrix (9 variants, ~20 min)
    python benchmarks/bench_lfm2_24b_sweep.py

    # Quick fidelity gate only (200 tokens, 1 run)
    python benchmarks/bench_lfm2_24b_sweep.py --quick

    # Single variant
    python benchmarks/bench_lfm2_24b_sweep.py --gate native --combine native

    # Baseline only (no patching)
    python benchmarks/bench_lfm2_24b_sweep.py --baseline-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MODEL = "LiquidAI/LFM2-24B-A2B-MLX-4bit"
CAPSULE_DIR = Path("benchmarks/repro_capsules")

GATE_MODES = ["kernel", "native", "fast64"]
COMBINE_MODES = ["kernel", "native", "fp32_no_fma"]


def run_variant(
    *,
    gate: str | None,
    combine: str | None,
    max_tokens: int,
    runs: int,
    label: str,
    patterns: list[str] | None = None,
) -> dict:
    """Run a single variant via zmlx.validate in a subprocess."""
    env = os.environ.copy()
    if gate is not None:
        env["ZMLX_LFM2_GATE_MODE"] = gate
    if combine is not None:
        env["ZMLX_LFM2_COMBINE_MODE"] = combine
    # Force moe_mlp pattern for LFM2 (normally auto-excluded by perf risk)
    env["ZMLX_PATCH_ALLOW_PERF_RISK"] = "1"

    cmd = [
        sys.executable, "-m", "zmlx.validate", MODEL,
        "--max-tokens", str(max_tokens),
        "--runs", str(runs),
        "--verbose",
    ]
    if patterns is not None:
        cmd.extend(["--patterns"] + patterns)

    print(f"\n{'='*70}")
    print(f"  Variant: {label}")
    print(f"  Gate: {gate or 'N/A'}, Combine: {combine or 'N/A'}")
    print(f"  Tokens: {max_tokens}, Runs: {runs}")
    print(f"  Patterns: {patterns or 'default'}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t0

    # Parse output
    out = result.stdout + result.stderr
    print(out[-3000:] if len(out) > 3000 else out)

    return {
        "label": label,
        "gate_mode": gate,
        "combine_mode": combine,
        "patterns": patterns,
        "max_tokens": max_tokens,
        "runs": runs,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def parse_validate_output(text: str) -> dict:
    """Extract key metrics from zmlx.validate output."""
    metrics = {}
    for line in text.split("\n"):
        line = line.strip()
        if "fidelity" in line.lower():
            metrics["fidelity_line"] = line
        if "matched" in line.lower() and "/" in line:
            # e.g. "Fidelity: 200/200 matched"
            parts = line.split()
            for p in parts:
                if "/" in p:
                    try:
                        a, b = p.split("/")
                        metrics["matched"] = int(a)
                        metrics["total"] = int(b)
                    except ValueError:
                        pass
        if "decode" in line.lower() and "tok/s" in line.lower():
            metrics["decode_line"] = line
        if "speedup" in line.lower() or "ratio" in line.lower():
            metrics["speedup_line"] = line
        if "median" in line.lower():
            metrics["median_line"] = line
    return metrics


def main():
    parser = argparse.ArgumentParser(description="LFM2-24B variant sweep")
    parser.add_argument("--gate", choices=GATE_MODES, help="Single gate mode")
    parser.add_argument("--combine", choices=COMBINE_MODES, help="Single combine mode")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Quick fidelity check (200 tok, 1 run)")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.max_tokens = 200
        args.runs = 1

    CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []

    # 1. Baseline (no patching) — always run first
    if not args.skip_baseline:
        r = run_variant(
            gate=None,
            combine=None,
            max_tokens=args.max_tokens,
            runs=args.runs,
            label="baseline_unpatched",
            patterns=["none"],  # no patterns = baseline
        )
        r["metrics"] = parse_validate_output(r["stdout"] + r["stderr"])
        results.append(r)

    if args.baseline_only:
        _print_summary(results)
        return

    # 2. Variant matrix
    if args.gate and args.combine:
        # Single variant
        variants = [(args.gate, args.combine)]
    elif args.gate:
        variants = [(args.gate, c) for c in COMBINE_MODES]
    elif args.combine:
        variants = [(g, args.combine) for g in GATE_MODES]
    else:
        # Full matrix
        variants = [(g, c) for g in GATE_MODES for c in COMBINE_MODES]

    for gate, combine in variants:
        label = f"lfm2_24b_g{gate}_c{combine}"
        r = run_variant(
            gate=gate,
            combine=combine,
            max_tokens=args.max_tokens,
            runs=args.runs,
            label=label,
            patterns=["moe_mlp"],
        )
        r["metrics"] = parse_validate_output(r["stdout"] + r["stderr"])
        results.append(r)

    # Save combined results
    capsule_path = CAPSULE_DIR / f"lfm2_24b_sweep_{timestamp}.json"
    with open(capsule_path, "w") as f:
        json.dump(
            {
                "model": MODEL,
                "timestamp": timestamp,
                "variants": [
                    {k: v for k, v in r.items() if k not in ("stdout", "stderr")}
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nCapsule saved: {capsule_path}")

    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Label':<35} {'Fidelity':<12} {'Return':<8}")
    print(f"{'-'*35} {'-'*12} {'-'*8}")
    for r in results:
        m = r.get("metrics", {})
        matched = m.get("matched", "?")
        total = m.get("total", "?")
        fid = f"{matched}/{total}"
        rc = r["returncode"]
        print(f"{r['label']:<35} {fid:<12} {rc:<8}")


if __name__ == "__main__":
    main()
