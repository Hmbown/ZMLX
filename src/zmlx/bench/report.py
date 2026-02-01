"""Read a repro capsule JSON and print a formatted benchmark report.

Usage::

    python -m zmlx.bench.report benchmarks/repro_capsules/lfm2_m1pro_20260131.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _fmt_change(baseline: float, patched: float, neutral_threshold: float = 2.0) -> str:
    if baseline <= 0:
        return "—"
    pct = (patched / baseline - 1) * 100
    if abs(pct) < neutral_threshold:
        return f"{pct:+.1f}% (neutral)"
    return f"**{pct:+.1f}%**"


def _print_model_table(key: str, data: dict) -> None:
    model = data.get("model", key)
    fidelity = data.get("fidelity", {})
    baseline = data.get("baseline", {})
    patched = data.get("patched", {})

    matched = fidelity.get("matched", 0)
    total = fidelity.get("total", 0)
    verdict = fidelity.get("verdict", "?")

    b_decode = baseline.get("median_decode", 0)
    p_decode = patched.get("median_decode", 0)
    b_prefill = baseline.get("median_prefill", 0)
    p_prefill = patched.get("median_prefill", 0)
    peak_mem = patched.get("peak_mem_gb", 0)

    print(f"\n  {model}")
    print(f"  {'Metric':<14} {'Baseline':>12} {'Patched':>12} {'Change':>18}")
    print(f"  {'-' * 58}")
    print(f"  {'Decode':<14} {b_decode:>10.1f}  {p_decode:>10.1f}  {_fmt_change(b_decode, p_decode):>18}")
    print(
        f"  {'Prefill':<14} {b_prefill:>10.1f}  {p_prefill:>10.1f}  "
        f"{_fmt_change(b_prefill, p_prefill, neutral_threshold=3.0):>18}"
    )
    fidelity_str = f"{matched}/{total}"
    print(f"  {'Fidelity':<14} {'':>12} {fidelity_str:>12}  {verdict:>18}")
    if peak_mem > 0:
        print(f"  {'Peak memory':<14} {'':>12} {peak_mem:>9.2f} GB")


def report(path: str | Path) -> None:
    """Load a repro capsule JSON and print a formatted report."""
    path = Path(path)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        capsule = json.load(f)

    meta = capsule.get("meta", {})

    print("=" * 62)
    print("  ZMLX Benchmark Report")
    print("=" * 62)
    print(f"  Date:    {meta.get('date', '?')}")
    print(f"  Device:  {meta.get('device', '?')} ({meta.get('memory_gb', '?')} GB)")
    print(f"  macOS:   {meta.get('macos', '?')}")
    print(f"  MLX:     {meta.get('mlx_version', '?')}")
    print(f"  ZMLX:    {meta.get('zmlx_version', '?')}")
    print(f"  Python:  {meta.get('python', '?')}")
    print(f"  Commit:  {meta.get('git_commit', '?')[:7]}")

    if "command_4bit" in meta:
        print("\n  Commands:")
        for k, v in meta.items():
            if k.startswith("command"):
                print(f"    $ {v}")

    # Print each model section (skip 'meta')
    for key, data in capsule.items():
        if key == "meta" or not isinstance(data, dict):
            continue
        _print_model_table(key, data)

    if meta.get("note"):
        print(f"\n  Note: {meta['note']}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m zmlx.bench.report",
        description="Print a formatted benchmark report from a repro capsule JSON.",
    )
    parser.add_argument("capsule", help="Path to repro capsule JSON file")
    args = parser.parse_args()
    report(args.capsule)


if __name__ == "__main__":
    main()
