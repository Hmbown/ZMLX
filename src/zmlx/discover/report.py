"""Reporting utilities for kernel search sessions."""

from __future__ import annotations

from pathlib import Path

from .session import Session
from .tree import SearchTree


def print_report(session: Session, top_n: int = 5) -> None:
    """Print a formatted terminal report of the search session."""
    meta = session.metadata
    print(f"\n{'=' * 60}")
    print("  ZMLX Discover — Session Report")
    print(f"{'=' * 60}")
    print(f"  Target:     {meta.target_name}")
    print(f"  Backend:    {meta.llm_backend}")
    print(f"  Device:     {meta.device_chip} ({meta.device_memory_gb}GB)")
    print(f"  Steps:      {meta.total_steps}")
    print(f"  Candidates: {meta.total_candidates} generated, {meta.total_evaluated} evaluated")
    print(f"  Baseline:   {meta.baseline_us:.1f} us")
    print(f"  Best:       {meta.best_speedup:.2f}x ({meta.best_reward:.2f} reward)")
    print()

    if not session.tree_data:
        print("  (no tree data)")
        return

    tree = SearchTree.from_dict(session.tree_data)
    evaluated = sorted(
        tree.evaluated_nodes,
        key=lambda n: n.eval_result.reward if n.eval_result else 0.0,
        reverse=True,
    )[:top_n]

    if not evaluated:
        print("  (no evaluated candidates)")
        return

    print(f"  Top {min(top_n, len(evaluated))} Candidates:")
    print(f"  {'Rank':>4s}  {'Speedup':>8s}  {'Median us':>10s}  {'Reward':>7s}  Strategy")
    print(f"  {'-' * 4}  {'-' * 8}  {'-' * 10}  {'-' * 7}  {'-' * 30}")

    for i, node in enumerate(evaluated):
        er = node.eval_result
        if er is None:
            continue
        reasoning = node.candidate.llm_reasoning[:40] if node.candidate.llm_reasoning else "n/a"
        print(
            f"  {i + 1:4d}  {er.speedup:7.2f}x  {er.median_us:10.1f}  "
            f"{er.reward:7.2f}  {reasoning}"
        )

    print(f"\n{'=' * 60}\n")


def export_kernel(session: Session, output_path: str | Path) -> None:
    """Write the best kernel to a Python file in ZMLX style."""
    meta = session.metadata
    if not meta.best_source:
        print("No best kernel to export.")
        return

    # Extract I/O names from the best node in the tree
    input_names = ["inp"]
    output_names = ["out"]
    if session.tree_data:
        tree = SearchTree.from_dict(session.tree_data)
        best = tree.best_node()
        if best.candidate.spec.input_names:
            input_names = list(best.candidate.spec.input_names)
        if best.candidate.spec.output_names:
            output_names = list(best.candidate.spec.output_names)

    input_names_str = repr(input_names)
    output_names_str = repr(output_names)

    code = f'''"""Auto-generated kernel from ZMLX Discover.

Target: {meta.target_name}
Speedup: {meta.best_speedup:.2f}x
Device: {meta.device_chip}
Session: {meta.session_id}
"""

from __future__ import annotations

from functools import cache
from typing import Any

from zmlx.metal import kernel as metal_kernel
from zmlx.msl import DEFAULT_HEADER


@cache
def _discovered_kernel() -> Any:
    """Build the discovered kernel."""
    source = """{meta.best_source}"""

    return metal_kernel(
        name="kk_discovered_{meta.target_name}",
        input_names={input_names_str},
        output_names={output_names_str},
        source=source,
        header=DEFAULT_HEADER,
        cache=True,
    )
'''

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code)
    print(f"Exported kernel to {p}")
