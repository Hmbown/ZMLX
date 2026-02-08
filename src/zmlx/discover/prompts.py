"""Prompt building and response parsing for LLM-guided kernel search."""

from __future__ import annotations

import re
from typing import Any

from .candidates import SearchSpace
from .tree import Node

SYSTEM_PROMPT = """\
You are an expert Metal Shading Language (MSL) kernel engineer optimizing GPU \
kernels for Apple Silicon.

You will be given a Metal kernel and asked to produce improved variants. \
Each variant should target a specific optimization strategy.

## MSL Constraints
- Metal uses `thread_position_in_grid`, `thread_position_in_threadgroup`, etc.
- SIMD width is 32 on all Apple GPUs.
- `threadgroup_barrier(mem_flags::mem_threadgroup)` for threadgroup sync.
- `simd_sum`, `simd_max`, `simd_min` for SIMD-level reductions.
- Template parameter `T` is the input/output dtype.
- Cast to `float` for computation, cast back to `(T)` for output.

## Optimization Strategies
- **Vectorization**: Use `float4`/`half4` loads to increase memory throughput.
- **Memory coalescing**: Ensure sequential threads access sequential memory.
- **Loop unrolling**: Manually unroll small loops for ILP.
- **SIMD intrinsics**: Use `simd_sum`, `simd_shuffle` for warp-level ops.
- **Threadgroup memory**: Use shared memory to reduce global memory traffic.
- **Reduced bank conflicts**: Pad shared memory to avoid bank conflicts.
- **Register pressure**: Balance register usage vs occupancy.
- **Instruction-level parallelism**: Interleave independent operations.

## Output Format
For each variant, produce a block separated by `---VARIANT---`:

```
---VARIANT---
REASONING: <1-2 sentences explaining the optimization strategy>
THREADGROUP: <x, y, z> (threadgroup dimensions)
SOURCE:
<Metal kernel source code — just the kernel body, no function signature>
```

Produce exactly the requested number of variants. Each variant MUST have \
different source code from all others.
"""


def build_generation_prompt(
    search_space: SearchSpace,
    parent_source: str,
    parent_timing_us: float,
    best_timing_us: float,
    history_summary: str,
    device_info: str,
    n_candidates: int = 8,
) -> str:
    """Build the user prompt for generating kernel variants."""
    parts = [
        f"## Target: {search_space.name}",
        f"{search_space.description}",
        "",
        "## I/O Specification",
    ]
    for ispec in search_space.input_specs:
        parts.append(f"- Input `{ispec.name}`: {ispec.shape_expr} ({ispec.dtype})")
    for ospec in search_space.output_specs:
        parts.append(f"- Output `{ospec.name}`: {ospec.shape_expr} ({ospec.dtype})")

    if search_space.constraints:
        parts.append("")
        parts.append("## Constraints")
        for c in search_space.constraints:
            parts.append(f"- {c}")

    parts.append("")
    parts.append("## Grid Configuration")
    parts.append(f"{search_space.grid_fn}")

    parts.append("")
    parts.append("## Device")
    parts.append(device_info)

    parts.append("")
    parts.append("## Current Best Source")
    parts.append(f"Timing: {parent_timing_us:.1f} us (best so far: {best_timing_us:.1f} us)")
    parts.append("```metal")
    parts.append(parent_source)
    parts.append("```")

    if history_summary:
        parts.append("")
        parts.append("## History (what worked / what didn't)")
        parts.append(history_summary)

    parts.append("")
    parts.append(f"Produce {n_candidates} improved variants. "
                 "Each must be a complete kernel body (no function signature). "
                 "Focus on strategies not yet tried in the history.")

    return "\n".join(parts)


def build_history_summary(evaluated_nodes: list[Node], max_entries: int = 20) -> str:
    """Build a summary of evaluated candidates for the LLM prompt."""
    if not evaluated_nodes:
        return ""

    # Sort by reward descending
    sorted_nodes = sorted(
        [n for n in evaluated_nodes if n.eval_result is not None],
        key=lambda n: n.eval_result.reward if n.eval_result else 0.0,
        reverse=True,
    )[:max_entries]

    lines = []
    for i, node in enumerate(sorted_nodes):
        er = node.eval_result
        if er is None:
            continue
        status = "OK" if er.correct else "FAIL"
        reasoning = node.candidate.llm_reasoning[:80] if node.candidate.llm_reasoning else "n/a"
        lines.append(
            f"{i + 1}. [{status}] reward={er.reward:.2f} "
            f"speedup={er.speedup:.2f}x "
            f"median={er.median_us:.1f}us — {reasoning}"
        )

    return "\n".join(lines)


_VARIANT_SEP = re.compile(r"---VARIANT---", re.IGNORECASE)
_REASONING_RE = re.compile(r"REASONING:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_THREADGROUP_RE = re.compile(r"THREADGROUP:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", re.IGNORECASE)
_SOURCE_RE = re.compile(r"SOURCE:\s*\n(.*?)(?=\n---VARIANT---|$)", re.IGNORECASE | re.DOTALL)


def parse_llm_response(raw_response: str) -> list[dict[str, Any]]:
    """Parse the LLM response into a list of variant dicts.

    Each dict has keys: ``source``, ``reasoning``, ``threadgroup``.
    """
    # Split on variant separator
    blocks = _VARIANT_SEP.split(raw_response)
    # First block before the first separator is preamble — skip it
    variant_blocks = [b.strip() for b in blocks[1:] if b.strip()] if len(blocks) > 1 else []

    # If no separators found, try treating the whole response as one variant
    if not variant_blocks and raw_response.strip():
        variant_blocks = [raw_response.strip()]

    results = []
    for block in variant_blocks:
        reasoning = ""
        threadgroup = (256, 1, 1)
        source = ""

        rm = _REASONING_RE.search(block)
        if rm:
            reasoning = rm.group(1).strip()

        tm = _THREADGROUP_RE.search(block)
        if tm:
            threadgroup = (int(tm.group(1)), int(tm.group(2)), int(tm.group(3)))

        sm = _SOURCE_RE.search(block)
        if sm:
            source = sm.group(1).strip()
            # Strip code fences if present
            source = re.sub(r"^```(?:metal)?\s*\n?", "", source)
            source = re.sub(r"\n?```\s*$", "", source)

        if source:
            results.append({
                "source": source,
                "reasoning": reasoning,
                "threadgroup": threadgroup,
            })

    return results
