"""Curriculum scheduler for staged dataset generation.

Organizes kernel ops into stages ordered from cheap/simple to
expensive/complex, allowing generation runs to progressively unlock
harder ops.  This mirrors LLM training curricula: start with easy
examples, add complexity as the model improves.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Default 5-stage curriculum
# ---------------------------------------------------------------------------

DEFAULT_CURRICULUM: list[list[str]] = [
    # Stage 0: cheapest elementwise / fused norms
    ["rmsnorm", "swiglu", "rope"],
    # Stage 1: more expensive attention-adjacent ops
    ["softmax", "layernorm"],
    # Stage 2: quantization primitives
    ["dequantize", "quantize"],
    # Stage 3: data-movement / KV cache
    ["gather", "scatter", "kv_append"],
    # Stage 4: MoE routing + dispatch (most expensive, largest kernel families)
    ["moe_topk", "moe_pack_assignments", "moe_dispatch_gather", "moe_combine_scatter"],
]


def flatten_curriculum(curriculum: list[list[str]] | None = None) -> list[str]:
    """Flatten a nested curriculum into a single ordered list of op names."""
    if curriculum is None:
        curriculum = DEFAULT_CURRICULUM
    out: list[str] = []
    for stage in curriculum:
        out.extend(stage)
    return out


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

@dataclass
class CurriculumScheduler:
    """Controls which ops are available at a given curriculum stage.

    Parameters
    ----------
    ops : list of str
        Requested ops (from config / CLI).  The scheduler intersects these
        with the curriculum to determine what is actually available.
    curriculum : list of list of str, optional
        Override the default 5-stage curriculum.
    stage : int
        Current stage (0-indexed).  Stages 0..stage are unlocked.
    """

    ops: list[str]
    curriculum: list[list[str]] = None  # type: ignore[assignment]
    stage: int = 0

    def __post_init__(self) -> None:
        if self.curriculum is None:
            self.curriculum = DEFAULT_CURRICULUM

    @property
    def n_stages(self) -> int:
        return len(self.curriculum)

    def set_stage(self, stage: int) -> None:
        """Set the current curriculum stage (clamped to valid range)."""
        self.stage = max(0, min(int(stage), self.n_stages - 1))

    def available_ops(self) -> list[str]:
        """Return ops unlocked at the current stage.

        The result is the intersection of requested ``ops`` and all ops
        in curriculum stages 0 through ``stage`` (inclusive), preserving
        curriculum order.  Falls back to all requested ops if the
        intersection would be empty.
        """
        allowed: list[str] = []
        for i in range(min(self.stage + 1, self.n_stages)):
            for op in self.curriculum[i]:
                if op in self.ops and op not in allowed:
                    allowed.append(op)
        if not allowed:
            # Fallback: no curriculum overlap, return all requested ops
            return list(self.ops)
        return allowed

    def stage_for_op(self, op: str) -> int | None:
        """Return the curriculum stage that introduces *op*, or None."""
        for i, stage_ops in enumerate(self.curriculum):
            if op in stage_ops:
                return i
        return None

    def summary(self) -> dict[str, object]:
        """Diagnostic summary for logging."""
        return {
            "n_stages": self.n_stages,
            "current_stage": self.stage,
            "requested_ops": self.ops,
            "available_ops": self.available_ops(),
            "curriculum": self.curriculum,
        }
