"""Fusion pass orchestration for phase-1/2/3 JIT fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .graph import FusedOp, Graph
from .rules import (
    BroadcastMulReduceRule,
    ElementwiseChainRule,
    FusionRule,
    QuantizedSharedInputSwiGLURule,
)


@dataclass(frozen=True, slots=True)
class FusionPassResult:
    """Result of running fusion rules over a graph."""

    fused_ops: tuple[FusedOp, ...]
    consumed_nodes: frozenset[int]


def run_phase1_fusion(
    graph: Graph,
    rules: Sequence[FusionRule] | None = None,
) -> FusionPassResult:
    """Apply only phase-1 elementwise rules in priority order."""
    active_rules = _sort_rules(rules or (ElementwiseChainRule(),))
    return _run_fusion_with_rules(graph, active_rules)


def run_fusion(
    graph: Graph,
    rules: Sequence[FusionRule] | None = None,
) -> FusionPassResult:
    """Apply phase-1/2/3 rules in priority order and return fused descriptors."""
    active_rules = _sort_rules(
        rules
        or (
            BroadcastMulReduceRule(),
            QuantizedSharedInputSwiGLURule(),
            ElementwiseChainRule(),
        )
    )
    return _run_fusion_with_rules(graph, active_rules)


def _run_fusion_with_rules(
    graph: Graph,
    active_rules: Sequence[FusionRule],
) -> FusionPassResult:
    consumed: set[int] = set()
    fused_ops: list[FusedOp] = []

    for node_id in graph.topological_order():
        if node_id in consumed:
            continue

        for rule in active_rules:
            match = rule.match(graph, start=node_id, consumed=consumed)
            if match is None:
                continue
            fused_ops.append(rule.fuse(graph, match))
            consumed.update(match.node_ids)
            break

    return FusionPassResult(
        fused_ops=tuple(fused_ops),
        consumed_nodes=frozenset(consumed),
    )


def _sort_rules(rules: Sequence[FusionRule]) -> list[FusionRule]:
    """Sort rules by descending priority and then by rule name."""
    return sorted(rules, key=lambda rule: (-rule.priority, rule.name))
