"""Fusion rule protocol and concrete phase-1/2/3 matchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Set
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .graph import ELEMENTWISE_OPS, FusedOp, Graph, TensorMeta


@dataclass(frozen=True, slots=True)
class FusionMatch:
    """A rule match over one or more graph nodes."""

    node_ids: tuple[int, ...]


@runtime_checkable
class FusionRule(Protocol):
    """Protocol for phase-1 fusion rules."""

    name: str
    priority: int

    def match(
        self,
        graph: Graph,
        *,
        start: int,
        consumed: Set[int],
    ) -> FusionMatch | None:
        """Return a match rooted at ``start`` or ``None`` if not applicable."""

    def fuse(self, graph: Graph, match: FusionMatch) -> FusedOp:
        """Convert a match into a typed ``FusedOp`` descriptor."""


class BaseFusionRule(ABC):
    """Convenience base class for rules with ``priority``."""

    name: str = "base_rule"

    def __init__(self, *, priority: int = 0) -> None:
        self.priority = priority

    @abstractmethod
    def match(
        self,
        graph: Graph,
        *,
        start: int,
        consumed: Set[int],
    ) -> FusionMatch | None: ...

    @abstractmethod
    def fuse(self, graph: Graph, match: FusionMatch) -> FusedOp: ...


class ElementwiseChainRule(BaseFusionRule):
    """Fuse linear chains of elementwise ops into one descriptor."""

    name = "elementwise_chain"

    def __init__(self, *, priority: int = 100, min_length: int = 2) -> None:
        super().__init__(priority=priority)
        self.min_length = min_length

    def match(
        self,
        graph: Graph,
        *,
        start: int,
        consumed: Set[int],
    ) -> FusionMatch | None:
        if start in consumed:
            return None
        if not self._can_start_chain(graph, start=start, consumed=consumed):
            return None

        chain: list[int] = [start]
        current = start

        while True:
            users = graph.users(current)
            if len(users) != 1:
                break

            nxt = users[0]
            if nxt in consumed:
                break

            nxt_node = graph.node(nxt)
            if nxt_node.op not in ELEMENTWISE_OPS:
                break

            elementwise_inputs = [
                input_id
                for input_id in nxt_node.inputs
                if input_id not in consumed and graph.node(input_id).op in ELEMENTWISE_OPS
            ]
            if elementwise_inputs != [current]:
                break

            chain.append(nxt)
            current = nxt

        if len(chain) < self.min_length:
            return None
        return FusionMatch(node_ids=tuple(chain))

    def fuse(self, graph: Graph, match: FusionMatch) -> FusedOp:
        chain_set = set(match.node_ids)
        input_ids: list[int] = []
        seen_inputs: set[int] = set()

        for node_id in match.node_ids:
            for input_id in graph.node(node_id).inputs:
                if input_id in chain_set or input_id in seen_inputs:
                    continue
                seen_inputs.add(input_id)
                input_ids.append(input_id)

        output_id = match.node_ids[-1]
        output_meta = graph.node(output_id).meta
        if output_meta is None:
            raise ValueError(f"Matched output node {output_id} has no metadata.")

        ops = tuple(graph.node(node_id).op for node_id in match.node_ids)
        return FusedOp(
            rule=self.name,
            kind="elementwise_chain",
            node_ids=match.node_ids,
            input_ids=tuple(input_ids),
            output_id=output_id,
            output_meta=output_meta,
            ops=ops,
        )

    def _can_start_chain(
        self,
        graph: Graph,
        *,
        start: int,
        consumed: Set[int],
    ) -> bool:
        node = graph.node(start)
        if node.op not in ELEMENTWISE_OPS:
            return False

        preds = [
            input_id
            for input_id in node.inputs
            if input_id not in consumed and graph.node(input_id).op in ELEMENTWISE_OPS
        ]
        if len(preds) > 1:
            return False
        if len(preds) == 1 and len(graph.users(preds[0])) == 1:
            return False
        return True


def _is_last_axis(axis: int | None, *, rank: int) -> bool:
    if axis is None:
        return False
    axis_norm = int(axis)
    if axis_norm < 0:
        axis_norm += rank
    return axis_norm == rank - 1


def _is_broadcast_mul(
    lhs_meta: TensorMeta | None,
    rhs_meta: TensorMeta | None,
    out_meta: TensorMeta | None,
) -> bool:
    if lhs_meta is None or rhs_meta is None or out_meta is None:
        return False
    if lhs_meta.shape == out_meta.shape and rhs_meta.shape == out_meta.shape:
        return False
    if lhs_meta.is_scalar or rhs_meta.is_scalar:
        return True
    return lhs_meta.shape != rhs_meta.shape


def _unique_external_inputs(graph: Graph, node_ids: tuple[int, ...]) -> tuple[int, ...]:
    fused_set = set(node_ids)
    out: list[int] = []
    seen: set[int] = set()
    for node_id in node_ids:
        node = graph.node(node_id)
        for input_id in node.inputs:
            if input_id in fused_set or input_id in seen:
                continue
            seen.add(input_id)
            out.append(input_id)
    return tuple(out)


class BroadcastMulReduceRule(BaseFusionRule):
    """Fuse ``mul`` followed by a last-axis reduction (phase-2 path)."""

    name = "broadcast_mul_reduce"

    def __init__(self, *, priority: int = 250) -> None:
        super().__init__(priority=priority)

    def match(
        self,
        graph: Graph,
        *,
        start: int,
        consumed: Set[int],
    ) -> FusionMatch | None:
        if start in consumed:
            return None
        mul_node = graph.node(start)
        if mul_node.op != "mul":
            return None
        if any(input_id in consumed for input_id in mul_node.inputs):
            return None

        users = tuple(user for user in graph.users(start) if user not in consumed)
        if len(users) != 1:
            return None

        reduce_id = users[0]
        reduce_node = graph.node(reduce_id)
        if reduce_node.op not in {"sum", "mean", "max"}:
            return None
        if reduce_node.inputs != (start,):
            return None
        if reduce_node.meta is None or mul_node.meta is None:
            return None
        if not _is_last_axis(reduce_node.attrs.get("axis"), rank=len(mul_node.meta.shape)):
            return None

        lhs_meta = graph.node(mul_node.inputs[0]).meta
        rhs_meta = graph.node(mul_node.inputs[1]).meta
        if not _is_broadcast_mul(lhs_meta, rhs_meta, mul_node.meta):
            return None
        return FusionMatch(node_ids=(start, reduce_id))

    def fuse(self, graph: Graph, match: FusionMatch) -> FusedOp:
        mul_id, reduce_id = match.node_ids
        reduce_node = graph.node(reduce_id)
        output_meta = reduce_node.meta
        if output_meta is None:
            raise ValueError(f"Matched reduction node {reduce_id} has no metadata.")

        attrs = {
            "reduce_op": reduce_node.op,
            "axis": int(reduce_node.attrs.get("axis", -1)),
            "keepdims": bool(reduce_node.attrs.get("keepdims", False)),
            "accumulate_dtype": "float32",
        }
        return FusedOp(
            rule=self.name,
            kind="broadcast_mul_reduce",
            node_ids=match.node_ids,
            input_ids=_unique_external_inputs(graph, match.node_ids),
            output_id=reduce_id,
            output_meta=output_meta,
            ops=("mul", reduce_node.op),
            attrs=attrs,
        )


def _decode_friendly_lhs(meta: TensorMeta | None) -> bool:
    if meta is None:
        return False
    if len(meta.shape) == 1:
        return True
    if len(meta.shape) == 2 and int(meta.shape[0]) == 1:
        return True
    return False


def _is_affine_qmm(node_attrs: dict[str, object] | object) -> bool:
    if not isinstance(node_attrs, dict):
        return False
    mode = str(node_attrs.get("mode", ""))
    bits = node_attrs.get("bits")
    transpose = bool(node_attrs.get("transpose", False))
    return mode == "affine" and transpose and bits in (4, 8)


class QuantizedSharedInputSwiGLURule(BaseFusionRule):
    """Fuse shared-input QMM pair with ``silu(gate) * up`` epilogue (phase-3)."""

    name = "quantized_shared_input_swiglu"

    def __init__(self, *, priority: int = 220) -> None:
        super().__init__(priority=priority)

    def match(
        self,
        graph: Graph,
        *,
        start: int,
        consumed: Set[int],
    ) -> FusionMatch | None:
        if start in consumed:
            return None
        gate_qmm = graph.node(start)
        if gate_qmm.op != "quantized_matmul":
            return None
        if not _is_affine_qmm(dict(gate_qmm.attrs)):
            return None
        if not gate_qmm.inputs:
            return None
        lhs_id = gate_qmm.inputs[0]
        if not _decode_friendly_lhs(graph.node(lhs_id).meta):
            return None

        epilogue = self._match_epilogue(graph, start=start, consumed=consumed)
        if epilogue is None:
            return None
        epilogue_node_ids, up_qmm_id, output_mul_id = epilogue

        up_qmm = graph.node(up_qmm_id)
        if up_qmm.op != "quantized_matmul":
            return None
        if not _is_affine_qmm(dict(up_qmm.attrs)):
            return None
        if not up_qmm.inputs or up_qmm.inputs[0] != lhs_id:
            return None
        if up_qmm.meta != gate_qmm.meta:
            return None

        gate_bits = gate_qmm.attrs.get("bits")
        up_bits = up_qmm.attrs.get("bits")
        gate_group = gate_qmm.attrs.get("group_size")
        up_group = up_qmm.attrs.get("group_size")
        if gate_bits != up_bits or gate_group != up_group:
            return None

        return FusionMatch(node_ids=epilogue_node_ids + (up_qmm_id, output_mul_id))

    def _match_epilogue(
        self,
        graph: Graph,
        *,
        start: int,
        consumed: Set[int],
    ) -> tuple[tuple[int, ...], int, int] | None:
        gate_users = tuple(user for user in graph.users(start) if user not in consumed)
        # Canonical form: silu(gate_qmm) * up_qmm
        if len(gate_users) == 1:
            silu_id = gate_users[0]
            silu_node = graph.node(silu_id)
            if silu_node.op == "silu" and silu_node.inputs == (start,):
                silu_users = tuple(user for user in graph.users(silu_id) if user not in consumed)
                if len(silu_users) != 1:
                    return None
                output_mul_id = silu_users[0]
                output_mul = graph.node(output_mul_id)
                if output_mul.op != "mul":
                    return None
                other_inputs = [node_id for node_id in output_mul.inputs if node_id != silu_id]
                if len(other_inputs) != 1:
                    return None
                up_qmm_id = other_inputs[0]
                if up_qmm_id in consumed:
                    return None
                return (start, silu_id), up_qmm_id, output_mul_id

        # Expanded form: (gate_qmm * sigmoid(gate_qmm)) * up_qmm
        sigmoid_ids = [node_id for node_id in gate_users if graph.node(node_id).op == "sigmoid"]
        if len(sigmoid_ids) != 1:
            return None
        sigmoid_id = sigmoid_ids[0]
        gate_mul_ids = [
            node_id
            for node_id in gate_users
            if graph.node(node_id).op == "mul"
            and set(graph.node(node_id).inputs) == {start, sigmoid_id}
        ]
        if len(gate_mul_ids) != 1:
            return None
        gate_mul_id = gate_mul_ids[0]
        gate_mul_users = tuple(user for user in graph.users(gate_mul_id) if user not in consumed)
        if len(gate_mul_users) != 1:
            return None
        output_mul_id = gate_mul_users[0]
        output_mul = graph.node(output_mul_id)
        if output_mul.op != "mul":
            return None
        other_inputs = [node_id for node_id in output_mul.inputs if node_id != gate_mul_id]
        if len(other_inputs) != 1:
            return None
        up_qmm_id = other_inputs[0]
        if up_qmm_id in consumed:
            return None
        return (start, sigmoid_id, gate_mul_id), up_qmm_id, output_mul_id

    def fuse(self, graph: Graph, match: FusionMatch) -> FusedOp:
        gate_qmm_id = match.node_ids[0]
        mul_id = match.node_ids[-1]
        up_candidates = [
            node_id
            for node_id in match.node_ids[1:-1]
            if graph.node(node_id).op == "quantized_matmul"
        ]
        if len(up_candidates) != 1:
            raise ValueError(
                "quantized_shared_input_swiglu expected exactly one up_qmm node in match."
            )
        up_qmm_id = up_candidates[0]
        mul_node = graph.node(mul_id)
        output_meta = mul_node.meta
        if output_meta is None:
            raise ValueError(f"Matched output node {mul_id} has no metadata.")

        gate_qmm = graph.node(gate_qmm_id)
        attrs = {
            "bits": gate_qmm.attrs.get("bits"),
            "group_size": gate_qmm.attrs.get("group_size"),
            "mode": gate_qmm.attrs.get("mode"),
            "transpose": bool(gate_qmm.attrs.get("transpose", False)),
            "gate_qmm_id": gate_qmm_id,
            "up_qmm_id": up_qmm_id,
        }
        return FusedOp(
            rule=self.name,
            kind="quantized_shared_input_swiglu",
            node_ids=match.node_ids,
            input_ids=_unique_external_inputs(graph, match.node_ids),
            output_id=mul_id,
            output_meta=output_meta,
            ops=tuple(graph.node(node_id).op for node_id in match.node_ids),
            attrs=attrs,
        )
