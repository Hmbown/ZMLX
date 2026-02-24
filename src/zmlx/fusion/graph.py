"""Graph IR for phase-1/2/3 JIT fusion."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

ELEMENTWISE_UNARY_OPS: frozenset[str] = frozenset(
    {"neg", "exp", "tanh", "sigmoid", "relu", "silu", "log", "sqrt", "rsqrt"}
)
ELEMENTWISE_BINARY_OPS: frozenset[str] = frozenset(
    {"add", "sub", "mul", "div", "maximum", "minimum"}
)
ELEMENTWISE_OPS: frozenset[str] = ELEMENTWISE_UNARY_OPS | ELEMENTWISE_BINARY_OPS
REDUCTION_OPS: frozenset[str] = frozenset({"sum", "mean", "max"})
QUANTIZED_LINEAR_OPS: frozenset[str] = frozenset({"quantized_matmul"})
SUPPORTED_OPS: frozenset[str] = ELEMENTWISE_OPS | REDUCTION_OPS | QUANTIZED_LINEAR_OPS


@dataclass(frozen=True, slots=True)
class TensorMeta:
    """Tensor metadata captured during symbolic tracing."""

    shape: tuple[int, ...]
    dtype: str

    @property
    def is_scalar(self) -> bool:
        return self.shape == ()


@dataclass(frozen=True, slots=True)
class Node:
    """A graph node representing an input, scalar literal, or tensor op."""

    id: int
    op: str
    inputs: tuple[int, ...] = ()
    meta: TensorMeta | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FusedOp:
    """Descriptor emitted by a fusion rule for downstream JIT lowering."""

    rule: str
    kind: str
    node_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    output_id: int
    output_meta: TensorMeta
    ops: tuple[str, ...]
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))


class Graph:
    """Small DAG used by phase-1 symbolic tracing and fusion matching."""

    def __init__(self) -> None:
        self._nodes: dict[int, Node] = {}
        self._order: list[int] = []
        self._users: dict[int, list[int]] = {}
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self) -> Iterator[Node]:
        for node_id in self._order:
            yield self._nodes[node_id]

    def topological_order(self) -> tuple[int, ...]:
        """Return node ids in insertion order (the graph is append-only DAG)."""
        return tuple(self._order)

    def node(self, node_id: int) -> Node:
        """Return node by id."""
        return self._nodes[node_id]

    def users(self, node_id: int) -> tuple[int, ...]:
        """Return all direct consumers of ``node_id``."""
        return tuple(self._users.get(node_id, ()))

    def add_input(self, *, name: str, meta: TensorMeta) -> int:
        """Add an input placeholder."""
        return self._add_node(op="input", inputs=(), meta=meta, attrs={"name": name})

    def add_constant(self, *, value: int | float | bool, meta: TensorMeta) -> int:
        """Add a scalar literal node."""
        return self._add_node(op="constant", inputs=(), meta=meta, attrs={"value": value})

    def add_op(
        self,
        op: str,
        *,
        inputs: Iterable[int],
        meta: TensorMeta,
        attrs: Mapping[str, Any] | None = None,
    ) -> int:
        """Add a tensor op node with shape/dtype metadata."""
        if op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported fusion op: {op}")
        input_ids = tuple(inputs)
        if op in ELEMENTWISE_UNARY_OPS | REDUCTION_OPS:
            expected_arity = 1
            if len(input_ids) != expected_arity:
                raise ValueError(
                    f"Op {op!r} expects {expected_arity} inputs, got {len(input_ids)}"
                )
        elif op in ELEMENTWISE_BINARY_OPS:
            expected_arity = 2
            if len(input_ids) != expected_arity:
                raise ValueError(
                    f"Op {op!r} expects {expected_arity} inputs, got {len(input_ids)}"
                )
        elif op == "quantized_matmul":
            if len(input_ids) < 2 or len(input_ids) > 4:
                raise ValueError(
                    "Op 'quantized_matmul' expects between 2 and 4 inputs "
                    f"(x, w, [scales], [biases]), got {len(input_ids)}"
                )
        else:
            raise ValueError(f"Unsupported fusion op: {op}")
        return self._add_node(op=op, inputs=input_ids, meta=meta, attrs=dict(attrs or {}))

    def _add_node(
        self,
        *,
        op: str,
        inputs: tuple[int, ...],
        meta: TensorMeta | None,
        attrs: dict[str, Any] | None,
    ) -> int:
        for input_id in inputs:
            if input_id not in self._nodes:
                raise KeyError(f"Input node {input_id} does not exist.")

        node_id = self._next_id
        self._next_id += 1

        frozen_attrs: Mapping[str, Any] = MappingProxyType(dict(attrs or {}))
        node = Node(id=node_id, op=op, inputs=inputs, meta=meta, attrs=frozen_attrs)
        self._nodes[node_id] = node
        self._order.append(node_id)
        self._users.setdefault(node_id, [])
        for input_id in inputs:
            self._users.setdefault(input_id, []).append(node_id)
        return node_id
