"""Kernel IR v1 — Typed, inspectable representations for ZMLX kernel patterns.

Each IR node is a frozen dataclass that carries all parameters needed to
produce Metal Shading Language source via ``to_source()``.  The critical
invariant is:

    ir_node.to_source() == codegen.function(same_params)

This is enforced by ``tests/test_ir.py``.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass
from typing import Any


class PatternKind(enum.Enum):
    """Discriminator for IR node dispatch."""

    ELEMENTWISE_UNARY = "elementwise_unary"
    ELEMENTWISE_BINARY = "elementwise_binary"
    ROWWISE_REDUCTION = "rowwise_reduction"
    ROWWISE_PARALLEL_REDUCTION = "rowwise_parallel_reduction"
    ROWWISE_MAPREDUCE = "rowwise_mapreduce"
    TILED_2D = "tiled_2d"
    RAW = "raw"


@dataclass(frozen=True)
class VJPSpec:
    """Backward pass specification for a differentiable kernel."""

    bwd_source: str
    bwd_input_names: tuple[str, ...]
    bwd_output_names: tuple[str, ...]
    use_output: bool = True
    prelude: str = ""
    header: str = ""


# ---------------------------------------------------------------------------
# IR Nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElementwiseUnaryIR:
    """y = f(x) elementwise kernel."""

    expr: str
    inp: str = "inp"
    out: str = "out"
    vjp: VJPSpec | None = None
    tags: frozenset[str] = frozenset()

    @property
    def kind(self) -> PatternKind:
        return PatternKind.ELEMENTWISE_UNARY

    def to_source(self) -> str:
        from .codegen import elementwise_unary_source

        return elementwise_unary_source(expr=self.expr, inp=self.inp, out=self.out)

    @property
    def input_names(self) -> tuple[str, ...]:
        return (self.inp,)

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self.out,)


@dataclass(frozen=True)
class ElementwiseBinaryIR:
    """z = f(a, b) elementwise kernel."""

    expr: str
    lhs: str = "lhs"
    rhs: str = "rhs"
    out: str = "out"
    vjp: VJPSpec | None = None
    tags: frozenset[str] = frozenset()

    @property
    def kind(self) -> PatternKind:
        return PatternKind.ELEMENTWISE_BINARY

    def to_source(self) -> str:
        from .codegen import elementwise_binary_source

        return elementwise_binary_source(
            expr=self.expr, lhs=self.lhs, rhs=self.rhs, out=self.out
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return (self.lhs, self.rhs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self.out,)


@dataclass(frozen=True)
class RowwiseReductionIR:
    """Serial rowwise reduction kernel."""

    reduce_expr: str
    init_expr: str
    finalize_expr: str
    d: int
    inp: str = "inp"
    out: str = "out"
    vjp: VJPSpec | None = None
    tags: frozenset[str] = frozenset()

    @property
    def kind(self) -> PatternKind:
        return PatternKind.ROWWISE_REDUCTION

    def to_source(self) -> str:
        from .codegen import rowwise_reduction_source

        return rowwise_reduction_source(
            reduce_expr=self.reduce_expr,
            init_expr=self.init_expr,
            finalize_expr=self.finalize_expr,
            d=self.d,
            inp=self.inp,
            out=self.out,
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return (self.inp,)

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self.out,)


@dataclass(frozen=True)
class RowwiseParallelReductionIR:
    """Parallel rowwise reduction with threadgroup tree reduction."""

    d: int
    tg: int
    init_expr: str
    update_expr: str
    reduce_op: str
    finalize_expr: str
    inp: str = "inp"
    out: str = "out"
    scratch: str = "buf"
    vjp: VJPSpec | None = None
    tags: frozenset[str] = frozenset()

    @property
    def kind(self) -> PatternKind:
        return PatternKind.ROWWISE_PARALLEL_REDUCTION

    def to_source(self) -> str:
        from .codegen import rowwise_parallel_reduction_source

        return rowwise_parallel_reduction_source(
            d=self.d,
            tg=self.tg,
            init_expr=self.init_expr,
            update_expr=self.update_expr,
            reduce_op=self.reduce_op,
            finalize_expr=self.finalize_expr,
            inp=self.inp,
            out=self.out,
            scratch=self.scratch,
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return (self.inp,)

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self.out,)


@dataclass(frozen=True)
class RowwiseMapReduceIR:
    """Two-pass rowwise map-reduce kernel (softmax, layernorm)."""

    d: int
    tg: int
    pass1_init: str
    pass1_update: str
    pass1_reduce_op: str
    pass2_init: str
    pass2_update: str
    pass2_reduce_op: str
    write_expr: str
    inp: str = "inp"
    out: str = "out"
    scratch1: str = "buf1"
    scratch2: str = "buf2"
    vjp: VJPSpec | None = None
    tags: frozenset[str] = frozenset()

    @property
    def kind(self) -> PatternKind:
        return PatternKind.ROWWISE_MAPREDUCE

    def to_source(self) -> str:
        from .codegen import rowwise_mapreduce_source

        return rowwise_mapreduce_source(
            d=self.d,
            tg=self.tg,
            pass1_init=self.pass1_init,
            pass1_update=self.pass1_update,
            pass1_reduce_op=self.pass1_reduce_op,
            pass2_init=self.pass2_init,
            pass2_update=self.pass2_update,
            pass2_reduce_op=self.pass2_reduce_op,
            write_expr=self.write_expr,
            inp=self.inp,
            out=self.out,
            scratch1=self.scratch1,
            scratch2=self.scratch2,
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return (self.inp,)

    @property
    def output_names(self) -> tuple[str, ...]:
        return (self.out,)


@dataclass(frozen=True)
class Tiled2DIR:
    """2D tiled operation (attention, matmul epilogues). Escape hatch."""

    source: str
    input_names_list: tuple[str, ...]
    output_names_list: tuple[str, ...]
    tile_m: int = 32
    tile_n: int = 32
    vjp: VJPSpec | None = None
    tags: frozenset[str] = frozenset()

    @property
    def kind(self) -> PatternKind:
        return PatternKind.TILED_2D

    def to_source(self) -> str:
        return self.source

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.input_names_list

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.output_names_list


@dataclass(frozen=True)
class RawIR:
    """Escape hatch for hand-written MSL."""

    source: str
    input_names_list: tuple[str, ...]
    output_names_list: tuple[str, ...]
    vjp: VJPSpec | None = None
    tags: frozenset[str] = frozenset()

    @property
    def kind(self) -> PatternKind:
        return PatternKind.RAW

    def to_source(self) -> str:
        return self.source

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.input_names_list

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.output_names_list


# Union type for all IR nodes
KernelIR = (
    ElementwiseUnaryIR
    | ElementwiseBinaryIR
    | RowwiseReductionIR
    | RowwiseParallelReductionIR
    | RowwiseMapReduceIR
    | Tiled2DIR
    | RawIR
)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _vjp_to_dict(v: VJPSpec | None) -> dict[str, Any] | None:
    if v is None:
        return None
    return {
        "bwd_source": v.bwd_source,
        "bwd_input_names": list(v.bwd_input_names),
        "bwd_output_names": list(v.bwd_output_names),
        "use_output": v.use_output,
        "prelude": v.prelude,
        "header": v.header,
    }


def _vjp_from_dict(d: dict[str, Any] | None) -> VJPSpec | None:
    if d is None:
        return None
    return VJPSpec(
        bwd_source=d["bwd_source"],
        bwd_input_names=tuple(d["bwd_input_names"]),
        bwd_output_names=tuple(d["bwd_output_names"]),
        use_output=d.get("use_output", True),
        prelude=d.get("prelude", ""),
        header=d.get("header", ""),
    )


_IR_TYPE_MAP: dict[str, type] = {
    "elementwise_unary": ElementwiseUnaryIR,
    "elementwise_binary": ElementwiseBinaryIR,
    "rowwise_reduction": RowwiseReductionIR,
    "rowwise_parallel_reduction": RowwiseParallelReductionIR,
    "rowwise_mapreduce": RowwiseMapReduceIR,
    "tiled_2d": Tiled2DIR,
    "raw": RawIR,
}


def ir_to_json(node: KernelIR) -> str:
    """Serialize an IR node to JSON."""
    d = asdict(node)  # type: ignore[arg-type]
    d["_kind"] = node.kind.value
    # Convert frozensets to lists for JSON
    if "tags" in d:
        d["tags"] = sorted(d["tags"])
    if "vjp" in d and d["vjp"] is not None:
        d["vjp"]["bwd_input_names"] = list(d["vjp"]["bwd_input_names"])
        d["vjp"]["bwd_output_names"] = list(d["vjp"]["bwd_output_names"])
    return json.dumps(d, indent=2)


def ir_from_json(s: str) -> KernelIR:
    """Deserialize an IR node from JSON."""
    d = json.loads(s)
    kind_str = d.pop("_kind")
    cls = _IR_TYPE_MAP[kind_str]

    # Convert lists back to tuples/frozensets
    if "tags" in d:
        d["tags"] = frozenset(d["tags"])
    if "vjp" in d:
        d["vjp"] = _vjp_from_dict(d["vjp"])

    # Handle tuple fields
    for key in ("input_names_list", "output_names_list", "bwd_input_names", "bwd_output_names"):
        if key in d and isinstance(d[key], list):
            d[key] = tuple(d[key])

    result: KernelIR = cls(**d)
    return result
