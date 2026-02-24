"""Symbolic tracing with proxy tensors for phase-1/2/3 JIT fusion."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .graph import Graph, TensorMeta

Scalar = bool | int | float


@runtime_checkable
class TensorLike(Protocol):
    """Minimal tensor protocol required by the symbolic tracer."""

    shape: Any
    dtype: Any


@dataclass(frozen=True, slots=True)
class TraceResult:
    """Output of symbolic tracing."""

    graph: Graph
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ProxyTensor:
    """Tensor proxy that records operations to a symbolic graph."""

    _tracer: SymbolicTracer
    _node_id: int
    _meta: TensorMeta

    @property
    def node_id(self) -> int:
        return self._node_id

    @property
    def meta(self) -> TensorMeta:
        return self._meta

    @property
    def shape(self) -> tuple[int, ...]:
        return self._meta.shape

    @property
    def dtype(self) -> str:
        return self._meta.dtype

    def __add__(self, other: ProxyTensor | Scalar) -> ProxyTensor:
        return self._tracer._binary("add", self, other)

    def __radd__(self, other: Scalar) -> ProxyTensor:
        return self._tracer._binary("add", other, self)

    def __sub__(self, other: ProxyTensor | Scalar) -> ProxyTensor:
        return self._tracer._binary("sub", self, other)

    def __rsub__(self, other: Scalar) -> ProxyTensor:
        return self._tracer._binary("sub", other, self)

    def __mul__(self, other: ProxyTensor | Scalar) -> ProxyTensor:
        return self._tracer._binary("mul", self, other)

    def __rmul__(self, other: Scalar) -> ProxyTensor:
        return self._tracer._binary("mul", other, self)

    def __truediv__(self, other: ProxyTensor | Scalar) -> ProxyTensor:
        return self._tracer._binary("div", self, other)

    def __rtruediv__(self, other: Scalar) -> ProxyTensor:
        return self._tracer._binary("div", other, self)

    def __neg__(self) -> ProxyTensor:
        return self._tracer._unary("neg", self)

    def exp(self) -> ProxyTensor:
        return self._tracer._unary("exp", self)

    def tanh(self) -> ProxyTensor:
        return self._tracer._unary("tanh", self)

    def sigmoid(self) -> ProxyTensor:
        return self._tracer._unary("sigmoid", self)

    def relu(self) -> ProxyTensor:
        return self._tracer._unary("relu", self)

    def silu(self) -> ProxyTensor:
        return self._tracer._unary("silu", self)

    def log(self) -> ProxyTensor:
        return self._tracer._unary("log", self)

    def sqrt(self) -> ProxyTensor:
        return self._tracer._unary("sqrt", self)

    def rsqrt(self) -> ProxyTensor:
        return self._tracer._unary("rsqrt", self)

    def maximum(self, other: ProxyTensor | Scalar) -> ProxyTensor:
        return self._tracer._binary("maximum", self, other)

    def minimum(self, other: ProxyTensor | Scalar) -> ProxyTensor:
        return self._tracer._binary("minimum", self, other)

    def sum(self, *, axis: int = -1, keepdims: bool = False) -> ProxyTensor:
        return self._tracer._reduce("sum", self, axis=axis, keepdims=keepdims)

    def mean(self, *, axis: int = -1, keepdims: bool = False) -> ProxyTensor:
        return self._tracer._reduce("mean", self, axis=axis, keepdims=keepdims)

    def max(self, *, axis: int = -1, keepdims: bool = False) -> ProxyTensor:
        return self._tracer._reduce("max", self, axis=axis, keepdims=keepdims)


class SymbolicTracer:
    """Constructs a symbolic graph by executing a function on proxies."""

    def __init__(self, graph: Graph | None = None) -> None:
        self.graph = graph or Graph()

    def input(self, *, name: str, meta: TensorMeta) -> ProxyTensor:
        """Create an input proxy tensor."""
        node_id = self.graph.add_input(name=name, meta=meta)
        return ProxyTensor(self, node_id, meta)

    def constant(self, value: Scalar, *, dtype: str) -> ProxyTensor:
        """Create a scalar-literal proxy node."""
        meta = TensorMeta(shape=(), dtype=dtype)
        node_id = self.graph.add_constant(value=value, meta=meta)
        return ProxyTensor(self, node_id, meta)

    def _unary(self, op: str, value: ProxyTensor) -> ProxyTensor:
        proxy = self._ensure_proxy(value)
        node_id = self.graph.add_op(op, inputs=(proxy.node_id,), meta=proxy.meta)
        return ProxyTensor(self, node_id, proxy.meta)

    def _binary(
        self,
        op: str,
        lhs: ProxyTensor | Scalar,
        rhs: ProxyTensor | Scalar,
    ) -> ProxyTensor:
        left, right = self._coerce_binary_operands(lhs, rhs)
        out_meta = _binary_output_meta(left.meta, right.meta)
        node_id = self.graph.add_op(op, inputs=(left.node_id, right.node_id), meta=out_meta)
        return ProxyTensor(self, node_id, out_meta)

    def _reduce(
        self,
        op: str,
        value: ProxyTensor,
        *,
        axis: int = -1,
        keepdims: bool = False,
    ) -> ProxyTensor:
        proxy = self._ensure_proxy(value)
        axis_norm = _normalize_reduction_axis(proxy.meta.shape, axis)
        out_meta = _reduction_output_meta(proxy.meta, axis=axis_norm, keepdims=keepdims)
        node_id = self.graph.add_op(
            op,
            inputs=(proxy.node_id,),
            meta=out_meta,
            attrs={"axis": axis_norm, "keepdims": bool(keepdims)},
        )
        return ProxyTensor(self, node_id, out_meta)

    def quantized_matmul(
        self,
        lhs: ProxyTensor,
        rhs: ProxyTensor,
        *,
        scales: ProxyTensor | None = None,
        biases: ProxyTensor | None = None,
        group_size: int | None = None,
        bits: int | None = None,
        mode: str | None = None,
        transpose: bool = False,
    ) -> ProxyTensor:
        left = self._ensure_proxy(lhs)
        right = self._ensure_proxy(rhs)
        out_meta = _quantized_matmul_output_meta(
            left.meta,
            right.meta,
            transpose=transpose,
            bits=bits,
        )

        input_ids: list[int] = [left.node_id, right.node_id]
        if scales is not None:
            input_ids.append(self._ensure_proxy(scales).node_id)
        if biases is not None:
            input_ids.append(self._ensure_proxy(biases).node_id)

        node_id = self.graph.add_op(
            "quantized_matmul",
            inputs=tuple(input_ids),
            meta=out_meta,
            attrs={
                "group_size": group_size,
                "bits": bits,
                "mode": mode,
                "transpose": bool(transpose),
            },
        )
        return ProxyTensor(self, node_id, out_meta)

    def _coerce_binary_operands(
        self,
        lhs: ProxyTensor | Scalar,
        rhs: ProxyTensor | Scalar,
    ) -> tuple[ProxyTensor, ProxyTensor]:
        if isinstance(lhs, ProxyTensor) and isinstance(rhs, ProxyTensor):
            return self._ensure_proxy(lhs), self._ensure_proxy(rhs)
        if isinstance(lhs, ProxyTensor):
            left = self._ensure_proxy(lhs)
            right = self.constant(_ensure_scalar(rhs), dtype=left.dtype)
            return left, right
        if isinstance(rhs, ProxyTensor):
            right = self._ensure_proxy(rhs)
            left = self.constant(_ensure_scalar(lhs), dtype=right.dtype)
            return left, right
        raise TypeError("At least one operand must be a ProxyTensor.")

    def _ensure_proxy(self, value: ProxyTensor) -> ProxyTensor:
        if value._tracer is not self:
            raise ValueError("Cannot mix ProxyTensor objects from different traces.")
        return value


def symbolic_trace(
    fn: Callable[..., ProxyTensor | Sequence[ProxyTensor]],
    *inputs: TensorMeta | TensorLike,
    input_names: Sequence[str] | None = None,
) -> TraceResult:
    """Trace ``fn`` with proxy tensors and return a symbolic graph."""
    tracer = SymbolicTracer()
    names = tuple(input_names or ())
    if names and len(names) != len(inputs):
        raise ValueError("input_names length must match number of inputs.")

    proxy_inputs: list[ProxyTensor] = []
    for idx, raw_input in enumerate(inputs):
        meta = _to_meta(raw_input)
        name = names[idx] if names else f"arg{idx}"
        proxy_inputs.append(tracer.input(name=name, meta=meta))

    raw_outputs = fn(*proxy_inputs)
    outputs = _normalize_outputs(raw_outputs)
    for out in outputs:
        if out._tracer is not tracer:
            raise ValueError("Trace output contains ProxyTensor from a different tracer.")

    return TraceResult(
        graph=tracer.graph,
        input_ids=tuple(proxy.node_id for proxy in proxy_inputs),
        output_ids=tuple(proxy.node_id for proxy in outputs),
    )


def add(lhs: ProxyTensor | Scalar, rhs: ProxyTensor | Scalar) -> ProxyTensor:
    return _binary_op("add", lhs, rhs)


def sub(lhs: ProxyTensor | Scalar, rhs: ProxyTensor | Scalar) -> ProxyTensor:
    return _binary_op("sub", lhs, rhs)


def mul(lhs: ProxyTensor | Scalar, rhs: ProxyTensor | Scalar) -> ProxyTensor:
    return _binary_op("mul", lhs, rhs)


def div(lhs: ProxyTensor | Scalar, rhs: ProxyTensor | Scalar) -> ProxyTensor:
    return _binary_op("div", lhs, rhs)


def neg(x: ProxyTensor) -> ProxyTensor:
    return x._tracer._unary("neg", x)


def exp(x: ProxyTensor) -> ProxyTensor:
    return x._tracer._unary("exp", x)


def tanh(x: ProxyTensor) -> ProxyTensor:
    return x._tracer._unary("tanh", x)


def sigmoid(x: ProxyTensor) -> ProxyTensor:
    return x._tracer._unary("sigmoid", x)


def relu(x: ProxyTensor) -> ProxyTensor:
    return x._tracer._unary("relu", x)


def silu(x: ProxyTensor) -> ProxyTensor:
    return x._tracer._unary("silu", x)


def log(x: ProxyTensor) -> ProxyTensor:
    return x._tracer._unary("log", x)


def sqrt(x: ProxyTensor) -> ProxyTensor:
    return x._tracer._unary("sqrt", x)


def rsqrt(x: ProxyTensor) -> ProxyTensor:
    return x._tracer._unary("rsqrt", x)


def maximum(lhs: ProxyTensor | Scalar, rhs: ProxyTensor | Scalar) -> ProxyTensor:
    return _binary_op("maximum", lhs, rhs)


def minimum(lhs: ProxyTensor | Scalar, rhs: ProxyTensor | Scalar) -> ProxyTensor:
    return _binary_op("minimum", lhs, rhs)


def reduce_sum(x: ProxyTensor, *, axis: int = -1, keepdims: bool = False) -> ProxyTensor:
    return x._tracer._reduce("sum", x, axis=axis, keepdims=keepdims)


def reduce_mean(x: ProxyTensor, *, axis: int = -1, keepdims: bool = False) -> ProxyTensor:
    return x._tracer._reduce("mean", x, axis=axis, keepdims=keepdims)


def reduce_max(x: ProxyTensor, *, axis: int = -1, keepdims: bool = False) -> ProxyTensor:
    return x._tracer._reduce("max", x, axis=axis, keepdims=keepdims)


def quantized_matmul(
    lhs: ProxyTensor,
    rhs: ProxyTensor,
    *,
    scales: ProxyTensor | None = None,
    biases: ProxyTensor | None = None,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str | None = None,
    transpose: bool = False,
) -> ProxyTensor:
    return lhs._tracer.quantized_matmul(
        lhs,
        rhs,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
        transpose=transpose,
    )


def _binary_op(op: str, lhs: ProxyTensor | Scalar, rhs: ProxyTensor | Scalar) -> ProxyTensor:
    tracer = _resolve_tracer(lhs, rhs)
    return tracer._binary(op, lhs, rhs)


def _resolve_tracer(*values: ProxyTensor | Scalar) -> SymbolicTracer:
    tracer: SymbolicTracer | None = None
    for value in values:
        if not isinstance(value, ProxyTensor):
            continue
        if tracer is None:
            tracer = value._tracer
            continue
        if value._tracer is not tracer:
            raise ValueError("Cannot mix ProxyTensor objects from different traces.")
    if tracer is None:
        raise TypeError("At least one operand must be a ProxyTensor.")
    return tracer


def _normalize_outputs(raw_outputs: ProxyTensor | Sequence[ProxyTensor]) -> tuple[ProxyTensor, ...]:
    if isinstance(raw_outputs, ProxyTensor):
        return (raw_outputs,)
    if not isinstance(raw_outputs, Sequence):
        raise TypeError("Trace function must return ProxyTensor or sequence of ProxyTensor.")
    outputs = tuple(raw_outputs)
    if not outputs:
        raise ValueError("Trace function must return at least one output.")
    for out in outputs:
        if not isinstance(out, ProxyTensor):
            raise TypeError("All trace outputs must be ProxyTensor instances.")
    return outputs


def _to_meta(value: TensorMeta | TensorLike) -> TensorMeta:
    if isinstance(value, TensorMeta):
        return value
    if isinstance(value, TensorLike):
        return TensorMeta(shape=_to_shape(value.shape), dtype=_to_dtype(value.dtype))
    raise TypeError("Trace inputs must be TensorMeta or TensorLike.")


def _to_shape(shape: Any) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)


def _to_dtype(dtype: Any) -> str:
    if isinstance(dtype, str):
        return dtype
    if hasattr(dtype, "name"):
        return str(dtype.name)
    return str(dtype)


def _ensure_scalar(value: ProxyTensor | Scalar) -> Scalar:
    if isinstance(value, ProxyTensor):
        raise TypeError("Expected scalar literal, got ProxyTensor.")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    raise TypeError(f"Unsupported scalar literal type: {type(value).__name__}")


def _binary_output_meta(lhs: TensorMeta, rhs: TensorMeta) -> TensorMeta:
    if not lhs.is_scalar and not rhs.is_scalar and lhs.dtype != rhs.dtype:
        raise ValueError(
            "Elementwise tensor operands must share dtype "
            f"(got {lhs.dtype!r} and {rhs.dtype!r})."
        )
    out_shape = _broadcast_shape(lhs.shape, rhs.shape)
    out_dtype = lhs.dtype
    if lhs.is_scalar and not rhs.is_scalar:
        out_dtype = rhs.dtype
    return TensorMeta(shape=out_shape, dtype=out_dtype)


def _broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
    if lhs == rhs:
        return lhs
    if not lhs:
        return rhs
    if not rhs:
        return lhs

    lhs_rank = len(lhs)
    rhs_rank = len(rhs)
    max_rank = lhs_rank if lhs_rank >= rhs_rank else rhs_rank
    lhs_pad = (1,) * (max_rank - lhs_rank) + lhs
    rhs_pad = (1,) * (max_rank - rhs_rank) + rhs

    out: list[int] = []
    for idx, (lhs_dim, rhs_dim) in enumerate(zip(lhs_pad, rhs_pad, strict=True)):
        if lhs_dim == rhs_dim:
            out.append(lhs_dim)
            continue
        if lhs_dim == 1:
            out.append(rhs_dim)
            continue
        if rhs_dim == 1:
            out.append(lhs_dim)
            continue
        raise ValueError(
            "Elementwise operands are not broadcast-compatible "
            f"(lhs={lhs}, rhs={rhs}, mismatch at dim {idx})."
        )
    return tuple(out)


def _normalize_reduction_axis(shape: tuple[int, ...], axis: int) -> int:
    if not shape:
        raise ValueError("Reduction over scalar tensors is unsupported.")
    if not isinstance(axis, int):
        raise TypeError(f"Reduction axis must be int, got {type(axis).__name__}.")

    rank = len(shape)
    axis_norm = axis + rank if axis < 0 else axis
    if axis_norm < 0 or axis_norm >= rank:
        raise ValueError(f"Reduction axis {axis} is out of range for shape {shape}.")
    if axis_norm != rank - 1:
        raise ValueError(
            "Only last-axis reductions are supported in phase-2 tracing "
            f"(axis={axis}, shape={shape})."
        )
    return axis_norm


def _reduction_output_meta(
    value: TensorMeta,
    *,
    axis: int,
    keepdims: bool,
) -> TensorMeta:
    shape = list(value.shape)
    if keepdims:
        shape[axis] = 1
    else:
        shape.pop(axis)
    return TensorMeta(shape=tuple(shape), dtype=value.dtype)


def _quantized_matmul_output_meta(
    lhs: TensorMeta,
    rhs: TensorMeta,
    *,
    transpose: bool,
    bits: int | None,
) -> TensorMeta:
    if len(rhs.shape) != 2:
        raise ValueError(
            "quantized_matmul rhs must be rank-2 for tracing metadata propagation "
            f"(got shape {rhs.shape})."
        )
    if len(lhs.shape) < 1:
        raise ValueError(
            "quantized_matmul lhs must be rank-1 or higher "
            f"(got shape {lhs.shape})."
        )

    k_lhs = lhs.shape[-1]
    pack_factor = 1
    if bits in (4, 8):
        pack_factor = 32 // int(bits)
    if transpose:
        n = rhs.shape[0]
        k_rhs = rhs.shape[1] * pack_factor
    else:
        k_rhs = rhs.shape[0] * pack_factor
        n = rhs.shape[1]

    if k_lhs != k_rhs:
        raise ValueError(
            "quantized_matmul inner dimensions must agree "
            f"(lhs_k={k_lhs}, rhs_k={k_rhs}, transpose={transpose})."
        )

    if len(lhs.shape) == 1:
        return TensorMeta(shape=(n,), dtype=lhs.dtype)
    return TensorMeta(shape=lhs.shape[:-1] + (n,), dtype=lhs.dtype)
