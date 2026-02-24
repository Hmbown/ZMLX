from __future__ import annotations

import inspect
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Literal

from .._compat import import_mx
from .graph import FusedOp, Graph, TensorMeta
from .passes import run_fusion
from .runtime import (
    CompiledFusedElementwise,
    CompiledFusedReductionTwoPass,
    CompiledQuantizedSharedInputFusion,
    compile_broadcast_mul_reduce_two_pass,
    compile_fused_elementwise,
    compile_quantized_shared_input_swiglu,
    launch_broadcast_mul_reduce_two_pass,
    launch_quantized_shared_input_swiglu,
)
from .trace import (
    ProxyTensor,
    exp,
    log,
    maximum,
    minimum,
    quantized_matmul,
    reduce_max,
    reduce_mean,
    reduce_sum,
    relu,
    rsqrt,
    sigmoid,
    silu,
    sqrt,
    symbolic_trace,
    tanh,
)

Scalar = bool | int | float
SignatureItem = tuple[str, Any, Any]


class _UnsupportedFusion(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _InputBinding:
    name: str
    kind: Literal["arg", "const"]
    arg_index: int | None = None
    const_value: Scalar | None = None
    const_dtype: str | None = None


@dataclass(frozen=True, slots=True)
class _FusionPlan:
    kind: str
    descriptor: dict[str, Any]
    bindings: tuple[_InputBinding, ...]
    output_meta: TensorMeta
    ops_repr: str


@dataclass(frozen=True, slots=True)
class _CompiledEntry:
    plan: _FusionPlan
    kernel: (
        CompiledFusedElementwise
        | CompiledFusedReductionTwoPass
        | CompiledQuantizedSharedInputFusion
    )


@dataclass
class _FusionState:
    traced: bool = False
    fusable: bool = False
    unsupported_reason: str | None = None
    expression: str | None = None
    traces: int = 0
    compiles: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    cache: dict[tuple[SignatureItem, ...], _CompiledEntry] = field(default_factory=dict)
    failed_signatures: set[tuple[SignatureItem, ...]] = field(default_factory=set)


def _is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float))


def _arg_signature(args: tuple[Any, ...]) -> tuple[SignatureItem, ...]:
    signature: list[SignatureItem] = []
    tensor_count = 0
    for arg in args:
        if _is_tensor_like(arg):
            tensor_count += 1
            signature.append(("tensor", tuple(int(dim) for dim in arg.shape), str(arg.dtype)))
            continue
        if _is_scalar(arg):
            signature.append(("scalar", type(arg).__name__, repr(arg)))
            continue
        raise _UnsupportedFusion(
            f"unsupported non-tensor argument type: {type(arg).__name__}"
        )
    if tensor_count == 0:
        raise _UnsupportedFusion("at least one tensor argument is required")
    return tuple(signature)


def _first_tensor_arg(args: tuple[Any, ...]) -> Any:
    for arg in args:
        if _is_tensor_like(arg):
            return arg
    raise _UnsupportedFusion("no tensor argument found")


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return int(n)


def _dtype_from_name(mx: Any, name: str | None, fallback: Any) -> Any:
    if not name:
        return fallback
    return getattr(mx, name, fallback)


def _arg_names(fn: Callable[..., Any]) -> tuple[str, ...]:
    sig = inspect.signature(fn)
    names: list[str] = []
    for param in sig.parameters.values():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise _UnsupportedFusion("only positional parameters are supported")
        names.append(param.name)
    return tuple(names)


@contextmanager
def _patched_mx_for_tracing() -> Any:
    mx = import_mx()
    proxy_ops: dict[str, Callable[..., Any]] = {
        "sigmoid": sigmoid,
        "silu": silu,
        "tanh": tanh,
        "exp": exp,
        "log": log,
        "sqrt": sqrt,
        "rsqrt": rsqrt,
        "relu": relu,
        "maximum": maximum,
        "minimum": minimum,
    }
    originals: dict[str, Any] = {}

    def _contains_proxy(values: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        if any(isinstance(v, ProxyTensor) for v in values):
            return True
        return any(isinstance(v, ProxyTensor) for v in kwargs.values())

    def _wrap(original: Any, proxy_fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(original)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            if _contains_proxy(args, kwargs):
                if kwargs:
                    raise _UnsupportedFusion("keyword arguments in traced mx ops are unsupported")
                return proxy_fn(*args)
            return original(*args, **kwargs)

        return _inner

    def _wrap_reduce(original: Any, reduce_name: str) -> Callable[..., Any]:
        @wraps(original)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            if not _contains_proxy(args, kwargs):
                return original(*args, **kwargs)
            if len(args) != 1:
                raise _UnsupportedFusion(
                    f"traced mx.{reduce_name} expects exactly one positional tensor argument"
                )
            axis = kwargs.get("axis", -1)
            keepdims = bool(kwargs.get("keepdims", False))
            x = args[0]
            if reduce_name == "sum":
                return reduce_sum(x, axis=axis, keepdims=keepdims)
            if reduce_name == "mean":
                return reduce_mean(x, axis=axis, keepdims=keepdims)
            return reduce_max(x, axis=axis, keepdims=keepdims)

        return _inner

    def _wrap_quantized_matmul(original: Any) -> Callable[..., Any]:
        @wraps(original)
        def _inner(*args: Any, **kwargs: Any) -> Any:
            if not _contains_proxy(args, kwargs):
                return original(*args, **kwargs)
            if len(args) < 2:
                raise _UnsupportedFusion(
                    "traced mx.quantized_matmul requires at least lhs and rhs arguments"
                )
            lhs = args[0]
            rhs = args[1]
            scales = kwargs.get("scales")
            biases = kwargs.get("biases")
            group_size = kwargs.get("group_size")
            bits = kwargs.get("bits")
            mode = kwargs.get("mode")
            transpose = bool(kwargs.get("transpose", False))
            return quantized_matmul(
                lhs,
                rhs,
                scales=scales,
                biases=biases,
                group_size=group_size,
                bits=bits,
                mode=mode,
                transpose=transpose,
            )

        return _inner

    try:
        for name, proxy_fn in proxy_ops.items():
            if not hasattr(mx, name):
                continue
            original = getattr(mx, name)
            originals[name] = original
            setattr(mx, name, _wrap(original, proxy_fn))
        for reduce_name in ("sum", "mean", "max"):
            if not hasattr(mx, reduce_name):
                continue
            original = getattr(mx, reduce_name)
            originals[reduce_name] = original
            setattr(mx, reduce_name, _wrap_reduce(original, reduce_name))
        if hasattr(mx, "quantized_matmul"):
            original = mx.quantized_matmul
            originals["quantized_matmul"] = original
            mx.quantized_matmul = _wrap_quantized_matmul(original)
        yield
    finally:
        for name, original in originals.items():
            setattr(mx, name, original)


def _select_fused_op(graph: Graph, output_id: int) -> FusedOp:
    result = run_fusion(graph)
    for fused in result.fused_ops:
        if fused.output_id == output_id:
            return fused
    raise _UnsupportedFusion("no fusion pattern ended at function output")


def _build_plan(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    arg_names: tuple[str, ...],
) -> _FusionPlan:
    tensor_positions: list[int] = []
    tensor_names: list[str] = []
    tensor_metas: list[TensorMeta] = []

    for idx, arg in enumerate(args):
        if not _is_tensor_like(arg):
            continue
        tensor_positions.append(idx)
        tensor_names.append(arg_names[idx])
        tensor_metas.append(
            TensorMeta(
                shape=tuple(int(dim) for dim in arg.shape),
                dtype=str(arg.dtype),
            )
        )

    if not tensor_positions:
        raise _UnsupportedFusion("fused path requires at least one tensor input")

    def _invoke_with_proxy_tensors(*proxy_tensors: ProxyTensor) -> Any:
        call_args = list(args)
        for pos, proxy in zip(tensor_positions, proxy_tensors, strict=True):
            call_args[pos] = proxy
        with _patched_mx_for_tracing():
            return fn(*call_args)

    try:
        trace = symbolic_trace(
            _invoke_with_proxy_tensors,
            *tensor_metas,
            input_names=tensor_names,
        )
    except Exception as exc:
        raise _UnsupportedFusion(f"symbolic tracing failed: {exc}") from exc

    if len(trace.output_ids) != 1:
        raise _UnsupportedFusion("phase-1 fusion currently supports a single output")

    fused = _select_fused_op(trace.graph, trace.output_ids[0])
    descriptor, bindings = _build_descriptor(trace.graph, fused, arg_names)
    return _FusionPlan(
        kind=fused.kind,
        descriptor=descriptor,
        bindings=bindings,
        output_meta=fused.output_meta,
        ops_repr=f"{fused.kind}: " + " -> ".join(fused.ops),
    )


def _build_descriptor(
    graph: Graph,
    fused: FusedOp,
    arg_names: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[_InputBinding, ...]]:
    tensor_arg_by_name = {name: idx for idx, name in enumerate(arg_names)}
    external_name_by_node_id: dict[int, str] = {}
    bindings: list[_InputBinding] = []

    for node_id in fused.input_ids:
        node = graph.node(node_id)
        if node.op == "input":
            name_raw = node.attrs.get("name")
            name = str(name_raw) if name_raw is not None else f"arg_{node_id}"
            if name not in tensor_arg_by_name:
                raise _UnsupportedFusion(f"trace input {name!r} was not found in function args")
            binding = _InputBinding(name=name, kind="arg", arg_index=tensor_arg_by_name[name])
            external_name_by_node_id[node_id] = name
            bindings.append(binding)
            continue

        if node.op == "constant":
            value = node.attrs.get("value")
            if not _is_scalar(value):
                raise _UnsupportedFusion("only scalar constants are supported")
            name = f"const_{node_id}"
            dtype = node.meta.dtype if node.meta is not None else "float32"
            binding = _InputBinding(
                name=name,
                kind="const",
                const_value=value,
                const_dtype=dtype,
            )
            external_name_by_node_id[node_id] = name
            bindings.append(binding)
            continue

        raise _UnsupportedFusion(f"unsupported fused external input op: {node.op}")

    if fused.kind == "elementwise_chain":
        fused_set = set(fused.node_ids)
        nodes: list[dict[str, Any]] = []
        for node_id in fused.node_ids:
            node = graph.node(node_id)
            args: list[str] = []
            for input_id in node.inputs:
                if input_id in fused_set:
                    args.append(f"n{input_id}")
                    continue
                ext = external_name_by_node_id.get(input_id)
                if ext is None:
                    raise _UnsupportedFusion(
                        f"fused node {node_id} depends on unsupported external node {input_id}"
                    )
                args.append(ext)

            nodes.append({
                "id": f"n{node_id}",
                "op": node.op,
                "args": args,
            })

        descriptor = {
            "name": "zmlx_phase1_elementwise",
            "kind": "elementwise_chain",
            "inputs": [binding.name for binding in bindings],
            "nodes": nodes,
            "output": f"n{fused.output_id}",
        }
        return descriptor, tuple(bindings)

    if fused.kind == "broadcast_mul_reduce":
        if len(bindings) != 2:
            raise _UnsupportedFusion(
                f"broadcast_mul_reduce expects exactly 2 external inputs, got {len(bindings)}"
            )
        descriptor = {
            "name": "zmlx_phase2_broadcast_mul_reduce",
            "kind": fused.kind,
            "inputs": [binding.name for binding in bindings],
            "reduce_op": str(fused.attrs.get("reduce_op", "sum")),
            "axis": int(fused.attrs.get("axis", -1)),
            "keepdims": bool(fused.attrs.get("keepdims", False)),
            "no_fma": bool(fused.attrs.get("no_fma", False)),
        }
        return descriptor, tuple(bindings)

    if fused.kind == "quantized_shared_input_swiglu":
        if len(bindings) != 7:
            raise _UnsupportedFusion(
                "quantized_shared_input_swiglu expects 7 external inputs "
                f"(got {len(bindings)})"
            )
        descriptor = {
            "name": "zmlx_phase3_quantized_shared_input_swiglu",
            "kind": fused.kind,
            "inputs": [binding.name for binding in bindings],
            "bits": fused.attrs.get("bits"),
            "group_size": fused.attrs.get("group_size"),
            "mode": fused.attrs.get("mode"),
            "transpose": bool(fused.attrs.get("transpose", False)),
            "no_fma": bool(fused.attrs.get("no_fma", False)),
        }
        return descriptor, tuple(bindings)

    raise _UnsupportedFusion(f"Unsupported fused kind {fused.kind!r}.")


def _materialize_runtime_inputs(
    args: tuple[Any, ...],
    bindings: tuple[_InputBinding, ...],
    materialize_shape: tuple[int, ...],
    materialize_dtype: Any,
) -> list[Any]:
    mx = import_mx()
    runtime_inputs: list[Any] = []
    for binding in bindings:
        if binding.kind == "arg":
            assert binding.arg_index is not None
            runtime_inputs.append(args[binding.arg_index])
            continue

        assert binding.const_value is not None
        dtype = _dtype_from_name(mx, binding.const_dtype, materialize_dtype)
        runtime_inputs.append(mx.full(materialize_shape, binding.const_value, dtype=dtype))

    return runtime_inputs


def _launch_with_compiled(
    compiled: CompiledFusedElementwise,
    inputs: list[Any],
    out_shape: tuple[int, ...],
    out_dtype: Any,
) -> Any:
    mx = import_mx()
    n = _prod(out_shape)
    tgx = 1 if n <= 0 else min(256, n)
    return compiled.kernel(
        *inputs,
        template=[("T", mx.float32), ("O", out_dtype)],
        grid=(n, 1, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[out_dtype],
    )[0]


def _compile_for_plan(
    plan: _FusionPlan,
    runtime_inputs: list[Any],
) -> (
    CompiledFusedElementwise
    | CompiledFusedReductionTwoPass
    | CompiledQuantizedSharedInputFusion
):
    if plan.kind == "elementwise_chain":
        return compile_fused_elementwise(plan.descriptor, runtime_inputs)
    if plan.kind == "broadcast_mul_reduce":
        return compile_broadcast_mul_reduce_two_pass(plan.descriptor, runtime_inputs)
    if plan.kind == "quantized_shared_input_swiglu":
        return compile_quantized_shared_input_swiglu(plan.descriptor, runtime_inputs)
    raise _UnsupportedFusion(f"Unsupported fusion kind {plan.kind!r}")


def _launch_plan_entry(
    entry: _CompiledEntry,
    runtime_inputs: list[Any],
    output_meta: TensorMeta,
    output_dtype: Any,
) -> Any:
    if isinstance(entry.kernel, CompiledFusedElementwise):
        return _launch_with_compiled(
            entry.kernel,
            runtime_inputs,
            output_meta.shape,
            output_dtype,
        )
    if isinstance(entry.kernel, CompiledFusedReductionTwoPass):
        return launch_broadcast_mul_reduce_two_pass(
            entry.plan.descriptor,
            runtime_inputs,
            output_dtype=output_dtype,
        )
    if isinstance(entry.kernel, CompiledQuantizedSharedInputFusion):
        return launch_quantized_shared_input_swiglu(
            entry.plan.descriptor,
            runtime_inputs,
            output_dtype=output_dtype,
        )
    raise _UnsupportedFusion(f"Unsupported compiled kernel type {type(entry.kernel).__name__}")


def jit(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Phase-1/2/3 JIT fusion via symbolic tracing + Metal codegen/runtime."""

    state = _FusionState()
    try:
        arg_names = _arg_names(fn)
    except _UnsupportedFusion as exc:
        arg_names = ()
        state.unsupported_reason = str(exc)

    def fusion_stats() -> dict[str, Any]:
        return {
            "traced": state.traced,
            "fusable": state.fusable,
            "unsupported_reason": state.unsupported_reason,
            "expression": state.expression,
            "traces": state.traces,
            "compiles": state.compiles,
            "cache_hits": state.cache_hits,
            "fallbacks": state.fallbacks,
            "cache_entries": len(state.cache),
            "failed_signatures": len(state.failed_signatures),
        }

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if kwargs or not arg_names:
            state.fallbacks += 1
            return fn(*args, **kwargs)

        try:
            signature = _arg_signature(args)
        except _UnsupportedFusion as exc:
            state.traced = True
            state.traces += 1
            state.fusable = False
            state.unsupported_reason = str(exc)
            state.fallbacks += 1
            return fn(*args, **kwargs)

        if signature in state.failed_signatures:
            state.fallbacks += 1
            return fn(*args, **kwargs)

        entry = state.cache.get(signature)
        if entry is None:
            state.traced = True
            state.traces += 1
            try:
                plan = _build_plan(fn, args, arg_names)
                first_tensor = _first_tensor_arg(args)
                materialize_shape = tuple(int(dim) for dim in first_tensor.shape)
                runtime_inputs = _materialize_runtime_inputs(
                    args,
                    plan.bindings,
                    materialize_shape,
                    first_tensor.dtype,
                )
                compiled = _compile_for_plan(plan, runtime_inputs)
            except Exception as exc:
                state.fusable = False
                state.unsupported_reason = str(exc)
                state.failed_signatures.add(signature)
                state.fallbacks += 1
                return fn(*args, **kwargs)

            entry = _CompiledEntry(plan=plan, kernel=compiled)
            state.cache[signature] = entry
            state.expression = plan.ops_repr
            state.fusable = True
            state.unsupported_reason = None
            state.compiles += 1
        else:
            state.cache_hits += 1

        try:
            first_tensor = _first_tensor_arg(args)
            materialize_shape = tuple(int(dim) for dim in first_tensor.shape)
            runtime_inputs = _materialize_runtime_inputs(
                args,
                entry.plan.bindings,
                materialize_shape,
                first_tensor.dtype,
            )
            mx = import_mx()
            output_dtype = _dtype_from_name(mx, entry.plan.output_meta.dtype, first_tensor.dtype)
            return _launch_plan_entry(
                entry,
                runtime_inputs,
                entry.plan.output_meta,
                output_dtype,
            )
        except Exception:
            state.failed_signatures.add(signature)
            state.fallbacks += 1
            return fn(*args, **kwargs)

    wrapper.fusion_stats = fusion_stats  # type: ignore[attr-defined]
    return wrapper
