from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._compat import import_mx
from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .cache import GLOBAL_FUSION_COMPILE_CACHE, FusionCompileCache, FusionCompileKey
from .codegen import (
    FusedGraph,
    GeneratedQuantizedSharedInputSource,
    generate_broadcast_mul_reduce_two_pass_source,
    generate_fused_elementwise_source,
    generate_quantized_shared_input_source,
    normalize_descriptor,
)


@dataclass(frozen=True)
class CompiledFusedElementwise:
    """Compiled fused kernel object cached by graph + concrete input metadata."""

    cache_key: FusionCompileKey
    op_signature: str
    kernel_name: str
    input_names: tuple[str, ...]
    output_name: str
    source: str
    n_elements: int
    kernel: Any


def _prod(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _shape_of(x: Any) -> tuple[int, ...]:
    return tuple(int(d) for d in x.shape)


def _dtype_of(x: Any) -> Any:
    return x.dtype


def _as_input_list(inputs: Sequence[Any] | Any) -> list[Any]:
    if isinstance(inputs, (list, tuple)):
        return list(inputs)
    return [inputs]


def _validate_inputs(inputs: list[Any], graph: FusedGraph) -> tuple[int, ...]:
    if not inputs:
        raise ValueError("fused runtime requires at least one input")
    if len(inputs) != len(graph.input_names):
        raise ValueError(
            f"expected {len(graph.input_names)} inputs for graph, got {len(inputs)}"
        )

    shape0 = _shape_of(inputs[0])
    for i, x in enumerate(inputs[1:], start=1):
        shape_i = _shape_of(x)
        if shape_i != shape0:
            raise ValueError(
                "phase-1 fused elementwise launch requires identical shapes; "
                f"input0={shape0} input{i}={shape_i}"
            )
    return shape0


def _cache_key_for(
    *,
    op_signature: str,
    inputs: Sequence[Any],
) -> FusionCompileKey:
    return FusionCompileKey.from_parts(
        op_signature=op_signature,
        input_shapes=[_shape_of(x) for x in inputs],
        input_dtypes=[_dtype_of(x) for x in inputs],
    )


def compile_fused_elementwise(
    descriptor: FusedGraph | dict[str, Any],
    inputs: Sequence[Any],
    *,
    cache: FusionCompileCache | None = None,
    header: str = DEFAULT_HEADER,
    ensure_row_contiguous: bool = True,
    kernel_name: str | None = None,
) -> CompiledFusedElementwise:
    """Compile (or cache-hit) a fused elementwise Metal kernel for concrete inputs."""

    graph = normalize_descriptor(descriptor)
    inputs_list = _as_input_list(inputs)
    shape0 = _validate_inputs(inputs_list, graph)
    n_elements = _prod(shape0)

    generated = generate_fused_elementwise_source(
        graph,
        n_elements=n_elements,
        kernel_name=kernel_name,
        output_name="out",
    )
    key = _cache_key_for(op_signature=generated.op_signature, inputs=inputs_list)

    cache_obj = cache or GLOBAL_FUSION_COMPILE_CACHE
    cached = cache_obj.get(key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    compiled_kernel = metal_kernel(
        name=generated.kernel_name,
        input_names=list(generated.input_names),
        output_names=[generated.output_name],
        source=generated.source,
        header=header or DEFAULT_HEADER,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    compiled = CompiledFusedElementwise(
        cache_key=key,
        op_signature=generated.op_signature,
        kernel_name=generated.kernel_name,
        input_names=generated.input_names,
        output_name=generated.output_name,
        source=generated.source,
        n_elements=n_elements,
        kernel=compiled_kernel,
    )
    return cache_obj.put(key, compiled)  # type: ignore[no-any-return]


def launch_fused_elementwise(
    descriptor: FusedGraph | dict[str, Any],
    *inputs: Any,
    output_dtype: Any | None = None,
    compute_dtype: Any | None = None,
    threadgroup_x: int = 256,
    cache: FusionCompileCache | None = None,
    header: str = DEFAULT_HEADER,
    ensure_row_contiguous: bool = True,
    kernel_name: str | None = None,
) -> Any:
    """Compile+launch a fused elementwise kernel with concrete MLX inputs."""

    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        input_list = list(inputs[0])
    else:
        input_list = list(inputs)

    graph = normalize_descriptor(descriptor)
    shape0 = _validate_inputs(input_list, graph)
    n_elements = _prod(shape0)

    compiled = compile_fused_elementwise(
        graph,
        input_list,
        cache=cache,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        kernel_name=kernel_name,
    )

    mx = import_mx()
    effective_compute_dtype = compute_dtype if compute_dtype is not None else mx.float32
    effective_output_dtype = output_dtype if output_dtype is not None else input_list[0].dtype

    tgx = int(threadgroup_x)
    if tgx <= 0:
        raise ValueError("threadgroup_x must be > 0")
    if n_elements > 0:
        tgx = min(tgx, n_elements)
    else:
        tgx = 1

    out = compiled.kernel(
        *input_list,
        template=[("T", effective_compute_dtype), ("O", effective_output_dtype)],
        grid=(n_elements, 1, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[shape0],
        output_dtypes=[effective_output_dtype],
    )[0]
    return out


def run_fused_elementwise(
    descriptor: FusedGraph | dict[str, Any],
    inputs: Sequence[Any],
    *,
    output_dtype: Any | None = None,
    compute_dtype: Any | None = None,
    threadgroup_x: int = 256,
    cache: FusionCompileCache | None = None,
    header: str = DEFAULT_HEADER,
    ensure_row_contiguous: bool = True,
    kernel_name: str | None = None,
) -> Any:
    """Sequence-based wrapper over launch_fused_elementwise()."""

    return launch_fused_elementwise(
        descriptor,
        list(inputs),
        output_dtype=output_dtype,
        compute_dtype=compute_dtype,
        threadgroup_x=threadgroup_x,
        cache=cache,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        kernel_name=kernel_name,
    )


def compile_and_launch_fused_elementwise(
    descriptor: FusedGraph | dict[str, Any],
    inputs: Sequence[Any],
    *,
    output_dtype: Any | None = None,
    compute_dtype: Any | None = None,
    threadgroup_x: int = 256,
    cache: FusionCompileCache | None = None,
    header: str = DEFAULT_HEADER,
    ensure_row_contiguous: bool = True,
    kernel_name: str | None = None,
) -> Any:
    """Compatibility alias for one-shot fused elementwise execution."""

    return run_fused_elementwise(
        descriptor,
        inputs,
        output_dtype=output_dtype,
        compute_dtype=compute_dtype,
        threadgroup_x=threadgroup_x,
        cache=cache,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        kernel_name=kernel_name,
    )


@dataclass(frozen=True)
class CompiledFusedReductionTwoPass:
    """Compiled two-pass reduction fusion kernels."""

    cache_key: FusionCompileKey
    op_signature: str
    reduce_op: str
    axis: int
    keepdims: bool
    no_fma: bool
    rhs_mode: str
    source_pass1: str
    source_pass2: str
    pass1_kernel_name: str
    pass2_kernel_name: str
    rows: int
    reduce_extent: int
    partial_cols: int
    output_shape: tuple[int, ...]
    pass1_kernel: Any
    pass2_kernel: Any


@dataclass(frozen=True)
class CompiledQuantizedSharedInputFusion:
    """Compiled decode-friendly quantized shared-input fusion kernel."""

    cache_key: FusionCompileKey
    op_signature: str
    kernel_name: str
    source: str
    bits: int
    group_size: int
    n: int
    k: int
    output_shape: tuple[int, ...]
    kernel: Any


def _normalize_axis(shape: tuple[int, ...], axis: int) -> int:
    rank = len(shape)
    if rank == 0:
        raise ValueError("Cannot reduce scalar input.")
    axis_norm = int(axis)
    if axis_norm < 0:
        axis_norm += rank
    if axis_norm < 0 or axis_norm >= rank:
        raise ValueError(f"axis {axis} out of range for shape {shape}")
    return axis_norm


def _output_shape_for_reduction(shape: tuple[int, ...], *, axis: int, keepdims: bool) -> tuple[int, ...]:
    dims = list(shape)
    if keepdims:
        dims[axis] = 1
    else:
        dims.pop(axis)
    return tuple(dims)


def _rhs_mode_for(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> str:
    if rhs_shape == lhs_shape:
        return "full"
    if rhs_shape == ():
        return "scalar"
    if rhs_shape == lhs_shape[:-1]:
        return "row"
    if rhs_shape == lhs_shape[:-1] + (1,):
        return "row"
    if rhs_shape == (1,):
        return "scalar"
    raise ValueError(
        "Unsupported broadcast shape for fused mul+reduce "
        f"(lhs={lhs_shape}, rhs={rhs_shape})."
    )


def compile_broadcast_mul_reduce_two_pass(
    descriptor: dict[str, Any],
    inputs: Sequence[Any],
    *,
    cache: FusionCompileCache | None = None,
    header: str = DEFAULT_HEADER,
    ensure_row_contiguous: bool = True,
    threadgroup_x: int = 256,
) -> CompiledFusedReductionTwoPass:
    """Compile a two-pass ``mul + reduce`` kernel pair."""

    input_list = _as_input_list(inputs)
    if len(input_list) != 2:
        raise ValueError(
            f"broadcast_mul_reduce expects 2 inputs (lhs, rhs), got {len(input_list)}"
        )

    lhs, rhs = input_list
    lhs_shape = _shape_of(lhs)
    rhs_shape = _shape_of(rhs)
    reduce_op = str(descriptor.get("reduce_op", "sum"))
    keepdims = bool(descriptor.get("keepdims", False))
    no_fma = bool(descriptor.get("no_fma", False))
    axis = _normalize_axis(lhs_shape, int(descriptor.get("axis", -1)))
    if axis != len(lhs_shape) - 1:
        raise ValueError(
            f"Only last-axis reductions are supported (axis={axis}, shape={lhs_shape})."
        )
    if reduce_op not in {"sum", "mean", "max"}:
        raise ValueError(f"Unsupported reduce_op {reduce_op!r}.")

    rhs_mode = _rhs_mode_for(lhs_shape, rhs_shape)
    rows = _prod(lhs_shape[:-1]) if lhs_shape[:-1] else 1
    reduce_extent = int(lhs_shape[-1])
    if reduce_extent <= 0:
        raise ValueError("Reduction extent must be > 0 for fusion.")

    tgx = int(threadgroup_x)
    if tgx <= 0:
        raise ValueError("threadgroup_x must be > 0")
    tgx = max(1, min(tgx, 512, reduce_extent))
    partial_cols = max(1, (reduce_extent + tgx - 1) // tgx)

    generated = generate_broadcast_mul_reduce_two_pass_source(
        rows=rows,
        reduce_extent=reduce_extent,
        partial_cols=partial_cols,
        reduce_op=reduce_op,
        rhs_mode=rhs_mode,
        no_fma=no_fma,
    )
    key = _cache_key_for(op_signature=generated.op_signature, inputs=input_list)

    cache_obj = cache or GLOBAL_FUSION_COMPILE_CACHE
    cached = cache_obj.get(key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    pass1_kernel = metal_kernel(
        name=generated.pass1_kernel_name,
        input_names=["lhs", "rhs"],
        output_names=["partials"],
        source=generated.source_pass1,
        header=header or DEFAULT_HEADER,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )
    pass2_kernel = metal_kernel(
        name=generated.pass2_kernel_name,
        input_names=["partials"],
        output_names=["out"],
        source=generated.source_pass2,
        header=header or DEFAULT_HEADER,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    compiled = CompiledFusedReductionTwoPass(
        cache_key=key,
        op_signature=generated.op_signature,
        reduce_op=reduce_op,
        axis=axis,
        keepdims=keepdims,
        no_fma=no_fma,
        rhs_mode=rhs_mode,
        source_pass1=generated.source_pass1,
        source_pass2=generated.source_pass2,
        pass1_kernel_name=generated.pass1_kernel_name,
        pass2_kernel_name=generated.pass2_kernel_name,
        rows=rows,
        reduce_extent=reduce_extent,
        partial_cols=partial_cols,
        output_shape=_output_shape_for_reduction(lhs_shape, axis=axis, keepdims=keepdims),
        pass1_kernel=pass1_kernel,
        pass2_kernel=pass2_kernel,
    )
    return cache_obj.put(key, compiled)  # type: ignore[no-any-return]


def launch_broadcast_mul_reduce_two_pass(
    descriptor: dict[str, Any],
    *inputs: Any,
    output_dtype: Any | None = None,
    cache: FusionCompileCache | None = None,
    header: str = DEFAULT_HEADER,
    ensure_row_contiguous: bool = True,
    threadgroup_x: int = 256,
) -> Any:
    """Compile and launch two-pass ``mul + reduce`` fusion."""

    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        input_list = list(inputs[0])
    else:
        input_list = list(inputs)
    if len(input_list) != 2:
        raise ValueError("broadcast_mul_reduce runtime expects exactly 2 inputs.")

    compiled = compile_broadcast_mul_reduce_two_pass(
        descriptor,
        input_list,
        cache=cache,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        threadgroup_x=threadgroup_x,
    )

    mx = import_mx()
    lhs = input_list[0]
    out_dtype = output_dtype if output_dtype is not None else lhs.dtype
    tgx = int(threadgroup_x)
    tgx = max(1, min(tgx, 512, compiled.reduce_extent))

    partials = compiled.pass1_kernel(
        lhs,
        input_list[1],
        grid=(tgx, compiled.rows * compiled.partial_cols, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(compiled.rows, compiled.partial_cols)],
        output_dtypes=[mx.float32],
    )[0]
    out_flat = compiled.pass2_kernel(
        partials,
        template=[("O", out_dtype)],
        grid=(tgx, compiled.rows, 1),
        threadgroup=(tgx, 1, 1),
        output_shapes=[(compiled.rows,)],
        output_dtypes=[out_dtype],
    )[0]
    return out_flat.reshape(compiled.output_shape)


def _validate_quantized_shared_input_inputs(
    descriptor: dict[str, Any],
    inputs: list[Any],
) -> tuple[int, int, int, int, Any, Any, Any, Any, Any, Any, Any]:
    if len(inputs) != 7:
        raise ValueError(
            "quantized_shared_input_swiglu expects 7 inputs "
            "(x, gate_w, gate_scales, gate_biases, up_w, up_scales, up_biases)."
        )
    x, gate_w, gate_scales, gate_biases, up_w, up_scales, up_biases = inputs

    mode = str(descriptor.get("mode", ""))
    transpose = bool(descriptor.get("transpose", False))
    bits = int(descriptor.get("bits", 0))
    group_size = int(descriptor.get("group_size", 0))
    if mode != "affine" or not transpose:
        raise ValueError("Only affine + transpose=True quantized fusion is supported.")
    if bits not in (4, 8):
        raise ValueError("Only 4-bit/8-bit quantized fusion is supported.")
    if group_size <= 0:
        raise ValueError("group_size must be > 0.")

    x_shape = _shape_of(x)
    if len(x_shape) == 1:
        m = 1
        k = int(x_shape[0])
        x_vec = x
    elif len(x_shape) == 2 and int(x_shape[0]) == 1:
        m = 1
        k = int(x_shape[1])
        x_vec = x.reshape((k,))
    else:
        raise ValueError(
            f"Decode-only M=1 shape is required for quantized fusion, got x.shape={x_shape}."
        )

    gate_w_shape = _shape_of(gate_w)
    up_w_shape = _shape_of(up_w)
    if gate_w_shape != up_w_shape or len(gate_w_shape) != 2:
        raise ValueError("gate_w/up_w must be matching rank-2 packed tensors.")
    n = int(gate_w_shape[0])
    k_packed = int(gate_w_shape[1])
    elements_per_word = 32 // bits
    if k_packed * elements_per_word != k:
        raise ValueError("Packed weight shape does not match x K dimension.")

    if _shape_of(gate_scales) != _shape_of(up_scales):
        raise ValueError("gate_scales and up_scales must match.")
    if _shape_of(gate_biases) != _shape_of(up_biases):
        raise ValueError("gate_biases and up_biases must match.")
    expected_groups = k // group_size
    if _shape_of(gate_scales) != (n, expected_groups):
        raise ValueError(
            "Unsupported scales shape for quantized fusion "
            f"(got {_shape_of(gate_scales)}, expected {(n, expected_groups)})."
        )
    if _shape_of(gate_biases) != (n, expected_groups):
        raise ValueError(
            "Unsupported biases shape for quantized fusion "
            f"(got {_shape_of(gate_biases)}, expected {(n, expected_groups)})."
        )
    if m != 1:
        raise ValueError("Only M=1 is supported in this phase.")

    return (
        n,
        k,
        bits,
        group_size,
        x_vec,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
    )


def compile_quantized_shared_input_swiglu(
    descriptor: dict[str, Any],
    inputs: Sequence[Any],
    *,
    cache: FusionCompileCache | None = None,
    header: str = DEFAULT_HEADER,
    ensure_row_contiguous: bool = True,
) -> CompiledQuantizedSharedInputFusion:
    """Compile decode-oriented quantized shared-input fusion."""

    input_list = _as_input_list(inputs)
    (
        n,
        k,
        bits,
        group_size,
        _x_vec,
        _gate_w,
        _gate_scales,
        _gate_biases,
        _up_w,
        _up_scales,
        _up_biases,
    ) = _validate_quantized_shared_input_inputs(descriptor, input_list)
    no_fma = bool(descriptor.get("no_fma", False))

    generated: GeneratedQuantizedSharedInputSource = generate_quantized_shared_input_source(
        n=n,
        k=k,
        bits=bits,
        group_size=group_size,
        no_fma=no_fma,
    )
    key = _cache_key_for(op_signature=generated.op_signature, inputs=input_list)
    cache_obj = cache or GLOBAL_FUSION_COMPILE_CACHE
    cached = cache_obj.get(key)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    kernel = metal_kernel(
        name=generated.kernel_name,
        input_names=[
            "x",
            "gate_w",
            "gate_scales",
            "gate_biases",
            "up_w",
            "up_scales",
            "up_biases",
        ],
        output_names=["out"],
        source=generated.source,
        header=header or DEFAULT_HEADER,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    compiled = CompiledQuantizedSharedInputFusion(
        cache_key=key,
        op_signature=generated.op_signature,
        kernel_name=generated.kernel_name,
        source=generated.source,
        bits=bits,
        group_size=group_size,
        n=n,
        k=k,
        output_shape=(1, n),
        kernel=kernel,
    )
    return cache_obj.put(key, compiled)  # type: ignore[no-any-return]


def launch_quantized_shared_input_swiglu(
    descriptor: dict[str, Any],
    *inputs: Any,
    output_dtype: Any | None = None,
    cache: FusionCompileCache | None = None,
    header: str = DEFAULT_HEADER,
    ensure_row_contiguous: bool = True,
) -> Any:
    """Compile and launch decode-oriented quantized shared-input fusion."""

    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        input_list = list(inputs[0])
    else:
        input_list = list(inputs)
    (
        _n,
        _k,
        _bits,
        _group_size,
        x_vec,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
    ) = _validate_quantized_shared_input_inputs(descriptor, input_list)

    compiled = compile_quantized_shared_input_swiglu(
        descriptor,
        input_list,
        cache=cache,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
    )

    mx = import_mx()
    out_dtype = output_dtype if output_dtype is not None else mx.float32
    out = compiled.kernel(
        x_vec,
        gate_w,
        gate_scales,
        gate_biases,
        up_w,
        up_scales,
        up_biases,
        template=[("O", out_dtype)],
        grid=(compiled.n, 1, 1),
        threadgroup=(min(compiled.n, 256), 1, 1),
        output_shapes=[(compiled.n,)],
        output_dtypes=[out_dtype],
    )[0]
    return out.reshape(compiled.output_shape)


def export_compiled_aot_payload(
    *,
    descriptor: dict[str, Any],
    compiled: CompiledFusedElementwise
    | CompiledFusedReductionTwoPass
    | CompiledQuantizedSharedInputFusion,
) -> dict[str, Any]:
    """Return a minimal AOT scaffold payload (descriptor + source)."""

    if isinstance(compiled, CompiledFusedElementwise):
        sources = {"kernel": compiled.source}
        kernels = {"kernel_name": compiled.kernel_name}
    elif isinstance(compiled, CompiledFusedReductionTwoPass):
        sources = {
            "pass1": compiled.source_pass1,
            "pass2": compiled.source_pass2,
        }
        kernels = {
            "pass1_kernel_name": compiled.pass1_kernel_name,
            "pass2_kernel_name": compiled.pass2_kernel_name,
        }
    else:
        sources = {"kernel": compiled.source}
        kernels = {"kernel_name": compiled.kernel_name}
    return {
        "descriptor": descriptor,
        "kernels": kernels,
        "sources": sources,
    }


__all__ = [
    "CompiledFusedElementwise",
    "CompiledFusedReductionTwoPass",
    "CompiledQuantizedSharedInputFusion",
    "compile_fused_elementwise",
    "launch_fused_elementwise",
    "run_fused_elementwise",
    "compile_and_launch_fused_elementwise",
    "compile_broadcast_mul_reduce_two_pass",
    "launch_broadcast_mul_reduce_two_pass",
    "compile_quantized_shared_input_swiglu",
    "launch_quantized_shared_input_swiglu",
    "export_compiled_aot_payload",
]
