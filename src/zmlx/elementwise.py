from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ._compat import import_mx
from .codegen import elementwise_binary_source
from .metal import kernel as metal_kernel
from .msl import DEFAULT_HEADER


def unary(
    *,
    name: str,
    expr: str,
    compute_dtype: Any,
    ensure_row_contiguous: bool = True,
    header: str = "",
    cache: bool = True,
) -> Callable[[Any], Any]:
    """Create a unary elementwise op.

    Parameters
    ----------
    name:
        Kernel name (used for caching + debug)
    expr:
        Metal expression that writes the output value. You may reference:
        - `x` (a compute value of type `T`)
        - `inp` / `out` (raw device pointers; advanced)
    compute_dtype:
        Dtype used for compute (`T` template). Commonly `mx.float32`.
    """

    source = f"""
        uint elem = thread_position_in_grid.x;
        T x = inp[elem];
        out[elem] = {expr};
    """

    k = metal_kernel(
        name=name,
        input_names=["inp"],
        output_names=["out"],
        source=source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=cache,
    )

    def op(a: Any) -> Any:
        outputs = k(
            a,
            template=[("T", compute_dtype)],
            output_shapes=[a.shape],
            output_dtypes=[a.dtype],
        )
        return outputs[0]

    return op


def binary(
    *,
    name: str,
    expr: str,
    compute_dtype: Any,
    ensure_row_contiguous: bool = True,
    header: str = "",
) -> Callable[[Any, Any], Any]:
    """Create a binary elementwise op from a Metal expression."""
    import_mx()
    source = elementwise_binary_source(expr=expr)
    k = metal_kernel(
        name=name,
        input_names=["lhs", "rhs"],
        output_names=["out"],
        source=source,
        header=header or DEFAULT_HEADER,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    def op(lhs: Any, rhs: Any) -> Any:
        if lhs.shape != rhs.shape:
            # simple broadcasting check - MLX fast metal kernels don't broadcast automatically
            # unless we handle it in codegen. For now, require same shape.
            raise ValueError(f"{name}: shapes must match (got {lhs.shape} vs {rhs.shape})")

        out = k(
            lhs,
            rhs,
            template=[("T", compute_dtype)],
            output_shapes=[lhs.shape],
            output_dtypes=[lhs.dtype],
        )[0]
        return out

    return op


def map(
    *,
    name: str,
    expr: str,
    input_names: Sequence[str],
    compute_dtype: Any,
    ensure_row_contiguous: bool = True,
    header: str = "",
) -> Callable[..., Any]:
    """Create an N-ary elementwise op from a Metal expression.
    
    The expression can reference each input by its name in `input_names`.
    """
    import_mx()
    
    # We use a clever trick: we map the user's names to internal names
    # to avoid shadowing the input pointers in the kernel signature.
    internal_names = [f"_in_{i}" for i in range(len(input_names))]
    
    input_reads = "\n".join([
        f"        T {user_name} = {internal_name}[elem];" 
        for user_name, internal_name in zip(input_names, internal_names, strict=True)
    ])
    
    source = f"""
        uint elem = thread_position_in_grid.x;
{input_reads}
        out[elem] = {expr};
    """
    
    k = metal_kernel(
        name=name,
        input_names=internal_names,
        output_names=["out"],
        source=source,
        header=header or DEFAULT_HEADER,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    def op(*inputs: Any) -> Any:
        if len(inputs) != len(input_names):
            raise ValueError(f"{name}: expected {len(input_names)} inputs, got {len(inputs)}")
        
        shape0 = inputs[0].shape
        dtype0 = inputs[0].dtype
        
        out = k(
            *inputs,
            template=[("T", compute_dtype)],
            output_shapes=[shape0],
            output_dtypes=[dtype0],
        )[0]
        return out

    return op
