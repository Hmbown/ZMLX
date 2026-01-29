from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ._compat import import_mx
from .codegen import rowwise_mapreduce_source
from .kernels.softmax import _validate_tg
from .metal import kernel as metal_kernel
from .msl import DEFAULT_HEADER


def map_reduce(
    *,
    name: str,
    d: int,
    pass1_init: str,
    pass1_update: str,
    pass1_reduce_op: str,
    pass2_init: str,
    pass2_update: str,
    pass2_reduce_op: str,
    write_expr: str,
    input_names: Sequence[str] = ("inp",),
    output_names: Sequence[str] = ("out",),
    threadgroup: int = 256,
    compute_dtype: Any = None,
    header: str = "",
) -> Callable[..., Any]:
    """Create a two-pass rowwise map-reduce kernel.

    High-level wrapper around :func:`codegen.rowwise_mapreduce_source`.  The
    kernel performs two threadgroup-parallel reduction passes over the last
    dimension followed by an elementwise write pass.

    Args:
        name: Unique kernel name (used for caching).
        d: Row width (last-dimension size).
        pass1_init: C expression initialising the pass-1 accumulator.
        pass1_update: C expression updating the pass-1 accumulator (vars:
            ``acc1``, ``x``).
        pass1_reduce_op: Threadgroup reduction operator for pass 1 (vars:
            ``a``, ``b``).
        pass2_init: C expression initialising the pass-2 accumulator.
        pass2_update: C expression updating the pass-2 accumulator (vars:
            ``acc2``, ``x``, ``s1``).
        pass2_reduce_op: Threadgroup reduction operator for pass 2.
        write_expr: Per-element output expression (vars: ``x``, ``s1``, ``s2``).
        input_names: Names of the kernel input buffers.
        output_names: Names of the kernel output buffers.
        threadgroup: Threads per threadgroup (must be a power of 2).
        compute_dtype: MLX dtype for the template parameter ``T``.
        header: Optional MSL header prepended to the source.

    Returns:
        A callable that accepts the input arrays and returns a tuple of output
        arrays.
    """
    mx = import_mx()
    TG = _validate_tg(threadgroup)
    D = int(d)
    
    # For now, codegen.rowwise_mapreduce_source only supports 1 input/1 output.
    # We'll use it as a base.
    src = rowwise_mapreduce_source(
        d=D,
        tg=TG,
        pass1_init=pass1_init,
        pass1_update=pass1_update,
        pass1_reduce_op=pass1_reduce_op,
        pass2_init=pass2_init,
        pass2_update=pass2_update,
        pass2_reduce_op=pass2_reduce_op,
        write_expr=write_expr,
        inp=input_names[0],
        out=output_names[0],
    )
    
    k = metal_kernel(
        name=name,
        input_names=input_names,
        output_names=output_names,
        source=src,
        header=header or DEFAULT_HEADER,
        ensure_row_contiguous=True,
        cache=True,
    )

    def op(*inputs: Any) -> Any:
        x0 = inputs[0]
        rows = x0.size // D
        cd = compute_dtype or mx.float32
        
        return k(
            *inputs,
            template=[("T", cd)],
            grid=(rows * TG, 1, 1),
            threadgroup=(TG, 1, 1),
            output_shapes=[x0.shape for _ in output_names],
            output_dtypes=[x0.dtype for _ in output_names],
        )

    return op
