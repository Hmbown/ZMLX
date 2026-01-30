"""IR → MetalKernel lowering.

Builds executable ``MetalKernel`` instances from IR nodes, optionally
wrapping them in ``@mx.custom_function`` for differentiability.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .ir import KernelIR, VJPSpec
from .metal import kernel as metal_kernel
from .msl import DEFAULT_HEADER


def build_kernel(
    name: str,
    ir_node: KernelIR,
    *,
    header: str = DEFAULT_HEADER,
    ensure_row_contiguous: bool = True,
    cache: bool = True,
) -> Any:
    """Lower an IR node to a MetalKernel.

    Args:
        name: Kernel name (used for caching and diagnostics).
        ir_node: The IR node describing the kernel.
        header: Metal header to prepend (default: ZMLX's standard helpers).
        ensure_row_contiguous: Whether to enforce row-contiguous inputs.
        cache: Whether to cache the compiled kernel.

    Returns:
        A callable MetalKernel.
    """
    source = ir_node.to_source()
    return metal_kernel(
        name=name,
        input_names=list(ir_node.input_names),
        output_names=list(ir_node.output_names),
        source=source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=cache,
    )


def build_differentiable(
    name: str,
    ir_node: KernelIR,
    *,
    compute_dtype: Any = mx.float32,
    header: str = DEFAULT_HEADER,
) -> Any:
    """Build a differentiable op from an IR node with VJP.

    The IR node must have a ``vjp`` attribute containing a :class:`VJPSpec`.

    Args:
        name: Kernel name.
        ir_node: IR node with ``vjp`` set.
        compute_dtype: Template specialization dtype.
        header: Metal header.

    Returns:
        A callable that, when called with MLX arrays, produces outputs
        and supports ``mx.grad()`` through a custom VJP.
    """
    vjp_spec: VJPSpec | None = getattr(ir_node, "vjp", None)
    if vjp_spec is None:
        raise ValueError(f"IR node for {name!r} has no VJP spec; use build_kernel instead")

    fwd_kernel = build_kernel(name, ir_node, header=header)

    bwd_header = vjp_spec.header or header
    if vjp_spec.prelude:
        bwd_header = bwd_header + "\n" + vjp_spec.prelude

    bwd_kernel = metal_kernel(
        name=f"{name}_bwd",
        input_names=list(vjp_spec.bwd_input_names),
        output_names=list(vjp_spec.bwd_output_names),
        source=vjp_spec.bwd_source,
        header=bwd_header,
        ensure_row_contiguous=True,
        cache=True,
    )

    return fwd_kernel, bwd_kernel
