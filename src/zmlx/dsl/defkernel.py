"""Declarative kernel definitions.

``defkernel()`` produces a ``@cache``-decorated builder that returns a
ready-to-call kernel function — the same objects produced by
``elementwise.unary()`` and ``autograd.unary_from_expr()``, with zero new
runtime overhead.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any, Literal

from .expr import Expr


def _resolve_expr(expr: str | Expr) -> str:
    """Convert an Expr to its Metal string, or pass strings through."""
    if isinstance(expr, Expr):
        return expr.to_metal()
    return expr


def _resolve_prelude(expr: str | Expr | None) -> str:
    """Extract prelude lines from Let bindings if expr is an Expr tree."""
    if expr is None:
        return ""
    if isinstance(expr, str):
        return ""
    # Walk Let bindings to extract preludes
    from .expr import collect_lets

    bindings, _ = collect_lets(expr)
    if not bindings:
        return ""
    return "\n".join(b.to_metal_prelude() for b in bindings)


def defkernel(
    name: str,
    expr: str | Expr,
    *,
    vjp_expr: str | Expr | None = None,
    kind: Literal["unary", "binary"] = "unary",
    use_output: bool = True,
    vjp_prelude: str = "",
    header: str | None = None,
    compute_dtype_default: str = "float32",
) -> Callable[..., Callable]:
    """Define an elementwise kernel declaratively.

    Returns a ``@cache``-decorated builder ``(compute_dtype_key=...) -> kernel``.

    When ``vjp_expr`` is provided, the result is a differentiable op
    (via ``autograd.unary_from_expr`` or ``autograd.binary_from_expr``).
    Otherwise it's a plain forward-only kernel.

    Parameters
    ----------
    name:
        Kernel name (e.g. ``"kk_silu"``).
    expr:
        Forward expression — a Metal string or an ``Expr`` tree.
    vjp_expr:
        Backward expression, or ``None`` for forward-only kernels.
    kind:
        ``"unary"`` or ``"binary"``.
    use_output:
        For VJP: if True, backward sees ``y`` (forward output);
        if False, backward sees ``x`` (input).
    vjp_prelude:
        Extra Metal statements before the backward write.
        If ``vjp_expr`` is a ``Let``-based Expr, preludes are
        extracted automatically and merged.
    header:
        Metal header. Defaults to ``DEFAULT_HEADER``.
    compute_dtype_default:
        Default compute dtype key.
    """
    fwd_str = _resolve_expr(expr)
    bwd_str = _resolve_expr(vjp_expr) if vjp_expr is not None else None

    # Merge explicit prelude with any Let-extracted preludes
    auto_prelude = _resolve_prelude(vjp_expr) if vjp_expr is not None else ""
    merged_prelude = (vjp_prelude + "\n" + auto_prelude).strip() if auto_prelude else vjp_prelude

    if kind == "unary":
        return _make_unary_builder(
            name=name,
            fwd_expr=fwd_str,
            vjp_expr=bwd_str,
            use_output=use_output,
            vjp_prelude=merged_prelude,
            header=header,
            compute_dtype_default=compute_dtype_default,
        )

    return _make_binary_builder(
        name=name,
        fwd_expr=fwd_str,
        vjp_expr=bwd_str,
        use_output=use_output,
        vjp_prelude=merged_prelude,
        header=header,
        compute_dtype_default=compute_dtype_default,
    )


def _make_unary_builder(
    *,
    name: str,
    fwd_expr: str,
    vjp_expr: str | None,
    use_output: bool,
    vjp_prelude: str,
    header: str | None,
    compute_dtype_default: str,
) -> Callable[..., Callable]:
    """Create a cached unary kernel builder."""

    @cache
    def builder(*, compute_dtype_key: str = compute_dtype_default) -> Callable[[Any], Any]:
        from .._compat import import_mx
        from ..msl import DEFAULT_HEADER

        mx = import_mx()
        compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
        hdr = header if header is not None else DEFAULT_HEADER

        if vjp_expr is not None:
            from ..autograd import unary_from_expr

            return unary_from_expr(
                name=f"{name}_{compute_dtype_key}",
                fwd_expr=fwd_expr,
                vjp_expr=vjp_expr,
                compute_dtype=compute_dtype,
                use_output=use_output,
                header=hdr,
                vjp_prelude=vjp_prelude,
            )

        from ..elementwise import unary as unary_kernel

        return unary_kernel(
            name=f"{name}_{compute_dtype_key}",
            expr=fwd_expr,
            compute_dtype=compute_dtype,
            header=hdr,
        )

    return builder


def _make_binary_builder(
    *,
    name: str,
    fwd_expr: str,
    vjp_expr: str | None,
    use_output: bool,  # noqa: ARG001 — reserved for future binary VJP modes
    vjp_prelude: str,
    header: str | None,
    compute_dtype_default: str,
) -> Callable[..., Callable]:
    """Create a cached binary kernel builder."""

    @cache
    def builder(*, compute_dtype_key: str = compute_dtype_default) -> Callable:
        from .._compat import import_mx
        from ..msl import DEFAULT_HEADER

        mx = import_mx()
        compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
        hdr = header if header is not None else DEFAULT_HEADER

        if vjp_expr is not None:
            from ..autograd import binary_from_expr

            return binary_from_expr(
                name=f"{name}_{compute_dtype_key}",
                fwd_expr=fwd_expr,
                vjp_lhs_expr=vjp_expr,
                vjp_rhs_expr=vjp_expr,
                compute_dtype=compute_dtype,
                header=hdr,
                vjp_prelude=vjp_prelude,
            )

        from ..elementwise import binary as binary_kernel

        return binary_kernel(
            name=f"{name}_{compute_dtype_key}",
            expr=fwd_expr,
            compute_dtype=compute_dtype,
            header=hdr,
        )

    return builder


def defkernel_mapreduce(
    name: str,
    *,
    pass1: dict[str, str],
    pass2: dict[str, str],
    write_expr: str | Expr,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    extra_source: str = "",
    header: str | None = None,
) -> Callable[..., Callable]:
    """Define a rowwise map-reduce kernel declaratively.

    Parameters
    ----------
    name:
        Kernel name.
    pass1:
        ``{"init": ..., "update": ..., "reduce_op": ...}`` for pass 1.
    pass2:
        ``{"init": ..., "update": ..., "reduce_op": ...}`` for pass 2.
    write_expr:
        Per-element output expression (Metal string or Expr).
    """
    write_str = _resolve_expr(write_expr)

    @cache
    def builder(
        *, d: int, tg: int = 256, compute_dtype_key: str = "float32"
    ) -> Callable:
        from .._compat import import_mx
        from ..codegen import rowwise_mapreduce_source
        from ..metal import kernel as metal_kernel
        from ..msl import DEFAULT_HEADER

        mx = import_mx()
        compute_dtype = mx.float32 if compute_dtype_key == "float32" else mx.float16
        hdr = header if header is not None else DEFAULT_HEADER

        source = rowwise_mapreduce_source(
            d=d,
            tg=tg,
            pass1_init=pass1["init"],
            pass1_update=pass1["update"],
            pass1_reduce_op=pass1["reduce_op"],
            pass2_init=pass2["init"],
            pass2_update=pass2["update"],
            pass2_reduce_op=pass2["reduce_op"],
            write_expr=write_str,
        )

        if extra_source:
            source = extra_source + "\n" + source

        in_names = input_names or ["inp"]
        out_names = output_names or ["out"]

        k = metal_kernel(
            name=name,
            input_names=in_names,
            output_names=out_names,
            source=source,
            header=hdr,
            ensure_row_contiguous=True,
            cache=True,
        )

        def op(*inputs: Any) -> Any:
            shape0 = inputs[0].shape
            n_rows = 1
            for s in shape0[:-1]:
                n_rows *= s

            outputs = k(
                *inputs,
                template=[("T", compute_dtype)],
                grid=(n_rows * tg, 1, 1),
                threadgroup=(tg, 1, 1),
                output_shapes=[shape0],
                output_dtypes=[inputs[0].dtype],
            )
            return outputs[0]

        return op

    return builder
