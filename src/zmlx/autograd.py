from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ._compat import import_mx
from .metal import kernel as metal_kernel


def unary_from_expr(
    *,
    name: str,
    fwd_expr: str,
    vjp_expr: str,
    compute_dtype: Any,
    ensure_row_contiguous: bool = True,
    use_output: bool = True,
    header: str = "",
    vjp_prelude: str = "",
) -> Callable[[Any], Any]:
    """Create a unary op implemented as a Metal kernel with a custom VJP.

    Parameters
    ----------
    name:
        Base name for the op; forward/backward kernels will be named `{name}_fwd` and `{name}_bwd`.
    fwd_expr:
        Metal expression for forward output, written in terms of:
        - `x` (compute value of type `T` read from `inp[elem]`)
    vjp_expr:
        Metal expression for `dinp[elem]`.
        Available symbols depend on `use_output`:
        - always: `g` (cotangent / upstream gradient)
        - if use_output=True: `y` (forward output)
        - if use_output=False: `x` (input compute value)
    compute_dtype:
        Dtype used for compute (`T` template). Commonly `mx.float32`.
    use_output:
        If True, backward kernel consumes the forward output (`y`) to avoid recomputation.
        If False, backward kernel consumes the input (`x`) and recomputes what it needs.
    vjp_prelude:
        Optional Metal statements inserted before `dinp[elem] = ...;` in the backward kernel.
        Useful for precomputing shared subexpressions, e.g. `T sig = ...;`.

    Returns
    -------
    A Python callable that behaves like an MLX op and participates in `mx.grad` via custom VJP.
    """
    mx = import_mx()

    fwd_source = f"""
        uint elem = thread_position_in_grid.x;
        T x = inp[elem];
        out[elem] = (O)({fwd_expr});
    """

    fwd = metal_kernel(
        name=f"{name}_fwd",
        input_names=["inp"],
        output_names=["out"],
        source=fwd_source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    prelude = (vjp_prelude or "").strip()
    prelude_lines = ("\n" + prelude + "\n") if prelude else "\n"

    if use_output:
        bwd_source = (
            "\n            uint elem = thread_position_in_grid.x;\n"
            "            T y = out[elem];\n"
            "            T g = cotan[elem];\n"
            + prelude_lines
            + f"            dinp[elem] = (O)({vjp_expr});\n        "
        )
        bwd = metal_kernel(
            name=f"{name}_bwd",
            input_names=["out", "cotan"],
            output_names=["dinp"],
            source=bwd_source,
            header=header,
            ensure_row_contiguous=ensure_row_contiguous,
            cache=True,
        )
    else:
        bwd_source = (
            "\n            uint elem = thread_position_in_grid.x;\n"
            "            T x = inp[elem];\n"
            "            T g = cotan[elem];\n"
            + prelude_lines
            + f"            dinp[elem] = (O)({vjp_expr});\n        "
        )
        bwd = metal_kernel(
            name=f"{name}_bwd",
            input_names=["inp", "cotan"],
            output_names=["dinp"],
            source=bwd_source,
            header=header,
            ensure_row_contiguous=ensure_row_contiguous,
            cache=True,
        )

    @mx.custom_function
    def op(inp: Any) -> Any:
        out = fwd(
            inp,
            template=[("T", compute_dtype), ("O", inp.dtype)],
            output_shapes=[inp.shape],
            output_dtypes=[inp.dtype],
        )[0]
        return out

    @op.vjp
    def op_vjp(primals: Any, cotangents: Any, outputs: Any):
        def _get_first(x):
            if isinstance(x, (list, tuple)):
                return x[0]
            return x

        inp = _get_first(primals)
        cotan = _get_first(cotangents)
        out = _get_first(outputs)

        if use_output:
            dx = bwd(
                out,
                cotan,
                template=[("T", compute_dtype), ("O", inp.dtype)],
                output_shapes=[inp.shape],
                output_dtypes=[inp.dtype],
            )[0]
        else:
            dx = bwd(
                inp,
                cotan,
                template=[("T", compute_dtype), ("O", inp.dtype)],
                output_shapes=[inp.shape],
                output_dtypes=[inp.dtype],
            )[0]

        return (dx,)

    return op  # type: ignore[no-any-return]


def binary_from_expr(
    *,
    name: str,
    fwd_expr: str,
    vjp_lhs_expr: str,
    vjp_rhs_expr: str,
    compute_dtype: Any,
    ensure_row_contiguous: bool = True,
    header: str = "",
    vjp_prelude: str = "",
) -> Callable[[Any, Any], Any]:
    """Create a binary op implemented as a Metal kernel with a custom VJP for both inputs.

    Forward:
        out[elem] = fwd_expr   (in terms of `a` and `b`)

    Backward:
        dlhs[elem] = vjp_lhs_expr
        drhs[elem] = vjp_rhs_expr

    Available symbols in VJP expressions:
      - `a`, `b`: input compute values
      - `g`: cotangent / upstream gradient
      - You may define temporaries in `vjp_prelude` (Metal statements)

    Template parameters:
      - `T`: compute dtype (e.g. float32 for precision)
      - `O`: output/storage dtype (matches input dtype, e.g. bfloat16)
    """
    mx = import_mx()

    fwd_source = f"""
        uint elem = thread_position_in_grid.x;
        T a = lhs[elem];
        T b = rhs[elem];
        {vjp_prelude if vjp_prelude.strip() else ""}
        out[elem] = (O)({fwd_expr});
    """

    fwd = metal_kernel(
        name=f"{name}_fwd",
        input_names=["lhs", "rhs"],
        output_names=["out"],
        source=fwd_source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    prelude = (vjp_prelude or "").strip()
    prelude_lines = ("\n" + prelude + "\n") if prelude else "\n"

    bwd_source = (
        "\n            uint elem = thread_position_in_grid.x;\n"
        "            T a = lhs[elem];\n"
        "            T b = rhs[elem];\n"
        "            T g = cotan[elem];\n"
        + prelude_lines
        + f"            dlhs[elem] = (O)({vjp_lhs_expr});\n"
        + f"            drhs[elem] = (O)({vjp_rhs_expr});\n        "
    )

    bwd = metal_kernel(
        name=f"{name}_bwd",
        input_names=["lhs", "rhs", "cotan"],
        output_names=["dlhs", "drhs"],
        source=bwd_source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    @mx.custom_function
    def op(lhs: Any, rhs: Any) -> Any:
        if lhs.shape != rhs.shape:
            raise ValueError(f"{name}: shapes must match (got {lhs.shape} vs {rhs.shape})")
        if lhs.dtype != rhs.dtype:
            raise ValueError(f"{name}: dtypes must match (got {lhs.dtype} vs {rhs.dtype})")

        out = fwd(
            lhs,
            rhs,
            template=[("T", compute_dtype), ("O", lhs.dtype)],
            output_shapes=[lhs.shape],
            output_dtypes=[lhs.dtype],
        )[0]
        return out

    @op.vjp
    def op_vjp(primals: Any, cotangents: Any, outputs: Any):
        if not isinstance(primals, (list, tuple)) or len(primals) != 2:
            # Fallback for unexpected formats
            lhs = primals[0] if isinstance(primals, (list, tuple)) else primals
            rhs = primals[1] if isinstance(primals, (list, tuple)) and len(primals) > 1 else None
        else:
            lhs, rhs = primals

        def _get_first(x):
            if isinstance(x, (list, tuple)):
                return x[0]
            return x

        cotan = _get_first(cotangents)

        dlhs, drhs = bwd(
            lhs,
            rhs,
            cotan,
            template=[("T", compute_dtype), ("O", lhs.dtype)],  # type: ignore[union-attr]
            output_shapes=[lhs.shape, rhs.shape],  # type: ignore[union-attr]
            output_dtypes=[lhs.dtype, rhs.dtype],  # type: ignore[union-attr]
        )

        return (dlhs, drhs)

    return op  # type: ignore[no-any-return]


def nary_from_expr(
    *,
    name: str,
    fwd_source: str,
    bwd_source: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    compute_dtype: Any,
    ensure_row_contiguous: bool = True,
    header: str = "",
) -> Callable[..., Any]:
    """Create an N-ary op with custom VJP using raw Metal source.

    VJP expects `bwd_source` to consume `input_names` + `cotan`
    and produce gradients for all inputs.
    """
    mx = import_mx()

    fwd = metal_kernel(
        name=f"{name}_fwd",
        input_names=input_names,
        output_names=output_names,
        source=fwd_source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    bwd_input_names = list(input_names) + ["cotan"]
    bwd_output_names = [f"d{n}" for n in input_names]

    bwd = metal_kernel(
        name=f"{name}_bwd",
        input_names=bwd_input_names,
        output_names=bwd_output_names,
        source=bwd_source,
        header=header,
        ensure_row_contiguous=ensure_row_contiguous,
        cache=True,
    )

    @mx.custom_function
    def op(*inputs: Any) -> Any:
        outs = fwd(
            *inputs,
            template=[("T", compute_dtype)],
            output_shapes=[inputs[0].shape for _ in output_names],
            output_dtypes=[inputs[0].dtype for _ in output_names],
        )
        return outs[0] if len(outs) == 1 else tuple(outs)

    @op.vjp
    def op_vjp(primals: Any, cotangents: Any, outputs: Any):
        # inputs
        inputs = list(primals)
        # Handle multiple cotangents if multiple outputs
        cotan = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents

        grads = bwd(
            *inputs, cotan,
            template=[("T", compute_dtype)],
            output_shapes=[x.shape for x in inputs],
            output_dtypes=[x.dtype for x in inputs],
        )
        return tuple(grads)

    return op  # type: ignore[no-any-return]
