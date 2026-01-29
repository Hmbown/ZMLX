"""Tiny IR/codegen helpers.

This is intentionally *minimal* — the goal is to encode a few common patterns
(elementwise, rowwise map-reduce) in a consistent way so we can:
- generate many kernels cheaply
- keep caching predictable (source string is the cache key)
- later share the same patterns with a Zig frontend

If you want a larger IR, see `docs/ARCHITECTURE.md` for the planned direction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElementwiseSpec:
    name: str
    expr: str
    arity: int  # 1 or 2
    input_names: tuple[str, ...]
    output_names: tuple[str, ...] = ("out",)


def elementwise_unary_source(*, expr: str, inp: str = "inp", out: str = "out") -> str:
    """Generate an elementwise unary kernel body.

    Args:
        expr: Metal expression to compute the output value.
        inp: Input buffer name in the generated source.
        out: Output buffer name in the generated source.

    Returns:
        A Metal Shading Language source snippet for the kernel body.
    """
    return f"""
        uint elem = thread_position_in_grid.x;
        T x = {inp}[elem];
        {out}[elem] = {expr};
    """


def elementwise_binary_source(
    *,
    expr: str,
    lhs: str = "lhs",
    rhs: str = "rhs",
    out: str = "out",
) -> str:
    """Generate an elementwise binary kernel body.

    Args:
        expr: Metal expression to compute the output value.
        lhs: Left-hand input buffer name in the generated source.
        rhs: Right-hand input buffer name in the generated source.
        out: Output buffer name in the generated source.

    Returns:
        A Metal Shading Language source snippet for the kernel body.
    """
    return f"""
        uint elem = thread_position_in_grid.x;
        T a = {lhs}[elem];
        T b = {rhs}[elem];
        T x = a;
        {out}[elem] = {expr};
    """


def rowwise_reduction_source(
    *,
    reduce_expr: str,
    init_expr: str,
    finalize_expr: str,
    d: int,
    inp: str = "inp",
    out: str = "out",
) -> str:
    """Generate a simple rowwise reduction over the last dimension.

    Launch convention:
        - ``grid.x == rows`` (one thread per row)
        - ``threadgroup.x`` can be 1 (no parallelism)

    This is correct but not necessarily fast; it is useful for \"get it working\"
    kernels and reference paths.

    Args:
        reduce_expr: C expression for updating the accumulator.
        init_expr: Initial value for the accumulator.
        finalize_expr: Expression to finalize the accumulator into the output.
        d: Row width (last-dimension size).
        inp: Input buffer name in the generated source.
        out: Output buffer name in the generated source.

    Returns:
        A Metal Shading Language source string ready to pass to
        :func:`zmlx.metal.kernel`.
    """
    D = int(d)
    return f"""
        uint row = thread_position_in_grid.x;
        uint base = row * {D};
        T acc = {init_expr};
        for (uint j = 0; j < {D}; ++j) {{
            T v = {inp}[base + j];
            acc = {reduce_expr};
        }}
        {out}[row] = {finalize_expr};
    """


def rowwise_parallel_reduction_source(
    *,
    d: int,
    tg: int,
    init_expr: str,
    update_expr: str,
    reduce_op: str,
    finalize_expr: str,
    inp: str = "inp",
    out: str = "out",
    scratch: str = "buf",
) -> str:
    """Generate a rowwise parallel reduction kernel (1 scalar pass + 1 output write).

    Launch convention:
        - ``threadgroup.x == TG``
        - ``grid.x == rows * TG``

    Args:
        d: Row width (last-dimension size).
        tg: Threadgroup size (must be a power of two).
        init_expr: Initial value for the accumulator.
        update_expr: Update expression (available vars: ``acc``, ``x``).
        reduce_op: Threadgroup reduction operator (available vars: ``a``, ``b``).
        finalize_expr: Expression to finalize the accumulator into the output.
        inp: Input buffer name in the generated source.
        out: Output buffer name in the generated source.
        scratch: Threadgroup scratch buffer name.

    Returns:
        A Metal Shading Language source string ready to pass to
        :func:`zmlx.metal.kernel`.
    """
    D = int(d)
    TG = int(tg)
    return f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float {scratch}[TG];

        float acc = {init_expr};
        for (uint j = tid; j < D; j += TG) {{
            float x = (float){inp}[base + j];
            acc = {update_expr};
        }}
        {scratch}[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                float a = {scratch}[tid];
                float b = {scratch}[tid + stride];
                {scratch}[tid] = {reduce_op};
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        
        if (tid == 0) {{
            float s = {scratch}[0];
            {out}[row] = (T)({finalize_expr});
        }}
    """


def rowwise_mapreduce_source(
    *,
    d: int,
    tg: int,
    # Pass 1: compute a rowwise scalar (e.g., max, sumsq)
    pass1_init: str,
    pass1_update: str,
    pass1_reduce_op: str,  # reduction across threadgroup (e.g. max or +)
    # Pass 2: compute a second scalar (e.g. sum(exp(x-max)))
    pass2_init: str,
    pass2_update: str,
    pass2_reduce_op: str,
    # Write: per-element output expression, can reference:
    #   - x (input scalar)
    #   - s1 (scalar from pass1)
    #   - s2 (scalar from pass2)
    write_expr: str,
    inp: str = "inp",
    out: str = "out",
    scratch1: str = "buf1",
    scratch2: str = "buf2",
) -> str:
    """Generate a rowwise map-reduce kernel (2 scalar passes + elementwise write).

    This is the workhorse behind :func:`zmlx.rowwise.map_reduce` and kernels
    like softmax and layer-norm that need two reductions before writing output.

    Launch convention:
        - ``threadgroup.x == TG``
        - ``grid.x == rows * TG``

    Indexing:
        - ``gid = thread_position_in_grid.x``
        - ``tid = thread_position_in_threadgroup.x``
        - ``row = gid / TG``

    Args:
        d: Row width (last-dimension size, compiled as a constant).
        tg: Threads per threadgroup (compiled as a constant, must be power of 2).
        pass1_init: C expression for the initial value of the pass-1
            accumulator (e.g. ``"-INFINITY"``).
        pass1_update: C update expression (available vars: ``acc1``, ``x``).
        pass1_reduce_op: Threadgroup tree-reduction operator for pass 1
            (available vars: ``a``, ``b``).
        pass2_init: Initial value for the pass-2 accumulator.
        pass2_update: Update expression (available vars: ``acc2``, ``x``,
            ``s1`` — the scalar result of pass 1).
        pass2_reduce_op: Tree-reduction operator for pass 2.
        write_expr: Per-element output expression (available vars: ``x``,
            ``s1``, ``s2``).
        inp: Name of the input buffer in the generated source.
        out: Name of the output buffer in the generated source.
        scratch1: Threadgroup scratch buffer name for pass 1.
        scratch2: Threadgroup scratch buffer name for pass 2.

    Returns:
        A Metal Shading Language source string ready to pass to
        :func:`zmlx.metal.kernel`.
    """
    D = int(d)
    TG = int(tg)
    # We allocate fixed-size scratch arrays (must match TG)
    return f"""
        constexpr uint D = {D};
        constexpr uint TG = {TG};

        uint gid = thread_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x;
        uint row = gid / TG;
        uint base = row * D;

        threadgroup float {scratch1}[TG];
        threadgroup float {scratch2}[TG];

        // pass 1
        float acc1 = {pass1_init};
        for (uint j = tid; j < D; j += TG) {{
            float x = (float){inp}[base + j];
            acc1 = {pass1_update};
        }}
        {scratch1}[tid] = acc1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                float a = {scratch1}[tid];
                float b = {scratch1}[tid + stride];
                {scratch1}[tid] = {pass1_reduce_op};
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float s1 = {scratch1}[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // pass 2
        float acc2 = {pass2_init};
        for (uint j = tid; j < D; j += TG) {{
            float x = (float){inp}[base + j];
            acc2 = {pass2_update};
        }}
        {scratch2}[tid] = acc2;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = TG / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                float a = {scratch2}[tid];
                float b = {scratch2}[tid + stride];
                {scratch2}[tid] = {pass2_reduce_op};
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}
        float s2 = {scratch2}[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // write outputs
        for (uint j = tid; j < D; j += TG) {{
            float x = (float){inp}[base + j];
            {out}[base + j] = (T)({write_expr});
        }}
    """
