from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FusedNode:
    """Single node in a fused elementwise DAG."""

    id: str
    op: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class FusedGraph:
    """Normalized fused elementwise graph descriptor."""

    input_names: tuple[str, ...]
    nodes: tuple[FusedNode, ...]
    output: str
    name: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inputs": list(self.input_names),
            "nodes": [
                {
                    "id": node.id,
                    "op": node.op,
                    "args": list(node.args),
                }
                for node in self.nodes
            ],
            "output": self.output,
        }


@dataclass(frozen=True)
class GeneratedFusedSource:
    """Generated source payload used by the runtime compiler."""

    kernel_name: str
    input_names: tuple[str, ...]
    output_name: str
    source: str
    op_signature: str
    graph: FusedGraph


@dataclass(frozen=True)
class GeneratedReductionTwoPassSource:
    """Generated source payload for two-pass reduction fusion."""

    pass1_kernel_name: str
    pass2_kernel_name: str
    source_pass1: str
    source_pass2: str
    op_signature: str
    reduce_op: str
    no_fma: bool
    rows: int
    reduce_extent: int
    partial_cols: int
    rhs_mode: str


@dataclass(frozen=True)
class GeneratedQuantizedSharedInputSource:
    """Generated source payload for decode-oriented quantized QMM fusion."""

    kernel_name: str
    source: str
    op_signature: str
    bits: int
    group_size: int
    n: int
    k: int


_OP_ALIASES: dict[str, str] = {
    "plus": "add",
    "minus": "sub",
    "times": "mul",
    "multiply": "mul",
    "swish": "silu",
    "minimum": "min",
    "maximum": "max",
}


_SUPPORTED_OPS: dict[str, tuple[int, Any]] = {
    "add": (2, lambda a: f"({a[0]} + {a[1]})"),
    "sub": (2, lambda a: f"({a[0]} - {a[1]})"),
    "mul": (2, lambda a: f"({a[0]} * {a[1]})"),
    "div": (2, lambda a: f"({a[0]} / {a[1]})"),
    "neg": (1, lambda a: f"(-({a[0]}))"),
    "exp": (1, lambda a: f"metal::exp({a[0]})"),
    "tanh": (1, lambda a: f"metal::tanh({a[0]})"),
    "sigmoid": (1, lambda a: f"kk_sigmoid({a[0]})"),
    "relu": (1, lambda a: f"metal::max({a[0]}, 0.0f)"),
    "silu": (1, lambda a: f"kk_silu({a[0]})"),
    "log": (1, lambda a: f"metal::log({a[0]})"),
    "sqrt": (1, lambda a: f"metal::sqrt({a[0]})"),
    "rsqrt": (1, lambda a: f"metal::rsqrt({a[0]})"),
    "min": (2, lambda a: f"metal::min({a[0]}, {a[1]})"),
    "max": (2, lambda a: f"metal::max({a[0]}, {a[1]})"),
}


_RHS_MODE_TO_INT: dict[str, int] = {
    "full": 0,
    "row": 1,
    "scalar": 2,
}


def _is_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _canonical_op_name(op: Any) -> str:
    if not isinstance(op, str):
        raise TypeError(f"op must be str, got {type(op).__name__}")
    key = op.strip().lower()
    key = _OP_ALIASES.get(key, key)
    if key not in _SUPPORTED_OPS:
        supported = ", ".join(sorted(_SUPPORTED_OPS))
        raise ValueError(f"Unsupported fusion op {op!r}. Supported ops: {supported}")
    return key


def _extract_input_names(descriptor: dict[str, Any]) -> tuple[str, ...]:
    raw = descriptor.get("input_names")
    if raw is None:
        raw = descriptor.get("inputs")
    if raw is None:
        raw = descriptor.get("args")

    if isinstance(raw, int):
        if raw <= 0:
            raise ValueError("descriptor input count must be > 0")
        return tuple(f"in{i}" for i in range(int(raw)))

    if not _is_seq(raw):
        n = descriptor.get("num_inputs")
        if isinstance(n, int) and n > 0:
            return tuple(f"in{i}" for i in range(int(n)))
        raise ValueError("descriptor must provide 'inputs' or 'input_names'")

    out: list[str] = []
    for i, item in enumerate(raw):
        if isinstance(item, dict):
            name = item.get("name") or item.get("id") or item.get("input")
            if name is None:
                name = f"in{i}"
            out.append(str(name))
        else:
            out.append(str(item))

    if not out:
        raise ValueError("descriptor must define at least one input")

    if len(set(out)) != len(out):
        raise ValueError(f"input names must be unique, got {out}")

    return tuple(out)


def _extract_raw_nodes(descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    raw_nodes = descriptor.get("nodes")
    if raw_nodes is None:
        raw_nodes = descriptor.get("ops")
    if raw_nodes is None:
        raw_nodes = descriptor.get("instructions")

    if raw_nodes is None:
        if "op" in descriptor:
            return [descriptor]
        raise ValueError("descriptor must provide 'nodes'/'ops' or a top-level 'op'")

    if not _is_seq(raw_nodes):
        raise TypeError("descriptor nodes must be a list/tuple")

    out: list[dict[str, Any]] = []
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise TypeError(f"node entries must be dicts, got {type(raw).__name__}")
        out.append(raw)
    return out


def _extract_node_args(node: dict[str, Any]) -> list[Any]:
    if "args" in node:
        args = node["args"]
    elif "inputs" in node:
        args = node["inputs"]
    elif "lhs" in node and "rhs" in node:
        args = [node["lhs"], node["rhs"]]
    elif "a" in node and "b" in node:
        args = [node["a"], node["b"]]
    elif "x" in node:
        args = [node["x"]]
    elif "input" in node:
        args = [node["input"]]
    else:
        args = []

    if _is_seq(args):
        return list(args)
    return [args]


def _normalize_ref(raw: Any, input_names: tuple[str, ...], node_ids: tuple[str, ...]) -> str:
    if isinstance(raw, str):
        return raw

    if isinstance(raw, int):
        i = int(raw)
        if 0 <= i < len(input_names):
            return input_names[i]
        if 0 <= i < len(node_ids):
            return node_ids[i]
        raise ValueError(f"integer ref {raw} is out of range")

    if isinstance(raw, dict):
        if "input" in raw:
            return _normalize_ref(raw["input"], input_names, node_ids)
        if "node" in raw:
            return _normalize_ref(raw["node"], input_names, node_ids)
        if len(raw) == 1 and "id" in raw:
            return str(raw["id"])
        if len(raw) == 1 and "name" in raw:
            return str(raw["name"])

    raise TypeError(f"unsupported ref type {type(raw).__name__}: {raw!r}")


def _extract_output_ref(
    descriptor: dict[str, Any],
    input_names: tuple[str, ...],
    node_ids: tuple[str, ...],
) -> str:
    raw = descriptor.get("output")
    if raw is None:
        raw = descriptor.get("out")
    if raw is None:
        raw = descriptor.get("result")
    if raw is None:
        if node_ids:
            return node_ids[-1]
        raise ValueError("descriptor must define an output")

    if _is_seq(raw):
        if len(raw) != 1:
            raise ValueError("phase-1 fused elementwise only supports one output")
        raw = raw[0]

    return _normalize_ref(raw, input_names, node_ids)


def _toposort_nodes(
    *,
    nodes_by_id: dict[str, FusedNode],
    output_ref: str,
    input_names: tuple[str, ...],
) -> tuple[FusedNode, ...]:
    inputs = set(input_names)
    resolved: set[str] = set()
    visiting: set[str] = set()
    ordered: list[FusedNode] = []

    def visit(ref: str) -> None:
        if ref in inputs:
            return
        if ref in resolved:
            return
        if ref in visiting:
            raise ValueError(f"cycle detected at {ref!r}")
        node = nodes_by_id.get(ref)
        if node is None:
            raise ValueError(f"unknown ref {ref!r}")
        visiting.add(ref)
        for arg in node.args:
            visit(arg)
        visiting.remove(ref)
        resolved.add(ref)
        ordered.append(node)

    visit(output_ref)
    return tuple(ordered)


def normalize_descriptor(descriptor: FusedGraph | dict[str, Any]) -> FusedGraph:
    """Normalize a user/fusion-rule descriptor into a deterministic graph."""

    if isinstance(descriptor, FusedGraph):
        return descriptor
    if not isinstance(descriptor, dict):
        raise TypeError(f"descriptor must be dict-like, got {type(descriptor).__name__}")

    input_names = _extract_input_names(descriptor)
    raw_nodes = _extract_raw_nodes(descriptor)

    node_ids: list[str] = []
    for i, raw in enumerate(raw_nodes):
        node_id = raw.get("id") or raw.get("name") or raw.get("out")
        if node_id is None:
            node_id = f"n{i}"
        node_ids.append(str(node_id))

    if len(set(node_ids)) != len(node_ids):
        raise ValueError(f"node ids must be unique, got {node_ids}")

    for node_id in node_ids:
        if node_id in set(input_names):
            raise ValueError(f"node id {node_id!r} collides with an input name")

    nodes: list[FusedNode] = []
    node_ids_t = tuple(node_ids)
    for i, raw in enumerate(raw_nodes):
        node_id = node_ids[i]
        op = _canonical_op_name(raw.get("op") or raw.get("kind") or raw.get("type"))
        raw_args = _extract_node_args(raw)
        expected_arity, _ = _SUPPORTED_OPS[op]
        if len(raw_args) != expected_arity:
            raise ValueError(f"node {node_id!r} op {op!r} expects {expected_arity} args")
        args = tuple(_normalize_ref(arg, input_names, node_ids_t) for arg in raw_args)
        nodes.append(FusedNode(id=node_id, op=op, args=args))

    output_ref = _extract_output_ref(descriptor, input_names, node_ids_t)
    nodes_by_id = {node.id: node for node in nodes}
    topo_nodes = _toposort_nodes(
        nodes_by_id=nodes_by_id,
        output_ref=output_ref,
        input_names=input_names,
    )

    if output_ref not in set(input_names) and output_ref not in nodes_by_id:
        raise ValueError(f"output ref {output_ref!r} must reference an input or node")

    name = str(descriptor.get("name") or descriptor.get("kernel_name") or "")
    return FusedGraph(
        input_names=input_names,
        nodes=topo_nodes,
        output=output_ref,
        name=name,
    )


def _sanitize_identifier(name: str, fallback: str) -> str:
    cleaned = "".join(ch if (ch.isascii() and (ch.isalnum() or ch == "_")) else "_" for ch in name)
    if not cleaned:
        cleaned = fallback
    if not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"_{cleaned}"
    return cleaned


def _sanitize_input_names(input_names: tuple[str, ...], output_name: str) -> tuple[str, ...]:
    out: list[str] = []
    used: set[str] = {output_name}
    for i, name in enumerate(input_names):
        base = _sanitize_identifier(name, f"in{i}")
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        out.append(candidate)
    return tuple(out)


def _emit_op(op: str, args: tuple[str, ...]) -> str:
    _, emitter = _SUPPORTED_OPS[op]
    return str(emitter(args))


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def fused_op_signature(descriptor: FusedGraph | dict[str, Any]) -> str:
    """Return a stable signature for a fused graph's operation structure."""

    graph = normalize_descriptor(descriptor)
    return _hash_payload(graph.to_payload())


def generate_fused_elementwise_source(
    descriptor: FusedGraph | dict[str, Any],
    *,
    n_elements: int,
    kernel_name: str | None = None,
    output_name: str = "out",
) -> GeneratedFusedSource:
    """Generate MSL body for one fused elementwise kernel launch shape."""

    graph = normalize_descriptor(descriptor)
    signature = fused_op_signature(graph)

    safe_output_name = _sanitize_identifier(output_name, "out")
    safe_inputs = _sanitize_input_names(graph.input_names, safe_output_name)

    value_by_ref: dict[str, str] = {}
    lines: list[str] = [
        f"constexpr uint N_ELEMS = {int(n_elements)}u;",
        "uint idx = thread_position_in_grid.x;",
        "if (idx >= N_ELEMS) {",
        "    return;",
        "}",
    ]

    for i, (raw_name, safe_name) in enumerate(zip(graph.input_names, safe_inputs, strict=True)):
        var = f"in_{i}"
        lines.append(f"float {var} = (float){safe_name}[idx];")
        value_by_ref[raw_name] = var

    for i, node in enumerate(graph.nodes):
        args = tuple(value_by_ref[arg] for arg in node.args)
        expr = _emit_op(node.op, args)
        var = f"tmp_{i}"
        lines.append(f"float {var} = {expr};")
        value_by_ref[node.id] = var

    if graph.output not in value_by_ref:
        raise ValueError(f"output ref {graph.output!r} was not resolved")

    lines.append(f"{safe_output_name}[idx] = (O)({value_by_ref[graph.output]});")
    source = "\n".join(lines)

    if kernel_name is None:
        base_name = graph.name or f"kk_fused_elem_{signature[:12]}"
        kernel_name = f"{_sanitize_identifier(base_name, 'kk_fused_elem')}_{int(n_elements)}"
    else:
        kernel_name = _sanitize_identifier(kernel_name, "kk_fused_elem")

    return GeneratedFusedSource(
        kernel_name=kernel_name,
        input_names=safe_inputs,
        output_name=safe_output_name,
        source=source,
        op_signature=signature,
        graph=graph,
    )


def generate_broadcast_mul_reduce_two_pass_source(
    *,
    rows: int,
    reduce_extent: int,
    partial_cols: int,
    reduce_op: str,
    rhs_mode: str,
    no_fma: bool,
    kernel_base: str | None = None,
) -> GeneratedReductionTwoPassSource:
    """Generate two-pass MSL for ``mul + reduce`` with float32 accumulation."""

    if reduce_op not in {"sum", "mean", "max"}:
        raise ValueError(f"Unsupported reduction op {reduce_op!r}.")
    if rhs_mode not in _RHS_MODE_TO_INT:
        raise ValueError(f"Unsupported rhs_mode {rhs_mode!r}.")
    if rows <= 0 or reduce_extent <= 0 or partial_cols <= 0:
        raise ValueError("rows/reduce_extent/partial_cols must be > 0.")

    payload = {
        "kind": "broadcast_mul_reduce",
        "rows": int(rows),
        "reduce_extent": int(reduce_extent),
        "partial_cols": int(partial_cols),
        "reduce_op": reduce_op,
        "rhs_mode": rhs_mode,
        "no_fma": bool(no_fma),
    }
    signature = _hash_payload(payload)
    base = _sanitize_identifier(
        kernel_base or f"kk_fused_reduce_{signature[:10]}",
        "kk_fused_reduce",
    )
    pass1_name = f"{base}_pass1"
    pass2_name = f"{base}_pass2"
    rhs_mode_int = _RHS_MODE_TO_INT[rhs_mode]

    if reduce_op == "max":
        init_value = "-INFINITY"
        combine_expr = "metal::max(scratch[tid], scratch[tid + stride])"
        row_reduce_combine = "acc = metal::max(acc, partials[base + g]);"
    else:
        init_value = "0.0f"
        combine_expr = "scratch[tid] + scratch[tid + stride]"
        row_reduce_combine = "acc += partials[base + g];"

    if no_fma:
        mul_acc_stmt = "float prod = a * b; acc += prod;"
        pragma_stmt = "#pragma clang fp contract(off)\n"
    else:
        mul_acc_stmt = "acc = metal::fma(a, b, acc);"
        pragma_stmt = ""

    if reduce_op == "max":
        mul_acc_stmt = "float prod = a * b; acc = metal::max(acc, prod);"

    mean_finalize = "float outv = acc / (float)REDUCE_EXTENT;" if reduce_op == "mean" else "float outv = acc;"

    source_pass1 = f"""
        {pragma_stmt}constexpr uint ROWS = {int(rows)}u;
        constexpr uint REDUCE_EXTENT = {int(reduce_extent)}u;
        constexpr uint PARTIAL_COLS = {int(partial_cols)}u;
        constexpr uint RHS_MODE = {rhs_mode_int}u;

        uint tid = thread_position_in_threadgroup.x;
        uint packed = thread_position_in_grid.y;
        uint row = packed / PARTIAL_COLS;
        uint part = packed % PARTIAL_COLS;
        if (row >= ROWS || part >= PARTIAL_COLS) {{
            return;
        }}

        uint tgx = threads_per_threadgroup.x;
        uint lhs_base = row * REDUCE_EXTENT;
        uint start = part * tgx + tid;

        float acc = {init_value};
        uint stride = PARTIAL_COLS * tgx;
        for (uint j = start; j < REDUCE_EXTENT; j += stride) {{
            uint lhs_idx = lhs_base + j;
            float a = (float)lhs[lhs_idx];
            float b = 0.0f;
            if (RHS_MODE == 0u) {{
                b = (float)rhs[lhs_idx];
            }} else if (RHS_MODE == 1u) {{
                b = (float)rhs[row];
            }} else {{
                b = (float)rhs[0];
            }}
            {mul_acc_stmt}
        }}

        threadgroup float scratch[512];
        scratch[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tgx >> 1; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] = {combine_expr};
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0u) {{
            partials[row * PARTIAL_COLS + part] = scratch[0];
        }}
    """

    source_pass2 = f"""
        constexpr uint ROWS = {int(rows)}u;
        constexpr uint PARTIAL_COLS = {int(partial_cols)}u;
        constexpr uint REDUCE_EXTENT = {int(reduce_extent)}u;

        uint tid = thread_position_in_threadgroup.x;
        uint row = thread_position_in_grid.y;
        if (row >= ROWS) {{
            return;
        }}

        uint tgx = threads_per_threadgroup.x;
        uint base = row * PARTIAL_COLS;
        float acc = {init_value};
        for (uint g = tid; g < PARTIAL_COLS; g += tgx) {{
            {row_reduce_combine}
        }}

        threadgroup float scratch[512];
        scratch[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = tgx >> 1; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                scratch[tid] = {combine_expr};
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        if (tid == 0u) {{
            {mean_finalize}
            out[row] = (O)(outv);
        }}
    """

    return GeneratedReductionTwoPassSource(
        pass1_kernel_name=pass1_name,
        pass2_kernel_name=pass2_name,
        source_pass1=source_pass1,
        source_pass2=source_pass2,
        op_signature=signature,
        reduce_op=reduce_op,
        no_fma=bool(no_fma),
        rows=int(rows),
        reduce_extent=int(reduce_extent),
        partial_cols=int(partial_cols),
        rhs_mode=rhs_mode,
    )


def generate_quantized_shared_input_source(
    *,
    n: int,
    k: int,
    bits: int,
    group_size: int,
    no_fma: bool = False,
    kernel_name: str | None = None,
) -> GeneratedQuantizedSharedInputSource:
    """Generate decode-oriented QMM pair + SwiGLU source (M=1)."""

    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if group_size <= 0:
        raise ValueError("group_size must be > 0")
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be > 0")
    if k % group_size != 0:
        raise ValueError("group_size must divide k")

    elements_per_word = 32 // bits
    if k % elements_per_word != 0:
        raise ValueError("k must be divisible by packed elements per word")

    k_packed = k // elements_per_word
    groups_per_row = k // group_size
    mask = (1 << bits) - 1
    payload = {
        "kind": "quantized_shared_input_swiglu",
        "n": int(n),
        "k": int(k),
        "bits": int(bits),
        "group_size": int(group_size),
        "no_fma": bool(no_fma),
    }
    signature = _hash_payload(payload)

    pragma_stmt = "#pragma clang fp contract(off)\n" if no_fma else ""
    if no_fma:
        gate_acc = "float g_prod = gate_val * xv; acc_gate += g_prod;"
        up_acc = "float u_prod = up_val * xv; acc_up += u_prod;"
    else:
        gate_acc = "acc_gate = metal::fma(gate_val, xv, acc_gate);"
        up_acc = "acc_up = metal::fma(up_val, xv, acc_up);"

    source = f"""
        {pragma_stmt}constexpr uint N = {int(n)}u;
        constexpr uint K = {int(k)}u;
        constexpr uint K_PACKED = {int(k_packed)}u;
        constexpr uint GROUP = {int(group_size)}u;
        constexpr uint GROUPS = {int(groups_per_row)}u;
        constexpr uint ELEMENTS = {int(elements_per_word)}u;
        constexpr uint MASK = {int(mask)}u;

        uint col = thread_position_in_grid.x;
        if (col >= N) {{
            return;
        }}

        uint w_row = col * K_PACKED;
        uint s_row = col * GROUPS;
        float acc_gate = 0.0f;
        float acc_up = 0.0f;

        for (uint i = 0; i < K; ++i) {{
            uint pack_idx = i / ELEMENTS;
            uint shift = (i % ELEMENTS) * {int(bits)};
            uint q_gate = (gate_w[w_row + pack_idx] >> shift) & MASK;
            uint q_up = (up_w[w_row + pack_idx] >> shift) & MASK;
            uint g = i / GROUP;

            float gate_val = (float)q_gate * (float)gate_scales[s_row + g]
                + (float)gate_biases[s_row + g];
            float up_val = (float)q_up * (float)up_scales[s_row + g]
                + (float)up_biases[s_row + g];
            float xv = (float)x[i];

            {gate_acc}
            {up_acc}
        }}

        float outv = (acc_gate * kk_sigmoid(acc_gate)) * acc_up;
        out[col] = (O)outv;
    """

    safe_name = _sanitize_identifier(
        kernel_name or f"kk_qshared_swiglu_{signature[:12]}",
        "kk_qshared_swiglu",
    )
    return GeneratedQuantizedSharedInputSource(
        kernel_name=safe_name,
        source=source,
        op_signature=signature,
        bits=int(bits),
        group_size=int(group_size),
        n=int(n),
        k=int(k),
    )


def generate_msl(
    descriptor: FusedGraph | dict[str, Any],
    *,
    n_elements: int,
    kernel_name: str | None = None,
    output_name: str = "out",
) -> GeneratedFusedSource:
    """Alias for generate_fused_elementwise_source()."""

    return generate_fused_elementwise_source(
        descriptor,
        n_elements=n_elements,
        kernel_name=kernel_name,
        output_name=output_name,
    )


__all__ = [
    "FusedNode",
    "FusedGraph",
    "GeneratedFusedSource",
    "GeneratedReductionTwoPassSource",
    "GeneratedQuantizedSharedInputSource",
    "normalize_descriptor",
    "fused_op_signature",
    "generate_fused_elementwise_source",
    "generate_broadcast_mul_reduce_two_pass_source",
    "generate_quantized_shared_input_source",
    "generate_msl",
]
