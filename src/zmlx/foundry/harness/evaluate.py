"""Generic evaluation orchestrator for foundry kernel candidates.

This is the KEY refactor from DataFoundry: instead of hard-coded
``if op == "rmsnorm" ... elif op == "swiglu"`` branches, the
orchestrator accepts a ``KernelOp`` instance that encapsulates all
op-specific logic (input generation, reference computation, output
shape inference, threadgroup memory estimation).

The flow is:
    1. Render the Metal template with knob values.
    2. Compile (via ``compile_kernel`` with caching).
    3. For each correctness seed: generate inputs, run kernel (or
       mock with reference), compute error metrics, gate on tolerances.
    4. If correctness passes: benchmark with adaptive repeats.
    5. Return a structured attempt record dict (ready for NDJSON).
"""
from __future__ import annotations

import datetime as _dt
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..ids import attempt_id, cache_key, shape_class
from ..ndjson import append_record
from ..taxonomy import KernelCandidate, KernelOp
from ..templates.render import render_template
from .backend import MLXBackend, MockBackend
from .bench import bench as bench_fn
from .cache import CompileCache
from .compile import compile_kernel
from .correctness import check_pass, compute_metrics_mlx, compute_metrics_numpy

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _env(backend: Any) -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "mlx_version": backend.mlx_version() if hasattr(backend, "mlx_version") else None,
        "python": sys.version.split()[0],
        "device": backend.device_info() if hasattr(backend, "device_info") else {},
    }


def _trim(s: Any, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else (s[:n] + "...")


def _classify_error(e: Exception) -> tuple[str, str, str]:
    """Classify an exception into (error_type, summary, log)."""
    msg = str(e)
    summary = msg.split("\n", 1)[0]
    etype = "api_error"
    lowered = msg.lower()
    if any(
        tok in lowered
        for tok in (
            "unable to build metal library",
            "program_source",
            "error:",
            "metal::device",
        )
    ):
        etype = "compile_error"
    return etype, _trim(summary, 200), _trim(msg, 4000)


def _metal_out_type(dtype: str) -> str:
    """Map a dtype string to the Metal type name for output casts."""
    if dtype == "float16":
        return "float16_t"
    if dtype == "bfloat16":
        return "bfloat16_t"
    if dtype == "float32":
        return "float"
    raise ValueError(f"unsupported dtype for Metal output: {dtype}")


def _mlx_dtype(mx: Any, dtype: str) -> Any:
    """Map a dtype string to an ``mx.*`` dtype object."""
    if dtype == "float16":
        return mx.float16
    if dtype == "float32":
        return mx.float32
    if dtype == "bfloat16":
        return mx.bfloat16
    raise ValueError(f"unsupported mlx dtype: {dtype}")


def _np_strides_elems(x: np.ndarray) -> list[int]:
    """Convert numpy byte-strides to element-strides."""
    item = x.dtype.itemsize
    return [int(s // item) for s in x.strides]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_attempt(
    *,
    session_dir: Path,
    backend_name: str,
    candidate: KernelCandidate,
    op: KernelOp,
    cache: CompileCache,
    correctness_tests: int = 3,
    correctness_seed: int = 0,
    warmup: int = 10,
    repeats: int = 50,
    bench_timeout_s: float = 10.0,
    compile_timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Compile, correctness-check, and benchmark one kernel attempt.

    This is the main orchestrator.  It is **generic** over ops: all
    op-specific logic is delegated to the ``op`` parameter (a ``KernelOp``
    instance).

    Parameters:
        session_dir: Directory for writing kernel source files and logs.
        backend_name: ``"mlx"`` for real GPU or ``"mock"`` for testing.
        candidate: The ``KernelCandidate`` describing template + knobs.
        op: A ``KernelOp`` instance for input generation, reference
            computation, and output shape inference.
        cache: ``CompileCache`` instance (shared across attempts).
        correctness_tests: Number of random seeds to test.
        correctness_seed: Base seed for reproducibility.
        warmup: Benchmark warmup iterations.
        repeats: Benchmark timed iterations.
        bench_timeout_s: Wall-clock timeout for benchmarking.
        compile_timeout_s: Wall-clock timeout for compilation (advisory).

    Returns:
        A dict suitable for NDJSON serialisation with keys:
        ``id``, ``ts``, ``op``, ``dtype``, ``shape``, ``layout``,
        ``spec``, ``kernel``, ``build``, ``correctness``, ``bench``, ``env``.
    """
    backend = MLXBackend() if backend_name == "mlx" else MockBackend()
    if backend_name == "mlx" and not backend.is_available():
        raise RuntimeError(
            "MLX backend requested but MLX is not available. "
            "Install mlx and run on Apple Silicon."
        )

    op_name = candidate.op
    dtype = candidate.dtype
    shape = dict(candidate.shape)
    layout = dict(candidate.layout)

    # -- Op-specific metadata via the KernelOp protocol ----------------
    # Support both taxonomy.KernelOp Protocol and ops.base.KernelOp ABC
    has_bridge = hasattr(op, "generate_inputs_numpy_bridge")

    math = op.canonical_math()
    constraints = op.correctness_constraints(dtype)
    bytes_est, flops_est = op.bytes_and_flops(shape, dtype)
    input_names = list(op.input_names)
    output_names = list(op.output_names)

    # -- Render template -----------------------------------------------
    template_path = (
        Path(__file__).resolve().parent.parent
        / "templates"
        / op_name
        / f"{candidate.template_id}.metal"
    )
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found: {template_path}"
        )

    tg_size = int(candidate.knobs.get("tg_size", 256))
    values = {
        "TG_SIZE": tg_size,
        "VEC": int(candidate.knobs.get("vec", 1)),
        "UNROLL": int(candidate.knobs.get("unroll", 1)),
        "FAST_MATH": 1 if candidate.knobs.get("fast_math", False) else 0,
        "INJECT_INCORRECT": 1 if candidate.knobs.get("inject_incorrect", False) else 0,
        "OUT_TYPE": _metal_out_type(dtype),
        "COMPILE_ERROR_SNIPPET": (
            "THIS_WILL_NOT_COMPILE;"
            if candidate.knobs.get("inject_compile_error", False)
            else ""
        ),
    }
    rendered = render_template(template_path, values)

    # -- Stable IDs ----------------------------------------------------
    sc = shape_class(op_name, dtype, shape, layout)
    rid = attempt_id(
        op=op_name,
        dtype=dtype,
        shape_class=sc,
        template_id=candidate.template_id,
        knobs=candidate.knobs,
        normalized_source=rendered.normalized_full_text,
    )
    ck = cache_key(
        op=op_name,
        dtype=dtype,
        template_id=candidate.template_id,
        knobs=candidate.knobs,
        normalized_source=rendered.normalized_full_text,
    )

    # -- Write kernel source to disk -----------------------------------
    kernels_dir = session_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)
    kernel_file = kernels_dir / f"{rid}.metal"
    if not kernel_file.exists():
        kernel_file.write_text(rendered.full_text, encoding="utf-8")

    # -- Launch config (one threadgroup per row) -----------------------
    rows = int(shape.get("batch", 1)) * int(shape.get("seq", 1))
    grid = (tg_size, rows, 1)
    threadgroup = (tg_size, 1, 1)
    tg_mem = op.tg_mem_bytes(candidate.template_id, tg_size)

    # -- Compile -------------------------------------------------------
    ensure_row_contig = bool(layout.get("contiguous", True))
    kernel_obj, build = compile_kernel(
        backend=backend,
        cache=cache,
        cache_key=ck,
        name=f"{op_name}_{candidate.template_id}",
        input_names=input_names,
        output_names=output_names,
        source=rendered.source,
        header=rendered.header,
        ensure_row_contiguous=ensure_row_contig,
        atomic_outputs=False,
    )

    # -- Base record ---------------------------------------------------
    record: dict[str, Any] = {
        "id": rid,
        "ts": _iso_now(),
        "op": op_name,
        "dtype": dtype,
        "shape": shape,
        "layout": {
            "contiguous": bool(layout.get("contiguous", True)),
            "strides": layout.get("strides", []),
        },
        "spec": {
            "math": math,
            "constraints": constraints,
            "reference_impl": f"{op_name}_reference_numpy_v1",
        },
        "kernel": {
            "template_id": candidate.template_id,
            "knobs": candidate.knobs,
            "source_metal": str(Path("kernels") / f"{rid}.metal"),
            "launch": {
                "grid": list(grid),
                "tg": list(threadgroup),
                "tg_mem_bytes": tg_mem,
            },
            "compile_flags": (
                ["fast_math"] if candidate.knobs.get("fast_math", False) else []
            ),
            "cache_key": ck,
        },
        "build": {
            "ok": bool(build.ok),
            "ms": float(build.ms),
            "error_type": build.error_type,
            "error_summary": build.error_summary,
            "error_log": build.error_log,
        },
        "correctness": {
            "ok": False,
            "max_abs_err": None,
            "max_rel_err": None,
            "ulp": None,
            "n_tests": int(correctness_tests),
            "seed": int(correctness_seed),
            "fail_reason": "build_failed" if not build.ok else None,
        },
        "bench": {
            "ok": False,
            "warmup": int(warmup),
            "repeats": int(repeats),
            "latency_ms": {"p50": None, "p90": None, "min": None, "mean": None},
            "throughput_gbs": None,
            "throughput_gflops": None,
            "timeout": False,
        },
        "env": _env(backend),
    }

    if not build.ok:
        return record

    # ================================================================
    # CORRECTNESS
    # ================================================================
    max_abs_all = 0.0
    max_rel_all = 0.0
    ulp_all: float | None = None
    fail_reason: str | None = None

    use_mlx = backend_name == "mlx"
    mx = None
    if use_mlx:
        import mlx.core as mx  # type: ignore[no-redef]

    def _to_mlx_input(np_arr: np.ndarray) -> Any:
        """Preserve integer tensors (e.g. index buffers), cast float tensors to candidate dtype."""
        assert mx is not None
        if np.issubdtype(np_arr.dtype, np.integer):
            return mx.array(np_arr.astype(np.int32, copy=False))
        if np_arr.ndim == 0 or np_arr.size == 1:
            return backend.array(np_arr, "float32")
        return backend.array(np_arr, dtype)

    for i in range(correctness_tests):
        seed_i = correctness_seed + i
        rng = np.random.default_rng(seed_i)

        # Generic input generation -- supports both protocol variants.
        if has_bridge:
            in_np = op.generate_inputs_numpy_bridge(rng, shape, dtype, layout)
        else:
            in_np = op.generate_inputs_numpy(rng, shape, dtype, layout)

        # Record strides from the first test case.
        if i == 0:
            # Use the first array that has meaningful strides.
            for arr_name in input_names:
                if arr_name in in_np and hasattr(in_np[arr_name], "strides"):
                    record["layout"]["strides"] = _np_strides_elems(in_np[arr_name])
                    break

        if use_mlx:
            # Convert numpy inputs to MLX arrays.
            mlx_inputs: list[Any] = []
            for arr_name in input_names:
                np_arr = in_np[arr_name]
                mlx_inputs.append(_to_mlx_input(np_arr))

            try:
                outs = kernel_obj(
                    inputs=mlx_inputs,
                    template=[],
                    grid=grid,
                    threadgroup=threadgroup,
                    output_shapes=op.output_shapes(in_np, shape),
                    output_dtypes=[_mlx_dtype(mx, dtype) for _ in output_names],
                )
                y = outs[0]

                # Build mlx-typed input dict for the reference.
                mlx_input_dict: dict[str, Any] = {}
                for idx, arr_name in enumerate(input_names):
                    mlx_input_dict[arr_name] = mlx_inputs[idx]

                try:
                    y_ref = op.reference_mlx(mlx_input_dict, dtype)
                    backend.eval(y)
                    backend.eval(y_ref)
                    backend.synchronize()
                    max_abs, max_rel, ulp = compute_metrics_mlx(mx, y, y_ref, dtype)
                except NotImplementedError:
                    # Most ops only provide numpy references; compare in numpy.
                    backend.eval(y)
                    backend.synchronize()
                    y_np = np.asarray(backend.to_numpy(y.astype(mx.float32)))
                    if has_bridge:
                        y_ref_np = op.reference_numpy_bridge(in_np, dtype)
                    else:
                        y_ref_np = op.reference_numpy(in_np, dtype)
                    y_ref_np = np.asarray(y_ref_np)
                    max_abs, max_rel, ulp = compute_metrics_numpy(y_np, y_ref_np, dtype)
            except Exception as exc:
                et, summ, log = _classify_error(exc)
                record["build"]["ok"] = False
                record["build"]["error_type"] = et
                record["build"]["error_summary"] = summ
                record["build"]["error_log"] = log
                record["correctness"]["ok"] = False
                record["correctness"]["fail_reason"] = "kernel_call_error"
                return record
        else:
            # Mock path: use numpy reference, optionally corrupt output.
            if has_bridge:
                y_ref = op.reference_numpy_bridge(in_np, dtype)
            else:
                y_ref = op.reference_numpy(in_np, dtype)
            y = y_ref.copy()
            if candidate.knobs.get("inject_incorrect", False):
                # Inject a small offset so correctness fails.
                y = y + 0.1
            max_abs, max_rel, ulp = compute_metrics_numpy(y, y_ref, dtype)

        max_abs_all = max(max_abs_all, float(max_abs))
        max_rel_all = max(max_rel_all, float(max_rel))
        if ulp is not None:
            ulp_all = float(ulp) if ulp_all is None else max(ulp_all, float(ulp))

        ok, reason = check_pass(dtype, max_abs, max_rel)
        if not ok:
            fail_reason = reason
            break

    corr_ok = fail_reason is None
    record["correctness"] = {
        "ok": bool(corr_ok),
        "max_abs_err": float(max_abs_all) if corr_ok or max_abs_all > 0 else None,
        "max_rel_err": float(max_rel_all) if corr_ok or max_rel_all > 0 else None,
        "ulp": ulp_all,
        "n_tests": int(correctness_tests),
        "seed": int(correctness_seed),
        "fail_reason": fail_reason,
    }

    if not corr_ok:
        return record

    # ================================================================
    # BENCHMARK
    # ================================================================
    rngb = np.random.default_rng(correctness_seed + 10_000)
    if has_bridge:
        in_np_bench = op.generate_inputs_numpy_bridge(rngb, shape, dtype, layout)
    else:
        in_np_bench = op.generate_inputs_numpy(rngb, shape, dtype, layout)

    if use_mlx:
        mlx_bench_inputs: list[Any] = []
        for arr_name in input_names:
            np_arr = in_np_bench[arr_name]
            mlx_bench_inputs.append(_to_mlx_input(np_arr))

        bench_out_shapes = op.output_shapes(in_np_bench, shape)
        bench_out_dtypes = [_mlx_dtype(mx, dtype) for _ in output_names]

        def run_once() -> None:
            outs = kernel_obj(
                inputs=mlx_bench_inputs,
                template=[],
                grid=grid,
                threadgroup=threadgroup,
                output_shapes=bench_out_shapes,
                output_dtypes=bench_out_dtypes,
            )
            backend.eval(outs[0])

        def sync() -> None:
            backend.synchronize()
    else:
        def run_once() -> None:
            if has_bridge:
                _ = op.reference_numpy_bridge(in_np_bench, dtype)
            else:
                _ = op.reference_numpy(in_np_bench, dtype)

        def sync() -> None:
            return

    try:
        ok_bench, lat, timed_out = bench_fn(
            run_once=run_once,
            sync=sync,
            warmup=warmup,
            repeats=repeats,
            timeout_s=bench_timeout_s,
            adaptive=True,
        )
    except Exception as exc:
        et, summ, log = _classify_error(exc)
        record["bench"] = {
            "ok": False,
            "warmup": int(warmup),
            "repeats": int(repeats),
            "latency_ms": {"p50": None, "p90": None, "min": None, "mean": None},
            "throughput_gbs": None,
            "throughput_gflops": None,
            "timeout": False,
            "error_type": et,
            "error_summary": summ,
            "error_log": log,
        }
        return record

    throughput_gbs: float | None = None
    throughput_gflops: float | None = None
    if ok_bench and lat.get("p50") is not None and float(lat["p50"]) > 0:
        sec = float(lat["p50"]) / 1000.0
        throughput_gbs = (bytes_est / sec) / 1e9
        throughput_gflops = (flops_est / sec) / 1e9

    record["bench"] = {
        "ok": bool(ok_bench),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "latency_ms": {
            k: (float(v) if v is not None else None) for k, v in lat.items()
        },
        "throughput_gbs": (
            float(throughput_gbs) if throughput_gbs is not None else None
        ),
        "throughput_gflops": (
            float(throughput_gflops) if throughput_gflops is not None else None
        ),
        "timeout": bool(timed_out),
    }
    return record


# ---------------------------------------------------------------------------
# Record writer (convenience wrapper around ndjson.append_record)
# ---------------------------------------------------------------------------


def write_record(
    *,
    session_dir: Path,
    record: dict[str, Any],
    worker_id: int | None = None,
) -> None:
    """Append an attempt record to the session's NDJSON log."""
    if worker_id is None:
        path = session_dir / "attempts.ndjson"
    else:
        path = session_dir / f"attempts.worker{worker_id}.ndjson"
    append_record(path, record)
