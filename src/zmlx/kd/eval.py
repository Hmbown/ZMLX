"""Correctness and performance evaluation for kernel candidates."""

from __future__ import annotations

import functools
import json
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .ops import mx_dtype
from .types import KernelCandidate


@dataclass
class CorrectnessSummary:
    ok: bool
    max_abs_err: float
    max_rel_err: float


@dataclass
class TimingSummary:
    median: float
    p10: float
    p90: float
    mean: float
    stdev: float
    count: int

    @property
    def cv(self) -> float:
        if self.mean <= 0:
            return 0.0
        return float(self.stdev / self.mean)


def tolerances(dtype_name: str) -> tuple[float, float]:
    mapping = {
        "float16": (3e-2, 3e-2),
        "bfloat16": (5e-2, 5e-2),
        "float32": (1e-4, 1e-4),
    }
    return mapping.get(dtype_name, (1e-4, 1e-4))


def assess_correctness(
    candidate_outputs: list[np.ndarray],
    reference_outputs: list[np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> CorrectnessSummary:
    """Compare outputs and return max abs/rel error."""
    if len(candidate_outputs) != len(reference_outputs):
        return CorrectnessSummary(ok=False, max_abs_err=float("inf"), max_rel_err=float("inf"))

    max_abs = 0.0
    max_rel = 0.0

    for cand, ref in zip(candidate_outputs, reference_outputs, strict=True):
        if cand.shape != ref.shape:
            return CorrectnessSummary(ok=False, max_abs_err=float("inf"), max_rel_err=float("inf"))
        with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
            cand64 = cand.astype(np.float64)
            ref64 = ref.astype(np.float64)
            diff = np.abs(cand64 - ref64)
            abs_err = float(np.max(diff)) if diff.size else 0.0
            denom = np.maximum(np.abs(ref64), 1e-12)
            rel_err = float(np.max(diff / denom)) if diff.size else 0.0
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)

    ok = max_abs <= atol or max_rel <= rtol
    return CorrectnessSummary(ok=ok, max_abs_err=max_abs, max_rel_err=max_rel)


def _sync(mx_mod: Any) -> None:
    sync = getattr(mx_mod, "synchronize", None)
    if callable(sync):
        sync()


def _materialize(mx_mod: Any, outputs: Any) -> list[Any]:
    if isinstance(outputs, (list, tuple)):
        mx_mod.eval(*outputs)
        _sync(mx_mod)
        return list(outputs)
    mx_mod.eval(outputs)
    _sync(mx_mod)
    return [outputs]


def _time_callable(mx_mod: Any, fn: Any, *, warmup: int, iters: int) -> list[float]:
    for _ in range(max(0, warmup)):
        _materialize(mx_mod, fn())

    timings_us: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter_ns()
        _materialize(mx_mod, fn())
        timings_us.append((time.perf_counter_ns() - t0) / 1e3)
    return timings_us


def _compute_quantiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("inf"), float("inf"), float("inf")
    ordered = sorted(values)
    median = statistics.median(ordered)
    p10_idx = int(0.1 * (len(ordered) - 1))
    p90_idx = int(0.9 * (len(ordered) - 1))
    return float(median), float(ordered[p10_idx]), float(ordered[p90_idx])


def _summarize_timings(values: list[float]) -> TimingSummary:
    if not values:
        return TimingSummary(
            median=float("inf"),
            p10=float("inf"),
            p90=float("inf"),
            mean=float("inf"),
            stdev=0.0,
            count=0,
        )
    median, p10, p90 = _compute_quantiles(values)
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    return TimingSummary(
        median=float(median),
        p10=float(p10),
        p90=float(p90),
        mean=float(mean),
        stdev=float(stdev),
        count=len(values),
    )


def _interleaved_timings(
    mx_mod: Any,
    *,
    ref_fn: Any,
    cand_fn: Any,
    warmup: int,
    iters: int,
    repeats: int,
    seed: int,
) -> tuple[list[float], list[float], list[float]]:
    rng = random.Random(seed)
    warmup = max(0, warmup)
    iters = max(1, iters)
    repeats = max(1, repeats)

    for _ in range(warmup):
        _materialize(mx_mod, ref_fn())
        _materialize(mx_mod, cand_fn())

    ref_times: list[float] = []
    cand_times: list[float] = []
    for _ in range(repeats):
        order = ["ref", "cand"] * iters
        rng.shuffle(order)
        for tag in order:
            t0 = time.perf_counter_ns()
            if tag == "ref":
                _materialize(mx_mod, ref_fn())
                ref_times.append((time.perf_counter_ns() - t0) / 1e3)
            else:
                _materialize(mx_mod, cand_fn())
                cand_times.append((time.perf_counter_ns() - t0) / 1e3)

    speedups: list[float] = []
    pair_count = min(len(ref_times), len(cand_times))
    for i in range(pair_count):
        if cand_times[i] > 0:
            speedups.append(ref_times[i] / cand_times[i])
    return ref_times, cand_times, speedups


def _launch_candidate(
    *,
    candidate_kernel: Any,
    candidate: KernelCandidate,
    op_module: Any,
    mx_mod: Any,
    inputs: list[Any],
    shape: dict[str, Any],
    dtype_name: str,
) -> list[Any]:
    grid, threadgroup = op_module.compute_launch(candidate.launch_params, shape, inputs)
    outputs = candidate_kernel(
        *inputs,
        template=[("T", mx_dtype(mx_mod, dtype_name))],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=op_module.output_shapes(shape),
        output_dtypes=op_module.output_dtypes(mx_mod, dtype_name),
    )
    return list(outputs)


def _baseline_for_shape(
    *,
    mx_mod: Any,
    op_module: Any,
    inputs: list[Any],
    shape: dict[str, Any],
    dtype_name: str,
    warmup: int,
    iters: int,
) -> tuple[float, list[float]]:
    def _ref() -> Any:
        return op_module.reference(mx_mod, inputs, shape, dtype_name)

    timings = _time_callable(mx_mod, _ref, warmup=warmup, iters=iters)
    median, _, _ = _compute_quantiles(timings)
    return median, timings


def evaluate_candidate(
    *,
    candidate: KernelCandidate,
    op_module: Any,
    dtype_name: str,
    shape_suite: list[dict[str, Any]],
    seed: int,
    warmup: int,
    iters: int,
    repeats: int = 3,
    bench_mode: str = "interleaved",
    baseline_cache: dict[tuple[str, str], float],
) -> KernelCandidate:
    """Compile, validate, and benchmark one candidate across a shape suite."""
    import mlx.core as mx

    compile_t0 = time.perf_counter_ns()
    try:
        k = metal_kernel(
            name=candidate.func_name,
            input_names=[spec["name"] for spec in candidate.inputs_spec],
            output_names=[spec["name"] for spec in candidate.outputs_spec],
            source=candidate.metal_source,
            header=DEFAULT_HEADER,
            cache=False,
        )
    except Exception as exc:
        candidate.status = "failed"
        candidate.metrics = {
            "failure": "compile_error",
            "failure_reason": str(exc),
        }
        return candidate

    compile_time_ms = (time.perf_counter_ns() - compile_t0) / 1e6
    candidate.status = "compiled"

    atol, rtol = tolerances(dtype_name)

    per_shape: list[dict[str, Any]] = []
    all_candidate_timings: list[float] = []
    all_reference_timings: list[float] = []
    max_abs_err = 0.0
    max_rel_err = 0.0

    for shape_idx, shape in enumerate(shape_suite):
        case_seed = seed + shape_idx * 1009
        try:
            inputs = op_module.make_inputs(mx, shape, dtype_name, case_seed)
        except Exception as exc:
            candidate.status = "failed"
            candidate.metrics = {
                "failure": "input_error",
                "failure_reason": str(exc),
                "compile_time_ms": compile_time_ms,
            }
            return candidate

        try:
            cand_outputs = _launch_candidate(
                candidate_kernel=k,
                candidate=candidate,
                op_module=op_module,
                mx_mod=mx,
                inputs=inputs,
                shape=shape,
                dtype_name=dtype_name,
            )
            cand_outputs = _materialize(mx, cand_outputs)
        except Exception as exc:
            candidate.status = "failed"
            candidate.metrics = {
                "failure": "runtime_error",
                "failure_reason": str(exc),
                "compile_time_ms": compile_time_ms,
            }
            return candidate

        try:
            ref_outputs = op_module.reference(mx, inputs, shape, dtype_name)
            ref_outputs = _materialize(mx, ref_outputs)
        except Exception as exc:
            candidate.status = "failed"
            candidate.metrics = {
                "failure": "reference_error",
                "failure_reason": str(exc),
                "compile_time_ms": compile_time_ms,
            }
            return candidate

        cand_np = [np.array(x) for x in cand_outputs]
        ref_np = [np.array(x) for x in ref_outputs]
        corr = assess_correctness(cand_np, ref_np, atol=atol, rtol=rtol)
        max_abs_err = max(max_abs_err, corr.max_abs_err)
        max_rel_err = max(max_rel_err, corr.max_rel_err)

        if not corr.ok:
            candidate.status = "failed"
            candidate.metrics = {
                "failure": "correctness_error",
                "failure_reason": "outputs differ from reference",
                "correctness_max_abs_err": corr.max_abs_err,
                "correctness_max_rel_err": corr.max_rel_err,
                "compile_time_ms": compile_time_ms,
            }
            return candidate

        candidate.status = "correct"

        run_candidate = functools.partial(
            _launch_candidate,
            candidate_kernel=k,
            candidate=candidate,
            op_module=op_module,
            mx_mod=mx,
            inputs=inputs,
            shape=shape,
            dtype_name=dtype_name,
        )
        try:
            cand_timings = _time_callable(mx, run_candidate, warmup=warmup, iters=iters)
        except Exception as exc:
            candidate.status = "failed"
            candidate.metrics = {
                "failure": "benchmark_runtime_error",
                "failure_reason": str(exc),
                "compile_time_ms": compile_time_ms,
            }
            return candidate

        shape_sig = op_module.shape_signature(shape)
        cache_key = (candidate.op_name, json.dumps(shape_sig, sort_keys=True))
        if cache_key in baseline_cache:
            baseline_median = baseline_cache[cache_key]
            ref_timings = []
        else:
            try:
                baseline_median, ref_timings = _baseline_for_shape(
                    mx_mod=mx,
                    op_module=op_module,
                    inputs=inputs,
                    shape=shape,
                    dtype_name=dtype_name,
                    warmup=warmup,
                    iters=iters,
                )
            except Exception as exc:
                candidate.status = "failed"
                candidate.metrics = {
                    "failure": "reference_timing_error",
                    "failure_reason": str(exc),
                    "compile_time_ms": compile_time_ms,
                }
                return candidate
            baseline_cache[cache_key] = baseline_median

        cand_median, cand_p10, cand_p90 = _compute_quantiles(cand_timings)
        gbps = 0.0
        if cand_median > 0:
            gbps = (op_module.bytes_moved(shape, dtype_name) / (cand_median * 1e-6)) / 1e9

        speedup = (baseline_median / cand_median) if cand_median > 0 else 0.0
        per_shape.append(
            {
                "shape": shape_sig,
                "latency_us": cand_median,
                "p10_us": cand_p10,
                "p90_us": cand_p90,
                "baseline_us": baseline_median,
                "speedup_vs_ref": speedup,
                "gbps_est": gbps,
                "max_abs_err": corr.max_abs_err,
                "max_rel_err": corr.max_rel_err,
            }
        )

        all_candidate_timings.extend(cand_timings)
        if ref_timings:
            all_reference_timings.extend(ref_timings)
        else:
            all_reference_timings.append(baseline_median)

    cand_median, cand_p10, cand_p90 = _compute_quantiles(all_candidate_timings)
    ref_median, _, _ = _compute_quantiles(all_reference_timings)
    speedup_vs_ref = (ref_median / cand_median) if cand_median > 0 else 0.0
    gbps_est = 0.0
    if per_shape:
        gbps_est = float(sum(case["gbps_est"] for case in per_shape) / len(per_shape))

    candidate.status = "benchmarked"
    candidate.metrics = {
        "compile_time_ms": compile_time_ms,
        "latency_us": cand_median,
        "p10_us": cand_p10,
        "p90_us": cand_p90,
        "reference_latency_us": ref_median,
        "speedup_vs_ref": speedup_vs_ref,
        "gbps_est": gbps_est,
        "correctness_max_abs_err": max_abs_err,
        "correctness_max_rel_err": max_rel_err,
        "dtype": dtype_name,
        "per_shape": per_shape,
    }
    return candidate
