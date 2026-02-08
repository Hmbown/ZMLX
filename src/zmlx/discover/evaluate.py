"""Compile -> correctness -> benchmark -> reward pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from .candidates import EvalResult, KernelCandidate
from .reward import compute_reward
from .safety import (
    KernelTimeoutError,
    eval_timeout,
    validate_metal_source,
)


def _time_fn(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    mx: Any,
    warmup: int,
    iters: int,
) -> list[float]:
    """Time a function and return per-iteration latencies in microseconds."""
    sync = getattr(mx, "synchronize", None)

    def _sync() -> None:
        if callable(sync):
            sync()

    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if isinstance(result, (list, tuple)):
            mx.eval(*result)
        else:
            mx.eval(result)
    _sync()

    # Timed
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        result = fn(*args)
        if isinstance(result, (list, tuple)):
            mx.eval(*result)
        else:
            mx.eval(result)
        _sync()
        times.append((time.perf_counter_ns() - t0) / 1e3)  # ns -> us

    return times


def evaluate_candidate(
    candidate: KernelCandidate,
    reference_fn: Callable[..., Any],
    test_inputs: Sequence[Any],
    *,
    baseline_us: float = 0.0,
    warmup: int = 5,
    iters: int = 20,
    timeout_s: float = 10.0,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    output_shapes: list[tuple[int, ...]] | None = None,
    output_dtypes: list[Any] | None = None,
    grid: tuple[int, int, int] | None = None,
    threadgroup: tuple[int, int, int] | None = None,
    template: list[tuple[str, Any]] | None = None,
) -> EvalResult:
    """Evaluate a single kernel candidate.

    Steps:
        1. Static safety checks
        2. Compile via ``metal.kernel(cache=False)``
        3. Correctness check against ``reference_fn``
        4. Benchmark timing
        5. Compute reward

    Returns an :class:`EvalResult` with all fields populated.
    """
    result = EvalResult()
    spec = candidate.spec

    # Static checks
    warnings = validate_metal_source(spec.source)
    if any("infinite loop" in w.lower() for w in warnings):
        result.compile_error = "; ".join(warnings)
        return result

    # Step 1: Compile
    try:
        with eval_timeout(timeout_s):
            from ..metal import kernel as metal_kernel
            from ..msl import DEFAULT_HEADER

            header = spec.header or DEFAULT_HEADER
            kern = metal_kernel(
                name=spec.name,
                input_names=spec.input_names,
                output_names=spec.output_names,
                source=spec.source,
                header=header,
                cache=False,
            )
            result.compiled = True
    except KernelTimeoutError:
        result.compile_error = "Compilation timed out"
        return result
    except Exception as e:
        result.compile_error = str(e)
        return result

    # Step 2: Correctness
    try:
        from .._compat import import_mx

        mx = import_mx()

        tg = threadgroup or spec.threadgroup
        tpl: list[tuple[str, Any]] = (
            template or [tuple(p) for p in spec.template_params] or [("T", mx.float32)]  # type: ignore[misc]
        )

        kern_out = kern(
            *test_inputs,
            template=tpl,
            grid=grid,
            threadgroup=tg,
            output_shapes=output_shapes,  # type: ignore[arg-type]
            output_dtypes=output_dtypes,
        )

        ref_out = reference_fn(*test_inputs)

        # Normalize to lists
        if not isinstance(kern_out, (list, tuple)):
            kern_out = [kern_out]
        if not isinstance(ref_out, (list, tuple)):
            ref_out = [ref_out]

        import numpy as np

        for i, (ko, ro) in enumerate(zip(kern_out, ref_out, strict=False)):
            mx.eval(ko)
            mx.eval(ro)
            np.testing.assert_allclose(
                np.array(ko),
                np.array(ro),
                rtol=rtol,
                atol=atol,
                err_msg=f"Output {i} mismatch",
            )
        result.correct = True
    except KernelTimeoutError:
        result.correctness_error = "Correctness check timed out"
        return result
    except Exception as e:
        result.correctness_error = str(e)
        return result

    # Step 3: Benchmark
    try:
        def _run(*inputs: Any) -> Any:
            return kern(
                *inputs,
                template=tpl,
                grid=grid,
                threadgroup=tg,
                output_shapes=output_shapes,  # type: ignore[arg-type]
                output_dtypes=output_dtypes,
            )

        with eval_timeout(timeout_s * 2):
            timings = _time_fn(_run, tuple(test_inputs), mx, warmup=warmup, iters=iters)

        result.timings_us = timings
        timings_sorted = sorted(timings)
        result.median_us = timings_sorted[len(timings_sorted) // 2]
    except KernelTimeoutError:
        result.timings_us = []
        result.median_us = float("inf")
        return result
    except Exception:
        result.timings_us = []
        result.median_us = float("inf")
        return result

    # Step 4: Reward
    if baseline_us > 0:
        result.reward = compute_reward(timings, baseline_us)
        result.speedup = baseline_us / result.median_us if result.median_us > 0 else 0.0

    return result
