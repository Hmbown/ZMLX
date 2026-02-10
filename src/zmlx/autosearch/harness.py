"""Evaluation harness bridging config space to discover's evaluation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from ..discover.candidates import EvalResult, KernelCandidate, KernelSpec
from ..discover.evaluate import _time_fn, evaluate_candidate
from ..discover.targets import TARGETS


@dataclass
class ScoredConfig:
    """A config with its evaluation result."""

    config: dict[str, Any]
    source: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int]
    eval_result: EvalResult
    reward: float = 0.0
    speedup: float = 0.0


@dataclass
class Harness:
    """Bridges autosearch config space to discover's evaluation pipeline.

    Loads a target from the discover registry, builds test inputs and
    reference function, measures baseline timing, and evaluates configs.
    """

    target_name: str
    D: int
    K: int = 2
    template_fn: Callable[..., tuple[str, tuple[int, int, int], tuple[int, int, int]]] | None = None

    # Populated by setup()
    _search_space: Any = field(default=None, repr=False)
    _reference_fn: Any = field(default=None, repr=False)
    _test_inputs: list[Any] = field(default_factory=list, repr=False)
    _output_shapes: list[tuple[int, ...]] = field(default_factory=list, repr=False)
    _output_dtypes: list[Any] = field(default_factory=list, repr=False)
    _baseline_us: float = 0.0
    _mx: Any = field(default=None, repr=False)
    _ready: bool = False

    def setup(self, warmup: int = 5, iters: int = 20) -> None:
        """Initialize test inputs, reference function, and baseline timing."""
        from .._compat import import_mx

        mx = import_mx()
        self._mx = mx
        mx.random.seed(42)

        # Build target search space
        target_factory = TARGETS[self.target_name]
        import inspect
        sig = inspect.signature(target_factory)
        kwargs: dict[str, Any] = {}
        if "D" in sig.parameters:
            kwargs["D"] = self.D
        if "K" in sig.parameters:
            kwargs["K"] = self.K
        self._search_space = target_factory(**kwargs)

        # Reference function
        self._reference_fn = self._search_space.make_reference_fn()

        # Test inputs from first concrete shape
        self._test_inputs = []
        for ispec in self._search_space.input_specs:
            shape = ispec.concrete_shapes[0] if ispec.concrete_shapes else (16,)
            if ispec.dtype == "uint32":
                inp = mx.zeros(shape, dtype=mx.uint32)
            else:
                inp = mx.random.normal(shape).astype(mx.float32)
            self._test_inputs.append(inp)

        # Output shapes/dtypes
        self._output_shapes = []
        self._output_dtypes = []
        for ospec in self._search_space.output_specs:
            shape = ospec.concrete_shapes[0] if ospec.concrete_shapes else (16,)
            self._output_shapes.append(shape)
            if ospec.dtype == "uint32":
                self._output_dtypes.append(mx.uint32)
            else:
                self._output_dtypes.append(mx.float32)

        # Baseline timing
        def _ref_run(*inputs: Any) -> Any:
            return self._reference_fn(*inputs)

        baseline_timings = _time_fn(
            _ref_run, tuple(self._test_inputs), mx,
            warmup=warmup, iters=iters,
        )
        baseline_timings.sort()
        self._baseline_us = baseline_timings[len(baseline_timings) // 2]
        self._ready = True

    @property
    def baseline_us(self) -> float:
        return self._baseline_us

    def evaluate(
        self,
        config: dict[str, Any],
        source: str,
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        *,
        warmup: int = 5,
        iters: int = 20,
        timeout_s: float = 10.0,
    ) -> ScoredConfig:
        """Evaluate a single config by compiling and benchmarking."""
        if not self._ready:
            raise RuntimeError("Call setup() before evaluate()")

        mx = self._mx

        # Adjust grid for batch dimension if needed
        actual_grid = self._adjust_grid(grid)

        spec = KernelSpec(
            name=f"kk_autosearch_{self.target_name}",
            input_names=self._search_space.input_names,
            output_names=self._search_space.output_names,
            source=source,
            header="",  # uses DEFAULT_HEADER in evaluate_candidate
            threadgroup=threadgroup,
            template_params=self._search_space.template_params,
        )
        candidate = KernelCandidate(spec=spec, generation=0)

        result = evaluate_candidate(
            candidate,
            self._reference_fn,
            self._test_inputs,
            baseline_us=self._baseline_us,
            warmup=warmup,
            iters=iters,
            timeout_s=timeout_s,
            output_shapes=self._output_shapes,
            output_dtypes=self._output_dtypes,
            grid=actual_grid,
            threadgroup=threadgroup,
            template=[("T", mx.float32)],
        )

        return ScoredConfig(
            config=config,
            source=source,
            grid=actual_grid,
            threadgroup=threadgroup,
            eval_result=result,
            reward=result.reward,
            speedup=result.speedup,
        )

    def _adjust_grid(self, grid: tuple[int, int, int]) -> tuple[int, int, int]:
        """Scale grid dimensions for the actual test input batch size."""
        ss = self._search_space
        if ss is None:
            return grid

        # For moe_combine: grid.y must be B (batch)
        if self.target_name in ("moe_combine", "glm_moe_combine"):
            B = int(self._test_inputs[0].shape[0])
            return (grid[0], B, grid[2])

        # For rmsnorm: grid.x = B * TG
        if self.target_name in ("rmsnorm", "glm_rmsnorm"):
            B = int(self._test_inputs[0].size) // self.D
            tg = grid[0]  # template returns (TG, 1, 1) for B=1
            return (B * tg, grid[1], grid[2])

        # For elementwise (swiglu): template grid is for D elements,
        # but actual test inputs may be (D,) so grid is already correct.
        # If batched, use the target's compute_grid.
        if self.target_name in ("fused_swiglu", "glm_fused_swiglu"):
            if ss.compute_grid is not None:
                actual_grid, _ = ss.compute_grid(self._test_inputs)
                return cast(tuple[int, int, int], actual_grid)

        return grid
