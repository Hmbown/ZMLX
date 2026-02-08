"""Data structures for kernel candidates and search spaces."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InputSpec:
    """Specification for a kernel input buffer."""

    name: str
    shape_expr: str  # e.g. "(B, K, D)"
    dtype: str  # e.g. "float32"
    concrete_shapes: Sequence[tuple[int, ...]] = field(default_factory=list)


@dataclass(frozen=True)
class OutputSpec:
    """Specification for a kernel output buffer."""

    name: str
    shape_expr: str
    dtype: str
    concrete_shapes: Sequence[tuple[int, ...]] = field(default_factory=list)


@dataclass(frozen=True)
class KernelSpec:
    """Immutable specification for a Metal kernel."""

    name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    source: str
    header: str = ""
    threadgroup: tuple[int, int, int] = (256, 1, 1)
    template_params: tuple[tuple[str, str], ...] = ()

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.source.encode()).hexdigest()


@dataclass(frozen=True)
class KernelCandidate:
    """A kernel candidate produced by the search process."""

    spec: KernelSpec
    parent_id: str | None = None
    generation: int = 0
    llm_reasoning: str = ""

    @property
    def source_hash(self) -> str:
        return self.spec.source_hash

    @property
    def candidate_id(self) -> str:
        return f"gen{self.generation}_{self.source_hash[:12]}"


@dataclass
class EvalResult:
    """Result of evaluating a kernel candidate."""

    compiled: bool = False
    correct: bool = False
    compile_error: str | None = None
    correctness_error: str | None = None
    timings_us: list[float] = field(default_factory=list)
    median_us: float = float("inf")
    reward: float = 0.0
    speedup: float = 0.0


@dataclass(frozen=True)
class SearchSpace:
    """Definition of a kernel optimization target."""

    name: str
    description: str
    reference_source: str
    reference_python: str  # Python code string describing the reference fn
    input_specs: tuple[InputSpec, ...]
    output_specs: tuple[OutputSpec, ...]
    constraints: tuple[str, ...] = ()
    seed_candidates: tuple[KernelCandidate, ...] = ()
    # Kernel compilation metadata
    input_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    header: str = ""
    grid_fn: str = ""  # description of how to compute grid
    compute_grid: Callable[..., tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None
    template_params: tuple[tuple[str, str], ...] = ()

    def make_reference_fn(self) -> Any:
        """Execute reference_python to get the reference callable.

        The reference_python should define a function named ``reference``.
        """
        ns: dict[str, Any] = {}
        exec(self.reference_python, ns)  # noqa: S102
        return ns["reference"]
