"""Plugin protocol definitions for the ZMLX foundry.

Each protocol defines a minimal contract that plugins must satisfy.
Using ``Protocol`` (structural subtyping) rather than ABC so that
external plugin authors do not need to inherit from anything.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Context objects passed to plugins
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryContext:
    """Context passed to discovery plugins."""

    op: str
    dtype: str
    shape_space: dict[str, Any]
    constraints: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


@dataclass
class FoundryContext:
    """Context passed to foundry plugins."""

    op: str
    dtype: str
    curriculum: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


# ---------------------------------------------------------------------------
# Candidate / artifact types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """A proposed kernel candidate from a discovery plugin."""

    template_id: str
    knobs: dict[str, Any]
    notes: str | None = None


@dataclass(frozen=True)
class AttemptSpec:
    """A fully specified attempt plan entry consumed by the harness.

    Carries everything needed to compile, run, and validate a single
    kernel attempt.
    """

    op: str
    dtype: str
    shape: dict[str, Any]
    spec: dict[str, Any]
    template_id: str
    knobs: dict[str, Any]
    compile_flags: list[str] = field(default_factory=list)
    launch: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportArtifacts:
    """Describes the output of an export plugin."""

    out_dir: str
    files: list[str]
    n_records: int


# ---------------------------------------------------------------------------
# Plugin protocols
# ---------------------------------------------------------------------------

class DiscoveryPlugin(Protocol):
    """Proposes kernel candidates and learns from evaluation results."""

    def propose_candidates(self, context: DiscoveryContext) -> list[Candidate]:
        ...

    def update(self, context: DiscoveryContext, results: list[dict[str, Any]]) -> None:
        ...


class FoundryPlugin(Protocol):
    """Generates a full attempt plan for dataset construction."""

    def generate_attempt_plan(
        self,
        target_attempts: int,
        curriculum: dict[str, Any],
    ) -> Iterator[AttemptSpec]:
        ...


class MoEPlugin(Protocol):
    """Registers MoE-specific ops and provides property tests."""

    def register_ops(self, registry: Any) -> None:
        ...

    def property_tests(self) -> list[Callable[[], None]]:
        ...


class ExportPlugin(Protocol):
    """Exports session data to a target format."""

    def export(self, session_dir: str, out_dir: str) -> ExportArtifacts:
        ...
