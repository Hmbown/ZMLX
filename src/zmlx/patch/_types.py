from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PatchConfig:
    """Configuration for model patching."""

    compute_dtype: str = "float32"
    threadgroup: int | str = 256
    verbose: bool = False


@dataclass
class PatchResult:
    """Summary of what was patched."""

    patched_count: int = 0
    pattern_counts: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    # Filled by smart_patch: pattern_name -> speedup ratio
    benchmarks: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Patched {self.patched_count} modules:"]
        for name, count in sorted(self.pattern_counts.items()):
            bench = self.benchmarks.get(name)
            suffix = f" ({bench:.3f}x)" if bench is not None else ""
            lines.append(f"  {name}: {count}{suffix}")
        if self.skipped:
            lines.append(f"  Skipped: {len(self.skipped)}")
        return "\n".join(lines)


@runtime_checkable
class PatchPattern(Protocol):
    """Protocol for a single patch pattern."""

    @property
    def name(self) -> str: ...

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        """Return True if this pattern applies to the given module."""
        ...

    def apply(self, module: Any, config: PatchConfig) -> Any:
        """Return a replacement module (or the same module modified in place)."""
        ...
