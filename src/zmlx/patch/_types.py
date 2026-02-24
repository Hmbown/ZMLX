from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

FusionSignatureItem = tuple[str, Any, Any]
FusionSignature = tuple[FusionSignatureItem, ...]
FusionLegacyCacheKey = tuple[str, FusionSignature]
FusionScopedCacheKey = tuple[str, str, FusionSignature]
FusionCacheKey = FusionLegacyCacheKey | FusionScopedCacheKey


@dataclass(frozen=True)
class FusionConfig:
    """Opt-in fusion runtime integration settings for patched modules."""

    enabled: bool = False
    validate: bool = True
    rtol: float = 1e-4
    atol: float = 1e-5
    # Empty means "all patched patterns".
    patterns: tuple[str, ...] = ()
    # When True, legacy (non-scoped) cache keys are migrated to scoped format on access.
    normalize_legacy_keys: bool = False

    def allows_pattern(self, pattern_name: str) -> bool:
        if not self.enabled:
            return False
        if not self.patterns:
            return True
        return pattern_name in self.patterns


@dataclass
class FusionRuntimeState:
    """Mutable cache/state for fusion integration safety decisions."""

    validated: set[FusionCacheKey] = field(default_factory=set)
    blacklist: set[FusionCacheKey] = field(default_factory=set)
    compile_failures: int = 0
    runtime_failures: int = 0
    validation_mismatches: int = 0
    blacklist_hits: int = 0


@dataclass(frozen=True)
class PatchConfig:
    """Configuration for model patching."""

    compute_dtype: str = "float32"
    threadgroup: int | str = 256
    verbose: bool = False
    moe_fused_swiglu_max_tokens: int | None = None
    fusion: FusionConfig = field(default_factory=FusionConfig)
    fusion_state: FusionRuntimeState = field(default_factory=FusionRuntimeState)


@dataclass
class PatchResult:
    """Summary of what was patched."""

    patched_count: int = 0
    pattern_counts: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    # Filled by smart_patch: pattern_name -> speedup ratio
    benchmarks: dict[str, float] = field(default_factory=dict)
    fusion_wrapped_count: int = 0
    fusion_skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Patched {self.patched_count} modules:"]
        for name, count in sorted(self.pattern_counts.items()):
            bench = self.benchmarks.get(name)
            suffix = f" ({bench:.3f}x)" if bench is not None else ""
            lines.append(f"  {name}: {count}{suffix}")
        if self.fusion_wrapped_count:
            lines.append(f"  Fusion wrappers: {self.fusion_wrapped_count}")
        if self.fusion_skipped:
            lines.append(f"  Fusion skipped: {len(self.fusion_skipped)}")
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
