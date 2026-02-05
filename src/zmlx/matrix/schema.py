"""Matrix entry and snapshot data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatrixEntry:
    """One test result: a (model, patterns, hardware) combination."""

    # Identity
    model_id: str
    model_family: str
    architecture: str  # "moe" or "dense"
    patterns_applied: list[str]
    hardware: str  # e.g. "Apple M4 Max 36GB"
    notes: str = ""

    # Environment
    macos_version: str = ""
    mlx_version: str = ""
    zmlx_version: str = ""
    zmlx_commit: str = ""
    python_version: str = ""
    custom_mlx: bool = False
    timestamp: str = ""  # ISO 8601

    # Results
    fidelity: str = "SKIP"  # "PASS", "FAIL", "SKIP", "EXCLUDED"
    fidelity_detail: str = ""
    modules_patched: int = 0

    decode_tps_baseline: float = 0.0
    decode_tps_patched: float = 0.0
    decode_speedup: float = 0.0
    decode_runs_baseline: list[float] = field(default_factory=list)
    decode_runs_patched: list[float] = field(default_factory=list)

    prefill_tps_baseline: float = 0.0
    prefill_tps_patched: float = 0.0
    prefill_change: float = 0.0

    peak_mem_baseline_gb: float = 0.0
    peak_mem_patched_gb: float = 0.0

    max_tokens: int = 0
    gen_tokens: int = 0
    runs: int = 0

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "model_id": self.model_id,
            "model_family": self.model_family,
            "architecture": self.architecture,
            "patterns_applied": self.patterns_applied,
            "hardware": self.hardware,
            "notes": self.notes,
            "macos_version": self.macos_version,
            "mlx_version": self.mlx_version,
            "zmlx_version": self.zmlx_version,
            "zmlx_commit": self.zmlx_commit,
            "python_version": self.python_version,
            "custom_mlx": self.custom_mlx,
            "timestamp": self.timestamp,
            "fidelity": self.fidelity,
            "fidelity_detail": self.fidelity_detail,
            "modules_patched": self.modules_patched,
            "decode_tps_baseline": self.decode_tps_baseline,
            "decode_tps_patched": self.decode_tps_patched,
            "decode_speedup": self.decode_speedup,
            "decode_runs_baseline": self.decode_runs_baseline,
            "decode_runs_patched": self.decode_runs_patched,
            "prefill_tps_baseline": self.prefill_tps_baseline,
            "prefill_tps_patched": self.prefill_tps_patched,
            "prefill_change": self.prefill_change,
            "peak_mem_baseline_gb": self.peak_mem_baseline_gb,
            "peak_mem_patched_gb": self.peak_mem_patched_gb,
            "max_tokens": self.max_tokens,
            "gen_tokens": self.gen_tokens,
            "runs": self.runs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MatrixEntry:
        """Deserialize from a dict (tolerant of missing keys)."""
        return cls(
            model_id=d.get("model_id", ""),
            model_family=d.get("model_family", ""),
            architecture=d.get("architecture", ""),
            patterns_applied=d.get("patterns_applied", []),
            hardware=d.get("hardware", ""),
            notes=d.get("notes", ""),
            macos_version=d.get("macos_version", ""),
            mlx_version=d.get("mlx_version", ""),
            zmlx_version=d.get("zmlx_version", ""),
            zmlx_commit=d.get("zmlx_commit", ""),
            python_version=d.get("python_version", ""),
            custom_mlx=d.get("custom_mlx", False),
            timestamp=d.get("timestamp", ""),
            fidelity=d.get("fidelity", "SKIP"),
            fidelity_detail=d.get("fidelity_detail", ""),
            modules_patched=d.get("modules_patched", 0),
            decode_tps_baseline=d.get("decode_tps_baseline", 0.0),
            decode_tps_patched=d.get("decode_tps_patched", 0.0),
            decode_speedup=d.get("decode_speedup", 0.0),
            decode_runs_baseline=d.get("decode_runs_baseline", []),
            decode_runs_patched=d.get("decode_runs_patched", []),
            prefill_tps_baseline=d.get("prefill_tps_baseline", 0.0),
            prefill_tps_patched=d.get("prefill_tps_patched", 0.0),
            prefill_change=d.get("prefill_change", 0.0),
            peak_mem_baseline_gb=d.get("peak_mem_baseline_gb", 0.0),
            peak_mem_patched_gb=d.get("peak_mem_patched_gb", 0.0),
            max_tokens=d.get("max_tokens", 0),
            gen_tokens=d.get("gen_tokens", 0),
            runs=d.get("runs", 0),
        )


@dataclass
class MatrixSnapshot:
    """A full matrix state: list of entries + metadata."""

    entries: list[MatrixEntry] = field(default_factory=list)
    hardware: str = ""
    date: str = ""
    notes: str = ""
