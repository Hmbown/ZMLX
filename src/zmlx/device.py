"""Device detection and profiling for Apple Silicon GPUs."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DeviceProfile:
    """Hardware profile for an Apple Silicon chip."""

    family: str  # e.g. "M1", "M2", "M3", "M4"
    variant: str  # e.g. "Pro", "Max", "Ultra", ""
    gpu_cores: int
    max_threadgroup_size: int  # Usually 1024
    simd_width: int  # 32 for all current Apple GPU
    has_ray_tracing: bool  # M3+ only
    memory_gb: int

    @property
    def full_name(self) -> str:
        v = f" {self.variant}" if self.variant else ""
        return f"Apple {self.family}{v}"

    @property
    def default_threadgroup_candidates(self) -> list[int]:
        """Reasonable threadgroup sizes to search for autotuning."""
        if self.gpu_cores >= 30:  # Pro/Max/Ultra
            return [64, 128, 256, 512, 1024]
        return [32, 64, 128, 256, 512]


_FAMILY_PATTERNS: dict[str, tuple[str, str]] = {
    "Apple M1": ("M1", ""),
    "Apple M1 Pro": ("M1", "Pro"),
    "Apple M1 Max": ("M1", "Max"),
    "Apple M1 Ultra": ("M1", "Ultra"),
    "Apple M2": ("M2", ""),
    "Apple M2 Pro": ("M2", "Pro"),
    "Apple M2 Max": ("M2", "Max"),
    "Apple M2 Ultra": ("M2", "Ultra"),
    "Apple M3": ("M3", ""),
    "Apple M3 Pro": ("M3", "Pro"),
    "Apple M3 Max": ("M3", "Max"),
    "Apple M3 Ultra": ("M3", "Ultra"),
    "Apple M4": ("M4", ""),
    "Apple M4 Pro": ("M4", "Pro"),
    "Apple M4 Max": ("M4", "Max"),
    "Apple M4 Ultra": ("M4", "Ultra"),
}


def _sysctl(key: str) -> str:
    """Read a sysctl value."""
    try:
        return subprocess.check_output(
            ["sysctl", "-n", key],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


@lru_cache(maxsize=1)
def detect_device() -> DeviceProfile:
    """Detect the current Apple Silicon device.

    Returns a :class:`DeviceProfile` with hardware details.
    Falls back to a conservative default if detection fails.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return DeviceProfile(
            family="Unknown",
            variant="",
            gpu_cores=0,
            max_threadgroup_size=1024,
            simd_width=32,
            has_ray_tracing=False,
            memory_gb=0,
        )

    chip_name = _sysctl("machdep.cpu.brand_string")

    family = "Unknown"
    variant = ""
    for pattern, (fam, var) in _FAMILY_PATTERNS.items():
        if pattern in chip_name:
            family = fam
            variant = var
            break

    # GPU cores
    gpu_cores_str = _sysctl("hw.perflevel0.logicalcpu")
    try:
        gpu_cores = int(gpu_cores_str)
    except (ValueError, TypeError):
        gpu_cores = 8  # Conservative default

    # Memory
    mem_str = _sysctl("hw.memsize")
    try:
        memory_gb = int(mem_str) // (1024 * 1024 * 1024)
    except (ValueError, TypeError):
        memory_gb = 8

    # Ray tracing: M3+
    has_rt = family in ("M3", "M4")

    return DeviceProfile(
        family=family,
        variant=variant,
        gpu_cores=gpu_cores,
        max_threadgroup_size=1024,
        simd_width=32,
        has_ray_tracing=has_rt,
        memory_gb=memory_gb,
    )
