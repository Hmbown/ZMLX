"""Device profiles and per-device autotuning configuration.

This module provides hardware-specific tuning profiles for different Apple Silicon
chips (M1/M2/M3/M4 × base/Pro/Max/Ultra) to optimize kernel performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class DeviceTuningProfile:
    """Hardware-specific tuning profile for Apple Silicon.
    
    Provides optimal default configurations for autotuning based on the
    specific chip variant (M1/M2/M3/M4 × base/Pro/Max/Ultra).
    
    Attributes:
        family: Chip family (M1, M2, M3, M4)
        variant: Chip variant (base, Pro, Max, Ultra)
        gpu_cores: Number of GPU cores
        memory_bandwidth_gbps: Memory bandwidth in GB/s
        default_threadgroup: Default threadgroup size for this chip
        threadgroup_candidates: Recommended threadgroup sizes for autotuning
        simd_width: SIMD group width (always 32 for current Apple GPUs)
        has_ray_tracing: Whether the chip supports hardware ray tracing
        autotune_warmup: Number of warmup iterations for autotuning
        autotune_iterations: Number of timing iterations for autotuning
    """
    family: str  # "M1", "M2", "M3", "M4"
    variant: str  # "", "Pro", "Max", "Ultra"
    gpu_cores: int
    memory_bandwidth_gbps: int
    default_threadgroup: int
    threadgroup_candidates: tuple[int, ...] = field(default_factory=lambda: (128, 256))
    simd_width: int = 32
    has_ray_tracing: bool = False
    autotune_warmup: int = 3
    autotune_iterations: int = 10

    @property
    def chip_id(self) -> str:
        """Unique identifier for this chip variant."""
        if self.variant:
            return f"{self.family}_{self.variant}"
        return self.family

    @property
    def full_name(self) -> str:
        """Human-readable full name."""
        v = f" {self.variant}" if self.variant else ""
        return f"Apple {self.family}{v}"

    def is_high_end(self) -> bool:
        """Check if this is a high-end chip (Pro/Max/Ultra)."""
        return self.variant in ("Pro", "Max", "Ultra")


# Per-chip specifications for all 16 Apple Silicon variants
# Data sourced from Apple technical specifications and benchmarking
_DEVICE_SPECS: dict[str, dict[str, Any]] = {
    # M1 Series
    "M1": {
        "gpu_cores": 8,
        "memory_bandwidth": 68,
        "default_tg": 128,
        "candidates": (64, 128, 256),
    },
    "M1_Pro": {
        "gpu_cores": 16,
        "memory_bandwidth": 200,
        "default_tg": 256,
        "candidates": (128, 256, 512),
    },
    "M1_Max": {
        "gpu_cores": 32,
        "memory_bandwidth": 400,
        "default_tg": 256,
        "candidates": (128, 256, 512),
    },
    "M1_Ultra": {
        "gpu_cores": 64,
        "memory_bandwidth": 800,
        "default_tg": 256,
        "candidates": (256, 512, 1024),
    },
    # M2 Series
    "M2": {
        "gpu_cores": 10,
        "memory_bandwidth": 100,
        "default_tg": 128,
        "candidates": (64, 128, 256),
    },
    "M2_Pro": {
        "gpu_cores": 19,
        "memory_bandwidth": 200,
        "default_tg": 256,
        "candidates": (128, 256, 512),
    },
    "M2_Max": {
        "gpu_cores": 38,
        "memory_bandwidth": 400,
        "default_tg": 256,
        "candidates": (128, 256, 512),
    },
    "M2_Ultra": {
        "gpu_cores": 76,
        "memory_bandwidth": 800,
        "default_tg": 256,
        "candidates": (256, 512, 1024),
    },
    # M3 Series
    "M3": {
        "gpu_cores": 10,
        "memory_bandwidth": 100,
        "default_tg": 128,
        "candidates": (64, 128, 256),
        "has_ray_tracing": True,
    },
    "M3_Pro": {
        "gpu_cores": 18,
        "memory_bandwidth": 150,
        "default_tg": 256,
        "candidates": (128, 256, 512),
        "has_ray_tracing": True,
    },
    "M3_Max": {
        "gpu_cores": 40,
        "memory_bandwidth": 400,
        "default_tg": 256,
        "candidates": (128, 256, 512),
        "has_ray_tracing": True,
    },
    "M3_Ultra": {
        "gpu_cores": 80,
        "memory_bandwidth": 800,
        "default_tg": 256,
        "candidates": (256, 512, 1024),
        "has_ray_tracing": True,
    },
    # M4 Series
    "M4": {
        "gpu_cores": 10,
        "memory_bandwidth": 120,
        "default_tg": 128,
        "candidates": (64, 128, 256),
        "has_ray_tracing": True,
    },
    "M4_Pro": {
        "gpu_cores": 20,
        "memory_bandwidth": 273,
        "default_tg": 256,
        "candidates": (128, 256, 512),
        "has_ray_tracing": True,
    },
    "M4_Max": {
        "gpu_cores": 40,
        "memory_bandwidth": 546,
        "default_tg": 256,
        "candidates": (256, 512, 1024),
        "has_ray_tracing": True,
    },
    "M4_Ultra": {
        "gpu_cores": 80,  # Projected
        "memory_bandwidth": 1000,  # Projected
        "default_tg": 256,
        "candidates": (256, 512, 1024),
        "has_ray_tracing": True,
    },
}


def get_device_profile(family: str, variant: str = "") -> DeviceTuningProfile:
    """Get the tuning profile for a specific Apple Silicon chip.
    
    Args:
        family: Chip family (M1, M2, M3, M4)
        variant: Chip variant (base, Pro, Max, Ultra)
    
    Returns:
        DeviceTuningProfile with optimal defaults for the chip.
        
    Example:
        >>> profile = get_device_profile("M3", "Max")
        >>> print(profile.gpu_cores)  # 40
        >>> print(profile.default_threadgroup)  # 256
    """
    key = f"{family}_{variant}".rstrip("_")
    specs = _DEVICE_SPECS.get(key, _DEVICE_SPECS.get("M1", {}))
    
    return DeviceTuningProfile(
        family=family,
        variant=variant,
        gpu_cores=specs.get("gpu_cores", 8),
        memory_bandwidth_gbps=specs.get("memory_bandwidth", 68),
        default_threadgroup=specs.get("default_tg", 128),
        threadgroup_candidates=specs.get("candidates", (128, 256)),
        has_ray_tracing=specs.get("has_ray_tracing", False),
    )


@lru_cache(maxsize=1)
def get_current_device_profile() -> DeviceTuningProfile:
    """Detect the current device and return its tuning profile.
    
    Returns:
        DeviceTuningProfile for the current Apple Silicon chip,
        or a conservative default if detection fails.
    """
    try:
        from .device import detect_device
        dev = detect_device()
        
        # Extract family and variant from detected device
        family = dev.family if dev.family.startswith("M") else "M1"
        variant = dev.variant
        
        # Override GPU cores if we detected them correctly
        profile = get_device_profile(family, variant)
        
        # Use detected GPU cores if available, otherwise use profile default
        if dev.gpu_cores > 0:
            # Create a new profile with detected values
            return DeviceTuningProfile(
                family=family,
                variant=variant,
                gpu_cores=dev.gpu_cores,
                memory_bandwidth_gbps=profile.memory_bandwidth_gbps,
                default_threadgroup=profile.default_threadgroup,
                threadgroup_candidates=profile.threadgroup_candidates,
                simd_width=dev.simd_width,
                has_ray_tracing=dev.has_ray_tracing,
            )
        return profile
    except Exception:
        # Return conservative default
        return DeviceTuningProfile(
            family="Unknown",
            variant="",
            gpu_cores=8,
            memory_bandwidth_gbps=68,
            default_threadgroup=128,
            threadgroup_candidates=(64, 128, 256),
        )


def get_threadgroup_candidates_for_shape(
    profile: DeviceTuningProfile,
    dim_size: int,
    kernel_type: str = "general",
) -> tuple[int, ...]:
    """Get optimized threadgroup candidates for a specific kernel shape.
    
    Args:
        profile: The device tuning profile
        dim_size: The size of the dimension being processed
        kernel_type: Type of kernel ("general", "reduction", "simd")
    
    Returns:
        Tuple of recommended threadgroup sizes for autotuning.
    """
    base_candidates = list(profile.threadgroup_candidates)
    
    if kernel_type == "simd":
        # SIMD kernels prefer multiples of 32 (SIMD width)
        return tuple(c for c in base_candidates if c % 32 == 0)
    
    elif kernel_type == "reduction":
        # Reduction kernels benefit from power-of-2 threadgroups
        # and should be sized to handle the reduction efficiently
        candidates = []
        for c in base_candidates:
            if dim_size <= c:
                # Use exact match or next power of 2
                candidates.append(c)
            else:
                candidates.append(c)
        return tuple(set(candidates))
    
    # General case: return base candidates
    return tuple(base_candidates)


# Export all profiles for easy access
ALL_DEVICE_PROFILES: list[DeviceTuningProfile] = [
    get_device_profile(family, variant)
    for family in ("M1", "M2", "M3", "M4")
    for variant in ("", "Pro", "Max", "Ultra")
]


__all__ = [
    "DeviceTuningProfile",
    "get_device_profile",
    "get_current_device_profile",
    "get_threadgroup_candidates_for_shape",
    "ALL_DEVICE_PROFILES",
]
