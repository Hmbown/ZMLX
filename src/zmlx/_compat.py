from __future__ import annotations

import platform
from typing import Any


def is_macos() -> bool:
    return platform.system() == "Darwin"

def is_arm64() -> bool:
    """Check if running on ARM64 architecture (Apple Silicon or ARM Linux)."""
    return platform.machine() in ("arm64", "aarch64")

def is_supported_host() -> bool:
    """Check if the host supports ZMLX Metal kernels.
    
    Note: This checks for Metal support specifically. ZMLX can still be
    used for pure-Python components on other platforms.
    """
    return is_macos() and is_arm64()


def detect_backend() -> str:
    """Detect the MLX backend being used.
    
    Returns one of: "metal", "cuda", "cpu", "unknown"
    
    - "metal": macOS with Metal GPU support
    - "cuda": Linux with CUDA support
    - "cpu": CPU-only backend
    - "unknown": Unable to detect backend
    """
    try:
        import mlx.core as mx
    except ImportError:
        return "unknown"
    
    # Check for Metal
    if hasattr(mx, "metal"):
        metal = getattr(mx, "metal")
        is_available = getattr(metal, "is_available", None)
        if callable(is_available):
            try:
                if is_available():
                    return "metal"
            except Exception:
                pass
    
    # Check for CUDA
    if hasattr(mx, "cuda"):
        cuda = getattr(mx, "cuda")
        is_available = getattr(cuda, "is_available", None)
        if callable(is_available):
            try:
                if is_available():
                    return "cuda"
            except Exception:
                pass
    
    # Check for CPU
    if hasattr(mx, "cpu") or not is_macos():
        return "cpu"
    
    return "unknown"


def has_gpu_backend() -> bool:
    """Check if a GPU backend (Metal or CUDA) is available."""
    backend = detect_backend()
    return backend in ("metal", "cuda")


def is_metal_available() -> bool:
    """Check if Metal is available and functional."""
    try:
        import mlx.core as mx
        metal = getattr(mx, "metal", None)
        if metal is None:
            return False
        fn = getattr(metal, "is_available", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    return False


def import_mx() -> Any:
    """Import mlx.core lazily to keep import errors friendly."""
    import mlx.core as mx  # type: ignore
    return mx
