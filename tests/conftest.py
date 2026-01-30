"""Pytest configuration for ZMLX cross-backend testing.

This module provides:
- Backend detection and pytest markers
- Test filtering for Metal/GPU/CPU backends
- Cross-backend golden value testing utilities
"""

import platform
import sys

import pytest


def _metal_available() -> bool:
    """Check if Metal is available and functional."""
    try:
        import mlx.core as mx
    except Exception:
        return False
    metal = getattr(mx, "metal", None)
    if metal is None:
        return False
    fn = getattr(metal, "is_available", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def _cuda_available() -> bool:
    """Check if CUDA is available and functional."""
    try:
        import mlx.core as mx
    except Exception:
        return False
    cuda = getattr(mx, "cuda", None)
    if cuda is None:
        return False
    fn = getattr(cuda, "is_available", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def _detect_backend() -> str:
    """Detect the MLX backend."""
    if _metal_available():
        return "metal"
    if _cuda_available():
        return "cuda"
    try:
        import mlx.core as mx  # noqa: F401
        return "cpu"
    except Exception:
        return "none"


# Detect backend at module load time
_BACKEND = _detect_backend()
_IS_MACOS = platform.system() == "Darwin"
_IS_ARM64 = platform.machine() in ("arm64", "aarch64")


def pytest_configure(config):
    """Configure pytest markers."""
    # Register custom markers
    config.addinivalue_line("markers", "metal: mark test as requiring Metal GPU")
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA GPU")
    config.addinivalue_line("markers", "gpu: mark test as requiring any GPU (Metal or CUDA)")
    config.addinivalue_line("markers", "cpu: mark test as CPU-compatible (pure Python)")
    config.addinivalue_line("markers", "slow: mark test as slow (performance/integration)")
    config.addinivalue_line("markers", "golden: mark test as using golden values for cross-backend comparison")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on backend capabilities."""
    # Determine what to skip based on backend
    skip_metal = pytest.mark.skip(reason="Metal kernel tests require macOS arm64 with Metal available.")
    skip_cuda = pytest.mark.skip(reason="CUDA tests require CUDA backend.")
    skip_gpu = pytest.mark.skip(reason="GPU tests require Metal or CUDA backend.")
    skip_no_mlx = pytest.mark.skip(reason="Tests require MLX to be installed.")
    
    for item in items:
        # Skip Metal tests on non-Metal backends
        if "metal" in item.keywords and _BACKEND != "metal":
            item.add_marker(skip_metal)
        
        # Skip CUDA tests on non-CUDA backends
        if "cuda" in item.keywords and _BACKEND != "cuda":
            item.add_marker(skip_cuda)
        
        # Skip GPU tests if no GPU backend
        if "gpu" in item.keywords and _BACKEND not in ("metal", "cuda"):
            item.add_marker(skip_gpu)
        
        # Mark all tests as requiring MLX if they don't have a specific backend marker
        has_backend_marker = any(
            m in item.keywords for m in ("metal", "cuda", "gpu", "cpu")
        )
        if not has_backend_marker and _BACKEND == "none":
            item.add_marker(skip_no_mlx)


# Pytest fixtures

@pytest.fixture(scope="session")
def backend():
    """Fixture providing the detected backend ("metal", "cuda", "cpu", or "none")."""
    return _BACKEND


@pytest.fixture(scope="session")
def is_metal():
    """Fixture returning True if Metal is the current backend."""
    return _BACKEND == "metal"


@pytest.fixture(scope="session")
def is_cuda():
    """Fixture returning True if CUDA is the current backend."""
    return _BACKEND == "cuda"


@pytest.fixture(scope="session")
def is_cpu():
    """Fixture returning True if CPU is the current backend."""
    return _BACKEND == "cpu"


@pytest.fixture(scope="session")
def has_gpu():
    """Fixture returning True if any GPU backend is available."""
    return _BACKEND in ("metal", "cuda")


@pytest.fixture(scope="session")
def mx_device(backend):
    """Fixture providing the MLX device for the current backend."""
    import mlx.core as mx
    
    if backend == "metal":
        return mx.gpu
    elif backend == "cuda":
        return mx.gpu
    else:
        return mx.cpu


# Golden value testing utilities

_golden_registry: dict[str, dict] = {}


def register_golden(name: str, value: any, backend: str = None):
    """Register a golden value for cross-backend testing.
    
    Args:
        name: Unique identifier for this golden value
        value: The reference value (numpy array or scalar)
        backend: The backend that produced this value ("metal", "cuda", "cpu")
    """
    if name not in _golden_registry:
        _golden_registry[name] = {}
    _golden_registry[name][backend or _BACKEND] = value


def get_golden(name: str, backend: str = None, default: any = None):
    """Get a registered golden value.
    
    Args:
        name: Unique identifier for the golden value
        backend: The backend to get the value for (defaults to current)
        default: Default value if not found
    """
    backend = backend or _BACKEND
    if name not in _golden_registry:
        return default
    return _golden_registry[name].get(backend, default)


@pytest.fixture
def golden_registry():
    """Fixture providing access to golden value registry."""
    return {
        "register": register_golden,
        "get": get_golden,
        "all": lambda: dict(_golden_registry),
    }


@pytest.fixture
def assert_allclose_cross_backend():
    """Fixture providing a cross-backend assert_allclose function.
    
    This function compares arrays with tolerances appropriate for the
    specific backends being compared.
    """
    import mlx.core as mx
    
    def _assert_allclose(
        actual,
        desired,
        rtol: float = 1e-4,
        atol: float = 1e-4,
        backend: str = None,
    ):
        """Assert that two arrays are close within tolerance.
        
        Automatically adjusts tolerances based on backend:
        - Metal vs CPU: higher tolerance for numerical differences
        - CUDA vs CPU: standard tolerance
        - Metal vs CUDA: higher tolerance
        """
        backend = backend or _BACKEND
        
        # Adjust tolerances based on backend comparison
        if backend in ("metal", "cuda") and _BACKEND != backend:
            # Comparing GPU to CPU
            rtol = max(rtol, 1e-3)
            atol = max(atol, 1e-3)
        
        # Convert to numpy for comparison
        if hasattr(actual, "tolist"):
            actual = actual.tolist()
        if hasattr(desired, "tolist"):
            desired = desired.tolist()
        
        import numpy as np
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)
    
    return _assert_allclose
