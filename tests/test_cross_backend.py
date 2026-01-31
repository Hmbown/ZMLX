"""Tests for cross-backend testing infrastructure.

These tests validate the backend detection and cross-backend
testing utilities work correctly.
"""

import platform

import pytest

# Pure Python tests
pytestmark = [pytest.mark.cpu]


class TestBackendDetection:
    """Test backend detection in _compat module."""
    
    def test_is_macos(self):
        """Test macOS detection."""
        from zmlx._compat import is_macos
        
        result = is_macos()
        expected = platform.system() == "Darwin"
        assert result == expected
    
    def test_is_arm64(self):
        """Test ARM64 detection."""
        from zmlx._compat import is_arm64
        
        result = is_arm64()
        expected = platform.machine() in ("arm64", "aarch64")
        assert result == expected
    
    def test_detect_backend(self):
        """Test backend detection."""
        from zmlx._compat import detect_backend
        
        backend = detect_backend()
        assert backend in ("metal", "cuda", "cpu", "unknown")
    
    def test_has_gpu_backend(self):
        """Test GPU backend detection."""
        from zmlx._compat import detect_backend, has_gpu_backend
        
        has_gpu = has_gpu_backend()
        backend = detect_backend()
        
        if backend in ("metal", "cuda"):
            assert has_gpu is True
        else:
            assert has_gpu is False
    
    def test_is_metal_available(self):
        """Test Metal availability check."""
        from zmlx._compat import is_arm64, is_macos, is_metal_available
        
        is_metal = is_metal_available()
        
        # Metal should only be available on macOS ARM64
        if is_metal:
            assert is_macos()
            assert is_arm64()


class TestImportMx:
    """Test mlx.core import utilities."""
    
    def test_import_mx(self):
        """Test that mlx.core can be imported."""
        from zmlx._compat import import_mx
        
        mx = import_mx()
        assert mx is not None
        # Should be able to create an array
        arr = mx.array([1.0, 2.0, 3.0])
        assert arr is not None


class TestCrossBackendFixtures:
    """Test that pytest fixtures work correctly."""
    
    def test_backend_fixture(self, backend):
        """Test backend fixture."""
        assert backend in ("metal", "cuda", "cpu", "none")
    
    def test_is_metal_fixture(self, is_metal):
        """Test is_metal fixture."""
        assert isinstance(is_metal, bool)
    
    def test_is_cuda_fixture(self, is_cuda):
        """Test is_cuda fixture."""
        assert isinstance(is_cuda, bool)
    
    def test_is_cpu_fixture(self, is_cpu):
        """Test is_cpu fixture."""
        assert isinstance(is_cpu, bool)
    
    def test_has_gpu_fixture(self, has_gpu):
        """Test has_gpu fixture."""
        assert isinstance(has_gpu, bool)
    
    def test_mx_device_fixture(self, mx_device):
        """Test mx_device fixture."""
        import mlx.core as mx
        
        # Should be able to create a tensor on this device
        arr = mx.array([1.0, 2.0, 3.0])
        assert arr is not None


class TestGoldenValues:
    """Test golden value utilities."""
    
    def test_register_and_get_golden(self, golden_registry):
        """Test golden value registration and retrieval."""
        # Register a golden value
        golden_registry["register"]("test_value", [1.0, 2.0, 3.0], "test_backend")
        
        # Retrieve it
        value = golden_registry["get"]("test_value", "test_backend")
        assert value == [1.0, 2.0, 3.0]
    
    def test_get_nonexistent_golden(self, golden_registry):
        """Test retrieving a non-existent golden value."""
        value = golden_registry["get"]("nonexistent", "test_backend")
        assert value is None
    
    def test_assert_allclose_cross_backend(self, assert_allclose_cross_backend):
        """Test cross-backend allclose assertion."""
        # Should pass for identical arrays
        assert_allclose_cross_backend([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        
        # Should pass for arrays within tolerance
        assert_allclose_cross_backend([1.0, 2.0], [1.0001, 2.0001], rtol=1e-3)


@pytest.mark.metal
class TestMetalSpecific:
    """Tests that specifically require Metal."""
    
    def test_metal_backend(self, backend):
        """Test that Metal backend is detected."""
        assert backend == "metal"


@pytest.mark.gpu
class TestGpuSpecific:
    """Tests that require any GPU backend."""
    
    def test_gpu_backend(self, has_gpu):
        """Test that GPU is available."""
        assert has_gpu is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
