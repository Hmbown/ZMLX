"""Tests for per-device autotune profiles.

These tests validate the device profile system and ensure that
appropriate defaults are selected for different Apple Silicon chips.
"""

import pytest

# Pure Python tests - can run on any backend
pytestmark = [pytest.mark.cpu]


class TestDeviceProfile:
    """Test device profile creation and attributes."""
    
    def test_get_device_profile_m1_base(self):
        """Test M1 base profile."""
        from zmlx.device_profile import get_device_profile
        
        profile = get_device_profile("M1", "")
        assert profile.family == "M1"
        assert profile.variant == ""
        assert profile.gpu_cores == 8
        assert profile.memory_bandwidth_gbps == 68
        assert profile.default_threadgroup == 128
        assert profile.chip_id == "M1"
        assert profile.full_name == "Apple M1"
        assert not profile.is_high_end()
    
    def test_get_device_profile_m3_max(self):
        """Test M3 Max profile."""
        from zmlx.device_profile import get_device_profile
        
        profile = get_device_profile("M3", "Max")
        assert profile.family == "M3"
        assert profile.variant == "Max"
        assert profile.gpu_cores == 40
        assert profile.memory_bandwidth_gbps == 400
        assert profile.default_threadgroup == 256
        assert profile.chip_id == "M3_Max"
        assert profile.full_name == "Apple M3 Max"
        assert profile.is_high_end()
        assert profile.has_ray_tracing
    
    def test_get_device_profile_m4_pro(self):
        """Test M4 Pro profile."""
        from zmlx.device_profile import get_device_profile
        
        profile = get_device_profile("M4", "Pro")
        assert profile.family == "M4"
        assert profile.variant == "Pro"
        assert profile.gpu_cores == 20
        assert profile.memory_bandwidth_gbps == 273
        assert profile.default_threadgroup == 256
    
    def test_all_device_profiles_exist(self):
        """Test that all 16 chip variants have profiles."""
        from zmlx.device_profile import ALL_DEVICE_PROFILES, get_device_profile
        
        families = ["M1", "M2", "M3", "M4"]
        variants = ["", "Pro", "Max", "Ultra"]
        
        for family in families:
            for variant in variants:
                profile = get_device_profile(family, variant)
                assert profile.family == family
                assert profile.variant == variant
                assert profile.gpu_cores > 0
                assert profile.memory_bandwidth_gbps > 0
        
        # Verify ALL_DEVICE_PROFILES contains all variants
        assert len(ALL_DEVICE_PROFILES) == 16
    
    def test_threadgroup_candidates(self):
        """Test threadgroup candidate selection."""
        from zmlx.device_profile import (
            get_device_profile,
            get_threadgroup_candidates_for_shape,
        )
        
        profile = get_device_profile("M3", "Max")
        
        # General kernel
        candidates = get_threadgroup_candidates_for_shape(profile, 1024, "general")
        assert len(candidates) > 0
        assert all(c > 0 for c in candidates)
        
        # SIMD kernel
        simd_candidates = get_threadgroup_candidates_for_shape(profile, 1024, "simd")
        assert all(c % 32 == 0 for c in simd_candidates)
        
        # Reduction kernel
        red_candidates = get_threadgroup_candidates_for_shape(profile, 256, "reduction")
        assert len(red_candidates) > 0


class TestAutotuneIntegration:
    """Test autotune integration with device profiles."""
    
    def test_get_device_candidates(self):
        """Test that device candidates are returned."""
        from zmlx.autotune import _get_device_candidates
        
        candidates = _get_device_candidates()
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        # Each candidate should be a (x, y, z) tuple
        for c in candidates:
            assert len(c) == 3
            assert all(isinstance(v, int) for v in c)
    
    def test_get_default_config(self):
        """Test that default config can be retrieved."""
        from zmlx.autotune import _get_default_config
        
        config = _get_default_config()
        assert config.threadgroup is not None
        assert len(config.threadgroup) == 3
    
    def test_autotune_decorator(self):
        """Test the @autotune decorator."""
        from zmlx.autotune import AutotunedFunction, autotune
        
        @autotune(warmup=2, iters=5)
        def dummy_kernel(x, threadgroup=(128, 1, 1)):
            return x
        
        assert isinstance(dummy_kernel, AutotunedFunction)
        assert dummy_kernel.warmup == 2
        assert dummy_kernel.iters == 5


class TestDeviceDetection:
    """Test device detection functionality."""
    
    def test_detect_current_device(self):
        """Test that we can detect the current device profile."""
        from zmlx.device_profile import get_current_device_profile
        
        profile = get_current_device_profile()
        assert profile is not None
        # Should return at least a default profile
        assert profile.default_threadgroup > 0
    
    def test_fallback_on_detection_failure(self):
        """Test fallback when detection fails."""
        from zmlx.device_profile import DeviceTuningProfile, get_current_device_profile
        
        # This should not raise an exception
        profile = get_current_device_profile()
        assert isinstance(profile, DeviceTuningProfile)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
