# Cross-Backend Testing

This document describes the cross-backend testing infrastructure for ZMLX.

## Overview

ZMLX now supports testing across multiple MLX backends:

- **Metal**: macOS with Apple Silicon GPUs (M1/M2/M3/M4)
- **CUDA**: Linux with NVIDIA GPUs
- **CPU**: Pure Python/CPU backend on any platform

The testing infrastructure ensures that:

1. Pure Python components work on all platforms
2. Metal-specific kernels run correctly on Apple Silicon
3. Numerical results are consistent across backends (within tolerance)

## Backend Detection

The `zmlx._compat` module provides backend detection utilities:

```python
from zmlx._compat import detect_backend, has_gpu_backend, is_metal_available

backend = detect_backend()
# Returns: "metal", "cuda", "cpu", or "unknown"

if has_gpu_backend():
    print("GPU acceleration available")

if is_metal_available():
    print("Running on Apple Silicon with Metal")
```

## Pytest Markers

The following pytest markers are available for test categorization:

| Marker | Description | Platforms Run |
|:-------|:------------|:--------------|
| `@pytest.mark.metal` | Requires Metal backend | macOS Apple Silicon only |
| `@pytest.mark.cuda` | Requires CUDA backend | Linux with CUDA only |
| `@pytest.mark.gpu` | Requires any GPU backend | macOS Metal or Linux CUDA |
| `@pytest.mark.cpu` | CPU-compatible (pure Python) | All platforms |
| `@pytest.mark.golden` | Uses golden values for cross-backend comparison | All platforms |

### Example Usage

```python
import pytest
import mlx.core as mx

# Pure Python test - runs everywhere
@pytest.mark.cpu
def test_pure_python():
    from zmlx.device_profile import get_device_profile
    profile = get_device_profile("M3", "Max")
    assert profile.gpu_cores == 40

# Metal-specific test
@pytest.mark.metal
def test_metal_kernel():
    from zmlx.kernels.transformer import swiglu
    x = mx.random.normal((4, 128))
    y = swiglu(x)  # Uses Metal kernel
    assert y.shape == (4, 64)

# GPU test (Metal or CUDA)
@pytest.mark.gpu
def test_gpu_operation():
    x = mx.random.normal((4, 64))
    y = mx.matmul(x, x.T)
    assert y.shape == (4, 4)
```

## Fixtures

The `conftest.py` provides several useful fixtures:

### Backend Fixtures

```python
def test_with_backend(backend, is_metal, has_gpu):
    """backend: 'metal', 'cuda', 'cpu', or 'none'
       is_metal: True if running on Metal
       has_gpu: True if GPU backend available
    """
    if has_gpu:
        # Run GPU-accelerated test
        pass
    else:
        # Skip or use CPU fallback
        pass
```

### Device Fixture

```python
def test_on_device(mx_device):
    """mx_device: mlx.core.gpu or mlx.core.cpu"""
    import mlx.core as mx
    x = mx.array([1.0, 2.0, 3.0], device=mx_device)
    # ... test with x
```

### Cross-Backend Comparison

```python
def test_cross_backend(assert_allclose_cross_backend):
    import mlx.core as mx
    
    x = mx.array([1.0, 2.0, 3.0])
    y = mx.exp(x)
    
    # Compares with adjusted tolerances based on backend
    assert_allclose_cross_backend(
        y, 
        [2.718, 7.389, 20.085],
        rtol=1e-4,
        atol=1e-4,
    )
```

## Golden Value Testing

Golden values provide reference outputs for cross-backend validation:

### Generating Golden Values

```bash
# On Metal (reference platform)
python tests/generate_golden_values.py
# Creates: tests/golden_values_metal.json
```

### Comparing Across Backends

```bash
# On target platform (e.g., Linux CPU)
python tests/compare_golden_values.py
# Compares against metal reference values
```

### In-Code Golden Values

```python
def test_with_golden_values(golden_registry):
    import mlx.core as mx
    
    # Run operation
    x = mx.array([1.0, 2.0, 3.0])
    y = mx.softmax(x)
    
    # Register for current backend
    golden_registry["register"]("softmax_basic", y.tolist(), "metal")
    
    # Later, compare against reference
    reference = golden_registry["get"]("softmax_basic", "metal")
    if reference:
        # Compare y against reference
        pass
```

## CI Configuration

The cross-backend CI runs tests on multiple platforms:

### GitHub Actions Matrix

```yaml
test-linux-cpu:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Install MLX
      run: pip install mlx
    - name: Test pure Python
      run: pytest -m "cpu or not (metal or cuda or gpu)"

test-macos-metal:
  runs-on: macos-14
  steps:
    - uses: actions/checkout@v4
    - name: Install with Metal
      run: pip install .
    - name: Test full suite
      run: pytest tests/
```

See `.github/workflows/cross-backend-ci.yml` for the complete configuration.

## Running Tests Locally

### Run all tests (requires Metal on macOS)

```bash
pytest tests/ -v
```

### Run only CPU-compatible tests

```bash
pytest tests/ -v -m "cpu or not (metal or cuda or gpu)"
```

### Run only Metal-specific tests

```bash
pytest tests/ -v -m metal
```

### Run tests excluding slow ones

```bash
pytest tests/ -v -m "not slow"
```

## Writing Cross-Backend Tests

### Best Practices

1. **Mark pure Python tests with `@pytest.mark.cpu`** so they run on Linux CI
2. **Use fixtures** for backend-specific behavior
3. **Use `assert_allclose_cross_backend`** for numerical comparisons
4. **Generate golden values** on Metal for cross-backend validation

### Example: Cross-Backend Kernel Test

```python
import pytest
import mlx.core as mx

pytestmark = [pytest.mark.cpu]

class TestMyKernel:
    """Tests that work on all backends."""
    
    def test_kernel_logic(self):
        """Pure Python logic - runs everywhere."""
        from zmlx.my_module import prepare_inputs
        x = mx.array([1.0, 2.0, 3.0])
        prepared = prepare_inputs(x)
        assert prepared.shape == x.shape
    
    @pytest.mark.gpu
    def test_kernel_execution(self, mx_device):
        """Requires GPU - runs on Metal or CUDA."""
        from zmlx.kernels.my_kernel import my_kernel
        x = mx.random.normal((4, 64), device=mx_device)
        y = my_kernel(x)
        assert y.shape == (4, 64)
    
    @pytest.mark.metal
    def test_metal_specific(self):
        """Metal-specific optimization."""
        from zmlx.kernels.my_kernel import optimized_metal_kernel
        # ... test Metal-specific variant
```

## Troubleshooting

### Tests Skipped on Linux

If tests are being skipped on Linux:

1. Check that tests have `@pytest.mark.cpu` marker
2. Verify MLX is installed: `pip install mlx`
3. Check backend detection: `python -c "from zmlx._compat import detect_backend; print(detect_backend())"`

### Golden Value Mismatches

If golden value comparisons fail:

1. Check tolerance settings (may need higher for CPU vs Metal)
2. Regenerate golden values on reference platform
3. Verify MLX version matches

### Metal Not Available in CI

GitHub Actions macOS runners may not expose Metal. Tests marked with `@pytest.mark.metal` will be skipped. This is expected behavior.

## Future Work

Planned improvements to cross-backend testing:

1. **CUDA CI**: Add NVIDIA GPU runners when available
2. **Property-based testing**: Use Hypothesis for broader test coverage
3. **Performance regression tests**: Track kernel performance across releases
4. **Memory leak detection**: Automated checking for Metal kernel leaks
