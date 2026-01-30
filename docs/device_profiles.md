# Device Autotune Profiles

This document describes the per-device autotune profile system introduced in ZMLX.

## Overview

The per-device autotune profile system provides hardware-specific tuning configurations for different Apple Silicon chips. This allows ZMLX to automatically select optimal threadgroup sizes and other kernel parameters based on the specific chip variant (M1/M2/M3/M4 × base/Pro/Max/Ultra).

## Supported Devices

ZMLX includes tuning profiles for all 16 Apple Silicon variants:

| Chip | GPU Cores | Memory Bandwidth | Default Threadgroup |
|:-----|:---------:|:----------------:|:-------------------:|
| M1 | 8 | 68 GB/s | 128 |
| M1 Pro | 16 | 200 GB/s | 256 |
| M1 Max | 32 | 400 GB/s | 256 |
| M1 Ultra | 64 | 800 GB/s | 256 |
| M2 | 10 | 100 GB/s | 128 |
| M2 Pro | 19 | 200 GB/s | 256 |
| M2 Max | 38 | 400 GB/s | 256 |
| M2 Ultra | 76 | 800 GB/s | 256 |
| M3 | 10 | 100 GB/s | 128 |
| M3 Pro | 18 | 150 GB/s | 256 |
| M3 Max | 40 | 400 GB/s | 256 |
| M3 Ultra | 80 | 800 GB/s | 256 |
| M4 | 10 | 120 GB/s | 128 |
| M4 Pro | 20 | 273 GB/s | 256 |
| M4 Max | 40 | 546 GB/s | 256 |
| M4 Ultra | 80 | 1000 GB/s | 256 |

## API Usage

### Getting Device Profiles

```python
from zmlx.device_profile import get_device_profile, get_current_device_profile

# Get profile for a specific chip
profile = get_device_profile("M3", "Max")
print(profile.gpu_cores)  # 40
print(profile.default_threadgroup)  # 256

# Get profile for current device
current = get_current_device_profile()
print(current.full_name)  # e.g., "Apple M3 Max"
```

### Using the @autotune Decorator

The `@autotune` decorator automatically uses device-optimized threadgroup candidates:

```python
import zmlx

@zmlx.autotune(warmup=3, iters=10, device_aware=True)
def my_kernel_op(x, y, threadgroup=(128, 1, 1)):
    kernel = zmlx.metal.kernel(...)
    return kernel(x, y, threadgroup=threadgroup)

# Automatically uses optimal threadgroup for current device
result = my_kernel_op(x, y)
```

### Custom Threadgroup Candidates

You can also get device-optimized candidates for custom kernels:

```python
from zmlx.device_profile import (
    get_current_device_profile,
    get_threadgroup_candidates_for_shape,
)

profile = get_current_device_profile()
candidates = get_threadgroup_candidates_for_shape(
    profile,
    dim_size=1024,
    kernel_type="reduction"  # or "general", "simd"
)
# Returns: (128, 256, 512) for high-end chips
```

## Cache Format (v3)

The autotune cache uses a v3 schema that includes device metadata:

```json
{
  "schema_version": "3.0",
  "devices": {
    "M3_Max_0.30.0": {
      "metadata": {
        "family": "M3",
        "variant": "Max",
        "gpu_cores": 40,
        "memory_bandwidth_gbps": 400,
        "default_threadgroup": 256
      },
      "entries": {
        // kernel configurations...
      },
      "saved_at": "2025-01-30T12:00:00"
    }
  }
}
```

The cache is stored at `~/.cache/zmlx/autotune_v3.json` by default.

## Migration from v2

The new cache format is automatically backward compatible. When loading a v2 cache, ZMLX will:

1. Recognize the old format (no `schema_version` field)
2. Load entries using the existing device key
3. Save new entries in v3 format

To force regeneration of cache entries with device-aware tuning:

```python
from zmlx.autotune import GLOBAL_AUTOTUNE_CACHE, save_autotune_cache

# Clear the cache
GLOBAL_AUTOTUNE_CACHE.clear()

# Run your kernels (they will be re-autotuned)
# ...

# Save new cache
save_autotune_cache()
```

## Performance Impact

Using device-specific profiles typically improves autotuning performance:

- **Search space reduction**: Fewer candidates to evaluate per kernel
- **Better defaults**: Initial configurations closer to optimal
- **Faster convergence**: Device-aware warmups and iteration counts

Expected speedups for autotuning:

| Device | Generic Candidates | Device-Aware | Speedup |
|:-------|:------------------:|:------------:|:-------:|
| M1 | 6 candidates | 3 candidates | ~1.5x |
| M3 Max | 6 candidates | 3 candidates | ~1.5x |
| M4 Max | 6 candidates | 3 candidates | ~1.5x |

## Future Work

Planned enhancements to the device profile system:

1. **Memory bandwidth-aware tuning**: Adjust threadgroups based on memory-bound vs compute-bound kernels
2. **Ray tracing support**: Leverage M3+ ray tracing hardware where applicable
3. **Thermal-aware tuning**: Adjust for sustained vs burst workloads
4. **Profile learning**: Collect runtime performance data to refine profiles
