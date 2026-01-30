# ZMLX Zig frontend

Zig frontend for ZMLX — define, cache, and launch custom Metal kernels on
Apple silicon using MLX as the runtime.

## Architecture

```
zig/
├── build.zig              # Build configuration (shim + demo + tests)
├── shim/
│   ├── shim.h             # C ABI for MLX metal_kernel + array helpers
│   └── shim.cc            # C++ implementation (links libmlx)
└── src/
    ├── main.zig           # Demo: elementwise add kernel
    ├── codegen.zig        # MSL code generation (port of codegen.py)
    ├── msl.zig            # Shared Metal header (port of msl.py)
    ├── metal.zig          # MetalKernel wrapper + KernelCache + Array
    ├── elementwise.zig    # Convenience API: unary(), binary()
    ├── rowwise.zig        # Convenience API: mapReduce(), parallelReduce()
    ├── kernels.zig        # Kernel catalog barrel export
    ├── metadata.zig       # JSON export for Python interop
    └── kernels/
        ├── activations.zig  # silu, gelu, relu, sigmoid, exp, tanh, softplus, mish
        ├── softmax.zig      # softmaxLastDim (two-pass map-reduce)
        └── norms.zig        # rmsNormNoWeight
```

**Layers:**

| Layer | Zig module | Python equivalent |
|-------|-----------|------------------|
| MSL header | `msl.zig` | `zmlx/msl.py` |
| Code generation | `codegen.zig` | `zmlx/codegen.py` |
| Kernel wrapper + cache | `metal.zig` | `zmlx/metal.py` + `zmlx/cache.py` |
| Convenience API | `elementwise.zig`, `rowwise.zig` | `zmlx/elementwise.py`, `zmlx/rowwise.py` |
| Kernel catalog | `kernels/*.zig` | `zmlx/kernels/*.py` |
| Metadata export | `metadata.zig` | (new — enables Zig→Python interop) |
| C++ shim | `shim/shim.cc` | (calls MLX C++ directly) |

The codegen produces **byte-identical** MSL source strings to the Python
version, so both frontends share MLX's source-based compilation cache.

## Prerequisites

- **macOS arm64** (Apple Silicon — Metal GPU required)
- **Zig 0.14+** (current stable)
- **MLX C++ library** installed (headers + libmlx)

### Installing MLX

Homebrew (recommended):
```bash
brew install mlx
```

Or build from source:
```bash
git clone https://github.com/ml-explore/mlx.git
cd mlx && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
sudo make install
```

## Build

```bash
cd zig

# Default (assumes Homebrew MLX at /opt/homebrew)
zig build

# Custom MLX paths
zig build -Dmlx-include=/path/to/mlx/include -Dmlx-lib=/path/to/mlx/lib
```

## Run the demo

```bash
zig build run
```

Expected output:
```
=== ZMLX Zig frontend demo ===

a = [1, 2, 3, 4]
b = [10, 20, 30, 40]

Generated MSL source (132 bytes):
        uint elem = thread_position_in_grid.x;
        T a = lhs[elem];
        T b = rhs[elem];
        T x = a;
        out[elem] = a + b;

result = [11.0, 22.0, 33.0, 44.0]

Expected: [11.0, 22.0, 33.0, 44.0]
```

## Run tests

```bash
# All tests (requires MLX installed + Apple Silicon GPU)
zig build test

# Codegen-only tests (no MLX needed, pure Zig string generation)
zig build test-codegen

# Kernel catalog tests only (GPU required)
zig build test-catalog
```

## Convenience API

The convenience layer reduces kernel creation from ~30 lines of boilerplate
to a single function call:

### Elementwise operations

```zig
const elementwise = @import("elementwise.zig");

// Create a unary kernel from a C expression
const square = try elementwise.unary(allocator, "x * x", "square", .float32, .{});
const result = try square.call(input_array);

// Create a binary kernel
const add = try elementwise.binary(allocator, "a + b", "add", .float32, .{});
const sum = try add.call(a, b);
```

### Rowwise operations

```zig
const rowwise = @import("rowwise.zig");

// Two-pass map-reduce (e.g. softmax)
const op = try rowwise.mapReduce(allocator, "my_kernel", d, tg,
    "-INFINITY", "max(acc1, x)", "max(a, b)",  // pass 1
    "0.0f", "acc2 + exp(x - s1)", "a + b",     // pass 2
    "exp(x - s1) / s2",                         // write
    .float32, .{});
const out = try op.call(x, rows);
```

## Kernel catalog

Pre-built kernels matching the Python catalog:

### Activations

```zig
const act = @import("kernels/activations.zig");
const silu_op = try act.silu(allocator, .float32);
const result = try silu_op.call(input);
```

Available: `silu`, `geluTanh`, `relu`, `sigmoid`, `exp`, `tanh`, `softplus`, `mish`

### Softmax

```zig
const sm = @import("kernels/softmax.zig");
const softmax_op = try sm.softmaxLastDim(allocator, d, .float32, .{ .threadgroup = 256 });
const probs = try softmax_op.call(logits, num_rows);
```

### Norms

```zig
const norms = @import("kernels/norms.zig");
const rms_op = try norms.rmsNormNoWeight(allocator, d, 1e-5, .float32, .{});
const normed = try rms_op.call(x, num_rows);
```

## Metadata export

Export kernel specs to JSON for Python interop:

```zig
const metadata = @import("metadata.zig");
const json = try metadata.specToJson(allocator, kernel_spec);
// Write to file, send to Python frontend, etc.
```

## Codegen functions

All five codegen patterns from `src/zmlx/codegen.py` are ported:

| Zig function | Python function | Pattern |
|-------------|----------------|---------|
| `elementwiseUnarySource` | `elementwise_unary_source` | 1 input → 1 output per element |
| `elementwiseBinarySource` | `elementwise_binary_source` | 2 inputs → 1 output per element |
| `rowwiseReductionSource` | `rowwise_reduction_source` | Simple row reduction (1 thread/row) |
| `rowwiseParallelReductionSource` | `rowwise_parallel_reduction_source` | Parallel row reduction with threadgroup |
| `rowwiseMapReduceSource` | `rowwise_mapreduce_source` | Two-pass map-reduce (softmax, layernorm) |

## Design notes

- **No MLX-C dependency**: MLX-C does not yet expose `mx.fast.metal_kernel`,
  so we use a minimal C++ shim (`shim/shim.cc`) that links against the MLX
  C++ library directly. When MLX-C gains metal_kernel support, the shim can
  be replaced.

- **Cache sharing**: Generated MSL strings are byte-identical to Python's
  output, so MLX's internal source-based Metal compilation cache is shared
  between both frontends running in the same process.

- **Kernel cache**: `metal.zig` includes a `KernelCache` keyed on SHA-256
  of the source + header, matching the Python `KernelCacheKey` pattern.

- **Forward-only catalog**: Catalog kernels are forward-only (no VJP/JVP).
  Gradient support requires MLX-C to expose `mx.custom_function`, which is
  planned but not yet available.

- **Threadgroup validation**: All rowwise kernels validate that threadgroup
  sizes are powers of 2 in (0, 1024], matching the Python frontend.
