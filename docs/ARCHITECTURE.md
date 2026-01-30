# Architecture: one core + multiple frontends (plan)

This repo is intentionally **small** and pragmatic: it wraps MLX’s existing “escape hatches”
for custom kernels and custom autodiff.

MLX primitives we build on:

- `mx.fast.metal_kernel(...)`: compile a Metal kernel from a source string and call it like an op.
- `mx.custom_function`: attach `vjp/jvp/vmap` so transforms like `mx.grad` work.

## Why a “kernel core” exists (even in Python)

Metal kernels are ultimately defined by:
- the **source string** (MSL body + optional header)
- the **launch configuration** (grid/threadgroup)
- template specializations (e.g., `T=float32`)

MLX’s compilation/caching is keyed heavily off the source string + specialization.
So we get stable behavior by generating source in a predictable way and memoizing kernels.

## Today: the “core” is Python modules

- `zmlx/codegen.py`: small codegen helpers for common patterns
  - elementwise
  - rowwise map-reduce (used by softmax, norms)
- `zmlx/msl.py`: shared header snippets (sigmoid, silu, gelu_tanh)

This is the minimal version of the “Kernel IR → MSL codegen” idea.

## Frontend A: Python (`zmlx`)

User-facing layers:

- Low-level:
  - `zmlx/metal.py`: cached kernel wrapper
  - `zmlx/jit_compiler.py`: Python AST to Metal compiler
- Mid-level:
  - `zmlx/elementwise.py`: elementwise generators
  - `zmlx/autograd.py`: `unary_from_expr`, `binary_from_expr`
- High-level:
  - `zmlx/nn.py`: high-throughput layers (PagedAttention, MoE)
  - `zmlx/optimizers.py`: fused optimizers (AdamW)
- High-level catalog:
  - `zmlx/kernels/*`: softmax, norms, RoPE, activations, fused ops, optimizers, attention

Design principles:
- compile once, call many
- correctness first (reference tests vs MLX ops)
- stable names + stable generated source for caching

## Frontend B: Zig via C++ shim

See `zig/README.md` for build instructions and full details.

The Zig frontend replicates layers 1–2 of the Python stack:

- `zig/src/msl.zig` — `DEFAULT_HEADER` (byte-identical to `zmlx/msl.py`)
- `zig/src/codegen.zig` — all 5 codegen patterns (byte-identical output)
- `zig/src/metal.zig` — `MetalKernel` wrapper + `KernelCache` (SHA-256 keyed)
- `zig/shim/shim.cc` — minimal C ABI wrapping `mlx::core::fast::metal_kernel`

Architecture:
```
  Zig program
    │
    ├─ codegen.zig   → generates MSL source strings
    ├─ msl.zig       → shared Metal header (sigmoid, silu, gelu_tanh)
    ├─ metal.zig     → MetalKernel struct, KernelCache, Array wrapper
    │     │
    │     └─ @cImport("shim.h")
    │            │
    └─ shim.cc ──┘   → C++ ABI: zmlx_metal_kernel_create/call/destroy
         │                       zmlx_array_from_float32/destroy/eval
         └─ links libmlx (MLX C++ library)
```

**Why a C++ shim instead of MLX-C?**  MLX-C does not yet expose
`mx.fast.metal_kernel`.  The shim (`shim.h` / `shim.cc`) fills this gap
with a minimal C ABI.  When MLX-C gains metal_kernel support, the shim
can be dropped and Zig can call MLX-C directly.

**What's implemented:**
- Kernel creation, launch, and destruction
- float32 / int32 array creation from host data
- Array evaluation and data readback
- Codegen for elementwise (unary, binary) and rowwise (reduction,
  parallel reduction, two-pass map-reduce) patterns
- In-process kernel cache keyed on SHA-256 of source + header

**What's not yet implemented (future work):**
- value+grad transforms (custom VJP/JVP in Zig)
- Parameter trees / optimizers
- Full dtype coverage beyond float32/int32
- Catalog kernels (softmax, layernorm, etc.) in Zig

## Contributing upstream to MLX

This repo is intentionally “bigger than what MLX should own”.
Good upstream candidates:
- docs/cookbook recipes (custom kernel + custom VJP)
- a tiny threadgroup heuristic helper
- minimal correctness tests around `metal_kernel`

See `docs/UPSTREAMING.md`.
