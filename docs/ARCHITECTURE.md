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
- Mid-level:
  - `zmlx/elementwise.py`: elementwise generators
  - `zmlx/autograd.py`: `unary_from_expr`, `binary_from_expr`
- High-level catalog:
  - `zmlx/kernels/*`: softmax, norms, RoPE, activations, fused ops

Design principles:
- compile once, call many
- correctness first (reference tests vs MLX ops)
- stable names + stable generated source for caching

## Frontend B (future): Zig via MLX-C

See `zig/README.md`.

The intended boundary:
- Zig calls into MLX via **MLX-C** (arrays, devices, streams)
- Zig reuses the **same kernel core ideas**:
  - generate stable MSL source for a kernel
  - call MLX’s Metal kernel facility and rely on caching

What’s missing for a real Zig path:
- a tiny C++ shim (if MLX-C doesn’t expose everything you need)
- Zig wrappers for:
  - tensor/array creation
  - value+grad transforms (if needed for training)
  - parameter trees / optimizers (optional)

## Contributing upstream to MLX

This repo is intentionally “bigger than what MLX should own”.
Good upstream candidates:
- docs/cookbook recipes (custom kernel + custom VJP)
- a tiny threadgroup heuristic helper
- minimal correctness tests around `metal_kernel`

See `docs/UPSTREAMING.md`.
