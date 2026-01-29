# Zig scaffold (experimental)

This folder is a **starter scaffold** for a Zig frontend that targets MLX as the runtime.

Goal:
- generate Metal kernel source (MSL) in Zig (eventually using the same “kernel core” patterns)
- call into MLX via **MLX-C** (C ABI)
- later: build a small training API (params/optims) on top of MLX transforms

## What exists here today

- `build.zig` / `src/main.zig`: placeholder “hello” project
- `docs/` (in repo root): describes the intended boundary

## Practical next steps (real work)

1) Add MLX-C as a dependency
   - clone `ml-explore/mlx-c`
   - build it for your macOS target
   - link it into Zig via `build.zig`

2) Decide which MLX-C calls you need first
   - arrays: create/from host, reshape, dtype
   - metal kernel path:
     - either: expose `metal_kernel` through MLX-C
     - or: write a tiny C++ shim that exposes it with a C ABI

3) Keep the “kernel core” portable
   - in the short term: hardcode MSL strings
   - in the medium term: port `zmlx/codegen.py` patterns to Zig

## Recommended “thin shim” approach

If MLX-C doesn’t expose the Metal kernel facility you want:

- write a `shim.cc` that includes MLX headers and exports a C ABI:
  - `zmlx_metal_kernel_create(...)`
  - `zmlx_metal_kernel_call(...)`

Zig then treats it like any other C library.

## Why this is useful

If Zig can call the same kernel core, you get:
- deterministic source generation and caching behavior
- a path to “MLX as backend, Zig as frontend language”
