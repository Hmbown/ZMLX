# AGENTS.md — ZMLX

**Triton for Apple Silicon** — ergonomic custom Metal kernel authoring for MLX.

## Project Status (v0.4.1)

- 70+ kernels in the catalog
- High-level API: `elementwise()`, `reduce()`, `map_reduce()`, `@jit`
- Model patching with +33% decode (32B dense), +36% decode (30B MoE)
- Testing, benchmarking, profiling utilities
- Training pipeline with `zmlx train` CLI
- Fused MoE gating, AdamW optimizer, PagedAttention

## How to use this file

If you're using AI agents to continue building, split work by role. Each agent should:
1) Read `README.md` + `docs/ARCHITECTURE.md`
2) Scan `src/zmlx/*` and `src/zmlx/kernels/*`
3) Pick tasks from the backlog below
4) Implement, add tests, update docs/examples

## Suggested agent roles

### Agent 1 — Kernel Authoring UX
- Expand `codegen.py` patterns (broadcasting, 2D launches, multiple outputs)
- Improve error messages (shape/dtype mismatches, non-contiguous inputs)
- Extend `@jit` decorator to support more Python constructs

### Agent 2 — Autograd / Transformations
- Add `jvp` and `vmap` patterns for elementwise ops
- Support higher-order derivatives (if MLX allows nesting `custom_function`)
- Symbolic/IR derivative for restricted DSL

### Agent 3 — Kernel Library
- Flash Attention with shared memory (threadgroup) for small tiles
- Fused dequant+compute (int4 dequant fused with matmul or activation)
- More MoE patterns (top-k > 2, load balancing)

### Agent 4 — Perf / Autotuning
- Per-device autotune profiles (M1/M2/M3/M4 families)
- Regression tracking with JSON output
- Memory bandwidth analysis tools

### Agent 5 — Zig / MLX-C Frontend (experimental)
- Audit MLX-C for metal-kernel and grad-transform coverage
- Zig wrappers + build integration in `zig/`
- Keep kernel codegen portable between Python and Zig

## Completed Milestones

All of the following have been implemented and shipped:

- Renamed from `mlx-kernelkit` to **ZMLX**
- 70+ kernel catalog: activations, norms, RoPE, attention, transformer fusions,
  MoE, quantization (int8/int4/FP8/NF4), bit ops, image, indexing, loss, scan
- High-level API: `elementwise()`, `reduce()`, `map_reduce()`, `@jit`
- Model patching system with `patch()`, `smart_patch()`, presets
- Fused MoE gating patch (+51% prompt, +36% decode on Qwen3-30B-A3B)
- Fused residual+RMSNorm (+33% decode on Qwen3-32B)
- Testing (`zmlx.testing`), benchmarking (`zmlx.bench`), profiling (`zmlx.profile`)
- Training pipeline (`zmlx train` CLI) with LoRA, callbacks, YAML config
- Fused AdamW optimizer, PagedAttention, MoE dispatch/combine
- Autotuning with persistent cache
- GitHub Actions CI + PyPI trusted publishing
- Documentation: QUICKSTART, COOKBOOK, KERNELS catalog, ARCHITECTURE

## Open Backlog

### High Priority
- [ ] Flash Attention with shared memory tiles (32x32 or 64x64)
- [ ] Fused dequant+compute (int4 dequant fused into activation)
- [ ] Continuous batching / serving integration
- [ ] Per-device autotune profiles

### Medium Priority
- [ ] `jvp` support for elementwise ops
- [ ] Speculative decoding acceleration
- [ ] More architecture support (Gemma, Phi, Mistral MoE)
- [ ] Upstream contributions to MLX (see `docs/UPSTREAMING.md`)

### Low Priority
- [ ] Zig frontend completion (blocked on MLX-C `metal_kernel` support)
- [ ] Higher-order derivatives
- [ ] WGSL backend (for WebGPU, long-term)
