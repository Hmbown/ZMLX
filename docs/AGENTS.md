# AGENTS.md — ZMLX

This repo aims to make MLX custom Metal kernels feel closer to “Numba/Triton”:
- ergonomic kernel authoring
- stable codegen + caching
- launch heuristics + autotuning
- autograd (custom VJP/JVP) patterns
- a growing catalog of reusable kernels (softmax/norms/rope/fused ops)
- (future) multi-language frontends (Zig via MLX-C)

## How to use this file
If you’re using AI agents to continue building, split work by role. Each agent should:
1) read `README.md` + `docs/ARCHITECTURE.md`
2) scan `src/zmlx/*` and `src/zmlx/kernels/*`
3) pick tasks from “Backlog”
4) implement, add tests, update docs/examples

## Suggested agent roles (3–5)

### Agent 1 — Kernel Authoring UX
Focus: API ergonomics and codegen helpers.
- Expand `codegen.py` patterns (broadcasting, 2D launches, multiple outputs)
- Add a `@kk.kernel` decorator that accepts a restricted DSL or raw MSL
- Improve error messages (shape/dtype mismatches, non-contiguous inputs)

### Agent 2 — Autograd / Transformations
Focus: making custom kernels composable with MLX transforms.
- Generalize beyond unary/binary: N-ary ops, multiple outputs
- Add `jvp` and `vmap` patterns for elementwise ops
- Add a small “auto-grad for restricted DSL” (symbolic/IR derivative)

### Agent 3 — Kernel Library (Transformers)
Focus: expand “as many kernels as possible” while keeping tests.
- More fused ops:
  - bias + activation (gelu/silu/relu)
  - fused residual patterns
  - fused dropout (if RNG strategy is acceptable)
- More reductions:
  - logsumexp
  - attention-softmax fragments
- Better norms:
  - fused RMSNorm + residual
  - (optional) backward kernels

### Agent 4 — Perf / Autotuning
Focus: performance measurement and auto-configuration.
- Robust timing utilities that respect MLX laziness (`mx.eval`)
- Threadgroup autotune search strategies per device family
- Microbench harness (JSON output, regression tracking)
- Profiling helpers: dump generated MSL + launch params

### Agent 5 — Zig / MLX-C Frontend (experimental)
Focus: making the “second frontend” real.
- Audit MLX-C for metal-kernel and grad-transform coverage
- If needed: implement a tiny C++ shim that exposes missing calls with a C ABI
- Zig wrappers + build integration in `zig/`
- Keep kernel codegen portable between Python and Zig

## RECENT MILESTONES (Phase 1-3)
- [x] Renamed project from `mlx-kernelkit` to **ZMLX** (Python package `zmlx`)
- [x] Fixed `mx.custom_function` VJP signature for compatibility with recent MLX
- [x] Added robust kernel name sanitization (handling dots/hyphens from floating point params)
- [x] Implemented **Transformer essentials**:
    - Fused SwiGLU / GeGLU (forward + VJP)
    - Fused RMSNorm + Residual add
    - Dropout (in-kernel LCG RNG)
    - Fused Bias + SwiGLU / GeGLU
- [x] Implemented **Attention / Reductions**:
    - `logsumexp_lastdim`, `masked_softmax`, `scale_mask_softmax`
    - `sum_lastdim`, `mean_lastdim`, `max_lastdim`, `var_lastdim`, `std_lastdim`
    - `apply_rope_interleaved`
- [x] Enabled differentiation for `rmsnorm` and `softmax_lastdim` via `custom_function`
- [x] Added `zmlx.elementwise.map` helper for easy N-ary elementwise kernels
- [x] Added `benchmarks/microbench.py` with significant speedups (e.g. 7x faster Dropout)
- [x] Added `zmlx.registry` for kernel cache management
- [x] Added `argmax_lastdim`, `cumsum_lastdim` (prefix sum)
- [x] Added `zmlx.rowwise.map_reduce` high-level helper
- [x] Added kernel stats tracking (compile time, run count)
- [x] Added `top2_gating_softmax` for MoE
- [x] Added `pack_bits` / `unpack_bits` kernels
- [x] Added `examples/transformer_block.py` integration demo
- [x] Added GitHub Actions CI workflow
- [x] Added `topk_lastdim` optimized for small K
- [x] Added `dequantize_int4` kernel
- [x] Implemented `attention_tile_proto` using shared memory (threadgroup)
- [x] Generated `docs/KERNELS.md` catalog
- [x] Added autotune caching and 2D/3D candidate support
- [x] Enabled GPU run-time measurement in `MetalKernel` (verbose mode)
- [x] Added `resize_bilinear` and `depthwise_conv_3x3` image kernels
- [x] Implemented `log_softmax_lastdim` and extra activations (softplus, mish, elu)
- [x] Expanded catalog to 50+ kernels
- [x] Implemented differentiable `cumsum_lastdim` (with gradient)
- [x] Added `fused_linear_bias_gelu` and `fused_linear_rmsnorm`
- [x] Reached 60+ kernels total
- [x] Added `softmax_cross_entropy` fused loss
- [x] Implemented `rms_norm_dropout` and `layer_norm_dropout`
- [x] Enabled differentiation for `bias_swiglu` and `bias_geglu`
- [x] Reached 70+ kernels total
- [x] Implemented `layernorm_residual` and `fused_add_layernorm`
- [x] Added Kahan summation support for high-precision reductions
- [x] Generalized autograd with `nary_from_expr` helper

## NEXT 20 TASKS (Phase 4-5)
1.  [x] **Flash Attention Building Blocks**:
    - [x] Implement `scale_mask_softmax` (fused input scaling + causal masking + softmax)
    - [ ] Explore shared memory (threadgroup) for small attention tiles (e.g. 16x16 or 32x32)
2.  [ ] **Fused Linear Patterns**:
    - [ ] Implement fused `Linear + Bias + Act` (limited to cases where it wins over `mx.matmul`)
    - [ ] Implement fused `Linear + RMSNorm`
3.  [x] **Enhanced DSL / Codegen**:
    - [x] Add `zmlx.elementwise.map` helper that takes a C++ expression and arity
    - [ ] Add `zmlx.rowwise.map_reduce` high-level helper
4.  [x] **Performance Benchmarking**:
    - [x] Create `benchmarks/microbench.py` using `time.perf_counter_ns`
    - [x] Compare `zmlx.kernels.norms.rmsnorm` vs `mx.fast.rms_norm`
    - [x] Compare `zmlx.kernels.transformer.swiglu` vs pure MLX `silu(x)*y`
5.  [x] **Kernel Registry**:
    - [x] Implement `zmlx.registry.list_kernels()`
    - [x] Implement `zmlx.registry.clear_cache()`
    - [ ] Add compile-time vs run-time stats tracking in `MetalKernel`
6.  [x] **Advanced Reductions**:
    - [x] `var_lastdim` and `std_lastdim`
    - [ ] `argmax_lastdim` (returning `uint32`)
7.  [ ] **Multi-language**:
    - [ ] Flesh out `zig/src/main.zig` to use `MLX-C`
    - [ ] Prototype sharing the `MSL` strings generated in Python with the Zig frontend
8.  [ ] **Autograd expansion**:
    - [ ] Support `jvp` (Jacobian-Vector Product) for unary/binary helpers
    - [ ] Support higher-order derivatives (if MLX allows nesting `custom_function`)
9.  [ ] **Quantization support**:
    - [ ] Simple dequantize kernels (int8/int4 to float32)
    - [ ] Fused dequantize + elementwise op
10. [ ] **CI / CD**:
    - [ ] Add `.github/workflows/test.yml` running `pytest` on macOS
    - [ ] Add `ruff` and `mypy` checks to CI
11. [ ] **Documentation**:
    - [ ] Add `docs/KERNELS.md` with detailed list of all kernels and their formulas
    - [ ] Add `docs/BENCHMARKS.md` with latest numbers from M4
12. [ ] **Bit manipulation kernels**:
    - [ ] `pack_bits`, `unpack_bits`
13. [ ] **Scanning / Prefix Sum**:
    - [ ] `rowwise_cumsum`
14. [ ] **Image processing**:
    - [ ] `resize_bilinear` or `conv2d_simple` (fused with activation)
15. [ ] **Top-K**:
    - [ ] `top1` (max) already done, try `topk` for small k
16. [ ] **Grouped Query Attention helpers**:
    - [ ] Fused rope for GQA where heads are grouped
17. [ ] **MoE (Mixture of Experts) kernels**:
    - [ ] `top2_gating_softmax`
18. [ ] **Sparse kernels**:
    - [ ] Simple `gather` / `scatter` fused with arithmetic
19. [ ] **Numerical Stability Audit**:
    - [ ] Check all kernels for overflow in `float16`
    - [ ] Add `Kahan summation` option for large reductions
20. [ ] **Packaging**:
    - [ ] Ensure versioning follows Semantic Versioning
    - [ ] Prepare for 0.3.0 release


## Definition of done (v0.3 goal)
- 15+ kernels in the catalog covering: activations, fused ops, norms, reductions
- Gradients validated vs MLX reference ops (documented tolerances)
- Autotune chooses a good threadgroup quickly for common shapes
- Docs: “write kernels”, “debug kernels”, “how to upstream”
