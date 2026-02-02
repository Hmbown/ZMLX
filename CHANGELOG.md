# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.13] - 2026-02-02

### Added

- **GLM model detection**: `_is_glm_moe_block` in `moe_mlp` pattern, with opt-in fused SwiGLU via `ZMLX_GLM_FUSED_SWIGLU` env var.
- **Experimental MoE stream pool**: `ZMLX_MOE_STREAMS=N` env var for multi-stream expert dispatch.
- **GLM-4.7-Flash-4bit validation**: 0.955x decode, FAIL fidelity (diverges at token 1). Added to auto-excludes and "Tested (no gain)" table.

### Fixed

- **Numerically stable sigmoid**: `kk_sigmoid` in `msl.py` now uses `abs+branch` to avoid overflow on large negative inputs.
- **SwiGLU native dtype**: forward and backward kernels use native dtype with `kk_sigmoid` helper instead of always casting to float32.

### Changed

- **GLM auto-excluded**: `moe_mlp` and `swiglu_mlp` excluded for GLM models (token fidelity failure). Override with explicit `patterns=[...]`.
- **README**: tighter install section, docs as table, prose polish.
- **Gitignore**: removed `AGENTS.md`, `UPSTREAM_PLAN.md`, `IMPLEMENTATION_SUMMARY.md` from tracking.

## [0.7.12] - 2026-02-01

### Added

- **ReLU2 kernels**: `relu2` and `relu2_grad` with catalog tests.
- **Docs**: `docs/BENCHMARKS.md` methodology + repro capsules, `docs/EXPERIMENTAL_MLX.md` for optional MLX fork work.
- **Stable MoE coverage**: Qwen3-30B-A3B and GPT-OSS-20B listed as token-identical on stock MLX.

### Fixed

- **MoE gating detection**: cache GPT-OSS/Qwen3 model detection before class replacement so `_gating` selects the correct path.
- **`moe_combine_exact` bf16 rounding**: explicit rounding after multiply/add to match MLX bf16 accumulation semantics.

### Changed

- **GPT-OSS combine routing**: float32 gating weights now use `moe_combine_fp32` to preserve MLX promotion behavior.
- **Auto-excludes**: Qwen3 excludes only `swiglu_mlp` + `residual_norm`; GPT-OSS excludes only `residual_norm`.
- **README**: simplified install (`zmlx[train]`), removed custom MLX details from main docs, refreshed stable model table.

## [0.6.3] - 2026-01-30

### Added

- **SIMD-group MoE gating kernels**: top-k gating now uses `simd_max` / `simd_sum` for D ≤ 32 to
  eliminate threadgroup barriers on common 32-expert MoE setups.
- **Bias-aware fused gating**: expert-bias + `norm_topk_prob` paths are fused when possible.
- **LFM2 benchmarks (M1 Pro, 16 GB)**: updated results with SIMD gating.

## [0.6.2] - 2026-01-30

### Fixed

- **MoE pattern uses model's actual `num_experts_per_tok`**: previously
  hardcoded top-2 regardless of model config, causing incorrect routing
  on models with top-4/6/8 (LFM2, Qwen3-MoE, GPT-OSS). Now reads
  `top_k`/`num_experts_per_tok` dynamically from the module.
- **MoE gating preserves model logic exactly**: pattern now replicates
  each model's original gating sequence (softmax ordering, `expert_bias`,
  `norm_topk_prob`) and only fuses the combine step. Output is bit-for-bit
  identical to unpatched (max logit diff 2.6e-6).

### Added

- **`topk_gating_softmax(x, k)`** kernel: dispatches to fused Metal kernel
  for k=2, standard MLX ops for other k values.
- **`router` attribute support**: MoE pattern now matches modules with
  `router` (GPT-OSS) in addition to `gate` (Qwen3, Mixtral, LFM2).
- **LFM2-8B-A1B benchmark**: validated on Liquid AI's MoE model (32 experts,
  top-4 routing, 8-bit). Correct routing with dynamic expert selection.
- **Missing `moe_mlp` import** in `_registry.py` `_ensure_loaded()`.

## [0.6.1] - 2026-01-30

### Fixed

- **Benchmark default now matches `patch()` default**: `inference_benchmark.py` now uses
  `FUSED_ACTIVATIONS` by default (matching `patch(model)` behavior since v0.5.0). Previously
  the benchmark defaulted to `ALL_PATTERNS`, which included norm/softmax kernels that cause
  3–5% decode regression — giving misleadingly low numbers. Use `--all-patterns` to opt in.

### Changed

- **README**: reframed top-level messaging to lead with MoE model results table. Added
  Qwen3-30B-A3B (base) benchmark (+61% prompt / +37% decode). Noted GLM-4.7 support is
  in progress.

## [0.6.0] - 2026-01-30

### Added

- **`mode` parameter for `patch()`**: `patch(model, mode="inference")` (default) selects
  `FUSED_ACTIVATIONS`; `patch(model, mode="training")` selects `TRAINING_RECOMMENDED`.
  Explicit `patterns=` overrides `mode` when both are given.
- **GLM-4.7-Flash benchmark**: validated on `mlx-community/GLM-4.7-Flash-4bit` (MoE, 30B-A3B).
  Neutral result (1.01x/0.99x) — GLM's `@mx.compile` gating is already optimized.

### Fixed

- **MoE pattern compatibility**: `moe_mlp` now handles models where `gate()` returns
  `(indices, scores)` tuple (GLM-4, DeepSeek-V3) instead of raw logits. Previously
  would crash on these models. Also handles `shared_experts` (additive dense MLP).

### Changed

- **`ALL_PATTERNS` docstring**: now explicitly warns that it causes 3–5% decode regression
  on ALL tested models (dense 8B/32B and MoE 30B), not just "smaller or MoE models".
- **Module docstring**: updated with validated benchmark findings — MoE routing fusion is
  the killer feature (1.3–1.6x), dense models are bandwidth-bound and neutral, norm/softmax
  kernels are slower than MLX built-ins for inference.
- **README**: reframed positioning around MoE speedup as the primary value proposition.
  Added "Where ZMLX won't help" section for honest guidance. Updated preset examples
  to use `mode` parameter.
- **Project description**: updated to highlight MoE inference speedup.

### Removed

- **Bogus Qwen3-32B benchmark**: deleted `benchmarks/results/qwen32b_results.json` which
  contained cold-cache artifact results (baseline ran without shader warmup, making it
  artificially slow). Corrected CHANGELOG v0.4.0 entry that cited the invalid 1.33x number.

## [0.5.0] - 2026-01-30

### Changed

- **BREAKING**: `patch(model)` default changed from all patterns to `FUSED_ACTIVATIONS`
  (SwiGLU/GeGLU/MoE fusions only). This eliminates the decode regression reported
  on MoE models (e.g. Qwen3-30B-A3B-Instruct: 115→111 tok/s with all patterns).
  Use `patch(model, patterns=ALL_PATTERNS)` to opt in to all 7 patterns.

### Added

- `ALL_PATTERNS` preset constant listing all 7 patterns for explicit opt-in.
- `qwen3-30b-a3b-instruct` model variant in inference benchmarks.

## [0.4.2] - 2026-01-30

### Fixed

- Resolve 3 mypy errors: `no-any-return` in callbacks and transformer, missing `autotune_threadgroup` function
- Add `autotune_threadgroup()` convenience wrapper and `AutotuneResult.best_threadgroup` property
- Fix all ruff lint errors (unused imports, import sorting, unused variables in optimizers)
- Fix 14 repo audit issues: broken doc references, stale claims, inconsistent messaging

## [0.4.1] - 2026-01-30

### Added

- **MoE patch pattern** (`zmlx.patch.patterns.moe_mlp`): fused
  `top2_gating_softmax` + `moe_combine` for Mixture of Experts models.
  +51% prompt / +36% decode on Qwen3-30B-A3B-4bit. Included in
  `FUSED_ACTIVATIONS` preset.
- **Multi-dimensional MoE kernel support**: `moe.py` kernels now handle
  batched (B, N, D) shapes used during prefill.
- **Qwen3-30B-A3B and Qwen3-8B benchmarks** in README, completing the
  scaling story across dense and MoE architectures.

### Fixed

- **SwiGLU patch MoE compatibility**: patch now correctly skips MoE
  `switch_mlp` modules that take routing indices as extra arguments.

## [0.4.0] - 2026-01-30

### Added

- **High-level API** (`zmlx.api`): `elementwise()`, `reduce()`, `map_reduce()`
  — one-line kernel authoring with automatic gradient support.
- **JIT compiler** (`zmlx.api.jit`): `@jit` decorator that compiles Python
  scalar expressions directly to Metal kernels.
- **Testing utilities** (`zmlx.testing`): `assert_matches()` and
  `assert_gradient_matches()` for verifying custom kernels against reference
  implementations across shapes and dtypes.
- **Benchmarking utilities** (`zmlx.bench`): `compare()` for side-by-side
  timing of multiple implementations with formatted tables.
- **Profiling** (`zmlx.profile`): `time_kernel()`, `memory_usage()`,
  `dump_msl()`, and `kernel_stats()` for kernel introspection.
- **Training pipeline** (`zmlx.train`): `zmlx train` CLI command for LoRA
  fine-tuning with ZMLX-patched models, YAML config support, gradient
  checkpointing, and training callbacks.
- **Smart patching** (`zmlx.patch.smart_patch`): auto-benchmarks each candidate
  pattern against the model's forward pass and keeps only patterns that help.
- **Neural network modules** (`zmlx.nn`): `PagedAttention` for high-throughput
  serving, `MoELayer` for mixture-of-experts dispatch/combine.
- **Fused AdamW optimizer** (`zmlx.optimizers`): single-kernel optimizer step
  that fuses m/v/parameter updates to reduce memory bandwidth.
- **New kernels**: paged attention, MoE dispatch/combine, FP8/NF4 dequantization,
  extended RoPE variants (interleaved, GQA), additional fused transformer ops.
- **Qwen3-32B-4bit benchmark**: initial results (later invalidated — see v0.6.0).
  Properly-warmed benchmarks show dense models are neutral with ZMLX patches.
- **Documentation**: quickstart guide (`docs/QUICKSTART.md`), cookbook
  (`docs/COOKBOOK.md`), new examples (custom activation, custom loss, custom
  reduction, Qwen fine-tuning).

### Changed

- **Branding**: "Triton for Apple Silicon" — clarified positioning as a kernel
  authoring toolkit, not a drop-in replacement for MLX built-ins.
- **README**: reorganized with "What's New" section, prominent 32B benchmark
  results, honest analysis of where ZMLX helps vs. where MLX built-ins win.
- **Autotune system** (`zmlx.autotune`): refactored with persistent cache
  support and cleaner API.
- **Patch system**: added preset constants (`FUSED_ACTIVATIONS`,
  `TRAINING_RECOMMENDED`) and `threadgroup="auto"` support for all patch
  modules.

### Fixed

- **MoE compatibility**: SwiGLU patch now correctly skips MoE `switch_mlp`
  modules that take routing indices as extra arguments.
- **`rmsnorm_residual` weight gradient**: VJP now correctly computes `d_weight`.
- **`compute_dtype` forwarding**: internal calls no longer emit spurious
  deprecation warnings.

### Deprecated

- **`compute_dtype` parameter**: emits `DeprecationWarning` when non-None
  value is passed. All kernels compute in float32 internally.

## [0.3.1] - 2026-01-30

### Added

- **Patch presets**: `FUSED_ACTIVATIONS` and `TRAINING_RECOMMENDED` constants
  in `zmlx.patch` for selecting the right set of kernel patches per workload.
  `FUSED_ACTIVATIONS` is safe for inference (SwiGLU/GeGLU only);
  `TRAINING_RECOMMENDED` includes norms and residual fusion for training.
- **Inference benchmark selective mode**: `--selective` flag in
  `benchmarks/inference_benchmark.py` to benchmark fused-activations-only
  patches.
- **Real benchmark data in README**: op-level and model-level results with
  honest analysis of where ZMLX helps and where MLX built-ins are faster.

### Fixed

- **`rmsnorm_residual` weight gradient**: the VJP now correctly computes
  `d_weight` (previously returned `None`). Training with `rmsnorm_residual`
  will now update the weight parameter as expected.
- **Internal `compute_dtype` forwarding**: patch modules and internal VJP
  calls no longer pass `compute_dtype` to kernel functions, eliminating
  spurious deprecation warnings during normal use.

### Deprecated

- **`compute_dtype` parameter**: all kernel functions that accepted
  `compute_dtype` now emit a `DeprecationWarning` when a non-None value is
  passed. All ZMLX Metal kernels already compute internally in float32
  regardless of this parameter. The parameter will be removed in a future
  release.

## [0.2.1] - 2026-01-30

### Added

- **VLSP kernels** (`zmlx.kernels.vlsp`) for recurrent latent reasoning:
  fused recurrent step, depth gate sigmoid (STE), GRPO advantage normalization,
  and a fused SiLU * residual helper (with tests).
- **Zig frontend (experimental)** with a minimal C++ shim over MLX:
  MetalKernel wrapper + cache, codegen, elementwise/rowwise helpers,
  and a small kernel catalog (activations, softmax, norms).

### Changed

- Zig build/test targets consolidate catalog tests under `kernels.zig`
  to avoid module path issues and make GPU catalog tests easy to run.
- Documentation updates for the Zig frontend architecture and build flow.

## [0.2.0] - 2026-01-29

First public release.

### Added

- **MetalKernel wrapper** (`zmlx.metal`) around `mx.fast.metal_kernel` with in-process
  caching (keyed on source hash + config), stats tracking, and optional GPU timing.
- **Kernel compilation cache** (`zmlx.cache`) with `KernelCacheKey` and global cache.
- **Code generation helpers** (`zmlx.codegen`) for Metal Shading Language templates:
  elementwise unary/binary, rowwise reduction, parallel reduction, and two-pass
  map-reduce patterns.
- **Elementwise API** (`zmlx.elementwise`) with `unary()`, `binary()`, and `map()`
  generators for quick kernel creation from C expressions.
- **Autograd API** (`zmlx.autograd`) with `unary_from_expr()`, `binary_from_expr()`,
  and `nary_from_expr()` for differentiable ops with Metal kernel forward+backward
  via `@mx.custom_function`.
- **Rowwise API** (`zmlx.rowwise`) with `map_reduce` helper for rowwise reduction patterns.
- **Autotuning** (`zmlx.autotune`) with `autotune_threadgroup()` search across candidates,
  cached results via `GLOBAL_AUTOTUNE_CACHE`, and `KernelSearch` for comparing
  different kernel implementations.
- **Kernel catalog** with 17 modules and 70+ kernels (including gradient helpers):
  - `activations` — exp, log, tanh, sigmoid, relu, silu, gelu_tanh, softplus, mish, elu
    (each with gradient-enabled variant)
  - `softmax` — softmax_lastdim (grad), log_softmax_lastdim, softmax_grad
  - `norms` — rmsnorm (grad), rmsnorm_grad, layernorm, rms_norm_no_weight,
    layer_norm_no_weight, layer_norm_dropout
  - `rope` — apply_rope, apply_rope_interleaved, apply_gqa_rope
  - `transformer` — swiglu, geglu, rmsnorm_residual, layernorm_residual,
    fused_add_rmsnorm, fused_add_layernorm, dropout, rms_norm_dropout,
    bias_swiglu, bias_geglu
  - `attention` — logsumexp_lastdim, masked_softmax, scale_mask_softmax,
    attention_tile_proto (experimental)
  - `reductions` — sum, mean, max, var, std, argmax, topk (all lastdim)
  - `fused` — add, mul, bias_gelu_tanh, bias_silu, silu_mul_grad, add_bias
  - `linear` — fused_linear_bias_silu, fused_linear_bias_gelu, fused_linear_rmsnorm
  - `loss` — softmax_cross_entropy
  - `quant` — dequantize_int8, dequantize_silu_int8, dequantize_int4
  - `bits` — pack_bits, unpack_bits
  - `moe` — top2_gating_softmax
  - `image` — resize_bilinear, depthwise_conv_3x3
  - `indexing` — fused_gather_add, fused_scatter_add
  - `scan` — cumsum_lastdim, cumsum_grad
- **Kernel registry** (`zmlx.registry`) for listing and managing cached kernels.
- **MSL snippet library** (`zmlx.msl`) with sigmoid, silu, gelu_tanh inline functions.
- **Platform guard** that raises on unsupported hosts when accessing the API.
- **8 examples** demonstrating elementwise ops, autograd, autotuning, catalog kernels,
  RoPE, transformer fragments, fused SiLU, and the kernel registry.
- **CI workflow** (`.github/workflows/ci.yml`) running tests on macOS.
- **Release workflow** (`.github/workflows/release.yml`) for PyPI trusted publishing.
- **Benchmarks** (`benchmarks/microbench.py`) with timing comparisons vs MLX reference ops.

[Unreleased]: https://github.com/Hmbown/ZMLX/compare/v0.6.2...HEAD
[0.6.2]: https://github.com/Hmbown/ZMLX/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/Hmbown/ZMLX/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Hmbown/ZMLX/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/Hmbown/ZMLX/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/Hmbown/ZMLX/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Hmbown/ZMLX/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Hmbown/ZMLX/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/Hmbown/ZMLX/compare/v0.2.1...v0.3.1
[0.2.1]: https://github.com/Hmbown/ZMLX/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Hmbown/ZMLX/releases/tag/v0.2.0
