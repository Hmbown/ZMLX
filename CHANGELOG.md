# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  - `linear` — fused_linear_bias_silu, fused_linear_bias_gelu, fused_linear_rmsnorm,
    dequantize_int4_matmul
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

[Unreleased]: https://github.com/Hmbown/ZMLX/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Hmbown/ZMLX/releases/tag/v0.2.0
