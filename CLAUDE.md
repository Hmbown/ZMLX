# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZMLX is a Python toolkit for authoring and autotuning MLX custom Metal kernels on Apple silicon. It wraps `mx.fast.metal_kernel` and `mx.custom_function` to provide Numba/Triton-like ergonomics: define-once caching, safe launch defaults, threadgroup autotuning, custom VJP/JVP, and a catalog of 70+ ready-to-use kernels.

**Requirements**: macOS >= 14, Apple silicon (M-series), Python >= 3.10, mlx >= 0.30.0

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Tests (require macOS arm64 with Metal GPU)
pytest -q                          # all tests
pytest tests/test_kernels_catalog.py -q   # catalog correctness tests only
pytest -k "test_softmax" -q        # single test by name

# Lint & type check
ruff check .                       # lint (line-length=100, rules: E,F,I,UP,B)
ruff check --fix .                 # autofix
mypy src                           # type check

# Benchmarks
python benchmarks/microbench.py    # timing comparisons vs MLX reference ops
```

## Architecture

The codebase follows a three-layer architecture, all under `src/zmlx/`:

### Layer 1: Metal Kernel Infrastructure
- **`metal.py`** — `MetalKernel` wrapper around `mx.fast.metal_kernel` with in-process caching (keyed on source hash + config), stats tracking, and optional GPU timing via verbose mode.
- **`cache.py`** — `KernelCacheKey` and the global kernel compilation cache.
- **`_compat.py`** — Platform detection (macOS arm64) and MLX import guard.

### Layer 2: Code Generation & Helpers
- **`codegen.py`** — MSL template generators: `elementwise_unary_source`, `elementwise_binary_source`, `rowwise_reduction_source`, `rowwise_parallel_reduction_source`, `rowwise_mapreduce_source` (two-pass, e.g. for softmax).
- **`msl.py`** — Shared Metal snippet library (sigmoid, silu, gelu_tanh inline functions).
- **`elementwise.py`** — `unary()`, `binary()`, `map()` generators for quick kernel creation from C expressions.
- **`autograd.py`** — `unary_from_expr()`, `binary_from_expr()`, `nary_from_expr()` — creates differentiable ops with Metal kernel forward+backward, wrapped via `@mx.custom_function`.
- **`rowwise.py`** — `map_reduce` helper for rowwise reduction patterns.
- **`autotune.py`** — `autotune_threadgroup()` search across candidates with timing; results cached in `GLOBAL_AUTOTUNE_CACHE`.

### Layer 3: Kernel Catalog (`kernels/`)
17 modules organized by domain: `activations`, `softmax`, `norms`, `rope`, `transformer`, `attention`, `reductions`, `fused`, `linear`, `loss`, `quant`, `bits`, `moe`, `image`, `indexing`, `scan`.

**Pattern for catalog kernels**: each function typically uses `@lru_cache` for compilation, validates threadgroup sizes (powers of 2), wraps in `@mx.custom_function` for autodiff, and returns a callable that behaves like a standard MLX op.

### Key Design Decisions
- **No broadcasting**: binary ops require matching shapes for stable kernel generation.
- **Source-string caching**: MLX caches compiled Metal by source + specialization, so ZMLX generates sources deterministically.
- **Gradients as kernels**: custom VJP/JVP backward passes are themselves Metal kernels (no autodiff tape overhead).
- **Minimal IR**: only elementwise and rowwise map-reduce patterns are encoded; no heavyweight DSL.

## Conventions

- Ruff config: line-length 100, target py310, rules `[E, F, I, UP, B]`, E501 ignored.
- New kernels need correctness tests vs MLX reference ops (see `tests/test_kernels_catalog.py`).
- Performance claims need reproducible benchmarks with shape/dtype/device reported.
- Public API additions should include an example in `examples/`.
- Tests auto-skip on non-macOS-arm64 or when Metal is unavailable (see `tests/conftest.py`).

## Key Documentation

- `docs/ARCHITECTURE.md` — Design philosophy and multi-frontend vision (Python now, Zig planned via MLX-C).
- `docs/KERNELS.md` — Complete kernel catalog reference.
- `docs/AGENTS.md` — Multi-agent development roles and project backlog.
- `docs/UPSTREAMING.md` — Guide for contributing patterns upstream to MLX.
