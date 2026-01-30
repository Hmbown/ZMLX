# ZMLX — Triton for Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

**The Triton-like toolkit for [MLX](https://github.com/ml-explore/mlx)** — write custom Metal GPU kernels from Python with one-line ergonomics, automatic gradients, and built-in autotuning. The killer feature: **fused MoE routing kernels** that deliver 1.3–1.6x inference speedup on Mixture-of-Experts models.

> **+60% prompt / +31% decode** on Qwen3-30B-A3B-Instruct (MoE) with `patch(model)` — safe default, zero regressions on any model. [Benchmarks](#model-level-inference)

```bash
pip install zmlx
```

**Speed up MoE inference in 3 lines:**

```python
import mlx_lm
from zmlx.patch import patch

model, tokenizer = mlx_lm.load("mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit")
patch(model)  # +60% prompt, +31% decode — safe on all models, MoE gets the win
```

**Or write custom GPU kernels in one line:**

```python
from zmlx.api import elementwise
import mlx.core as mx

# Math formula → compiled Metal kernel → runs on GPU
mish = elementwise("x * tanh(log(1 + exp(x)))", name="mish")
y = mish(mx.random.normal((1024,)))
```

---

## What's New in v0.6.0

- **`mode` parameter**: `patch(model, mode="training")` for workload-aware preset selection.
  `mode="inference"` (default) applies fused activations only; `mode="training"` adds norm fusions.
- **Validated benchmarks**: comprehensive 3-model benchmark suite with proper warmup confirms:
  - MoE models: **1.3–1.6x speedup** (the killer feature)
  - Dense models: neutral (no regression, no gain — MLX built-ins are already at bandwidth limit)
  - `ALL_PATTERNS`: **causes 3–5% regression on ALL models** — docstrings now warn explicitly
- **Bogus 32B result removed**: the previously-reported 1.33x Qwen3-32B speedup was a cold-cache
  benchmark artifact. Properly-warmed benchmarks show dense models are bandwidth-bound and neutral.

### Previous highlights (v0.4–0.5)

- **MoE patch** — fused `top2_gating_softmax` + `moe_combine` eliminates 4+ kernel launches in expert routing
- **High-level API** — `elementwise()`, `reduce()`, `map_reduce()` for kernel authoring in one line
- **JIT compiler** — `@jit` decorator compiles Python scalar expressions to Metal
- **Smart patching** — `smart_patch()` auto-benchmarks each pattern and keeps only what helps
- **Training pipeline** — `zmlx train` CLI for LoRA fine-tuning with ZMLX patches
- **70+ kernel catalog** — activations, norms, RoPE, attention, MoE, quantization, loss, bit ops

---

## Why ZMLX?

When you need a custom GPU op on Apple Silicon, your options today are:
1. Write raw Metal source strings, manage caching, figure out threadgroups, wire up autodiff manually
2. Use ZMLX

ZMLX wraps `mx.fast.metal_kernel` and `mx.custom_function` to provide **Triton-like ergonomics**:

- **One-line kernel authoring** — define elementwise, reduction, and map-reduce ops from C expressions
- **Automatic gradients** — custom VJP backward passes (themselves Metal kernels) via `mx.custom_function`
- **Define-once caching** — kernels compile once, reused by source hash + config
- **Autotuning** — threadgroup size search with persistent caching
- **Testing & benchmarking** — verify against reference ops, compare timings side-by-side
- **Model patching** — swap MLX layers for fused ZMLX kernels with `patch(model)`

---

## Install

**Requirements**: macOS (Apple Silicon), Python >= 3.10, mlx >= 0.30.0

```bash
pip install zmlx
```

From source (development):

```bash
git clone https://github.com/Hmbown/ZMLX.git
cd ZMLX
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Custom elementwise kernel

```python
from zmlx.api import elementwise
import mlx.core as mx

# Non-differentiable — just forward pass
fast_exp = elementwise("metal::exp(x)", name="fast_exp")
y = fast_exp(mx.random.normal((1024,)))

# Differentiable — with custom VJP
from zmlx import msl

silu = elementwise(
    "kk_silu(x)",
    name="my_silu",
    grad_expr="g * (s + x * s * ((T)1 - s))",
    grad_prelude="T s = kk_sigmoid(x);",
    use_output=False,
    header=msl.DEFAULT_HEADER,
)
gx = mx.grad(lambda z: silu(z).sum())(mx.random.normal((1024,)))
```

### 2. Custom reduction

```python
from zmlx.api import reduce
import mlx.core as mx

my_sum = reduce(init="0.0f", update="acc + v", name="row_sum")
y = my_sum(mx.random.normal((8, 1024)))  # shape (8,)
```

### 3. Two-pass map-reduce (softmax pattern)

```python
from zmlx.api import map_reduce
import mlx.core as mx

my_softmax = map_reduce(
    pass1={"init": "-INFINITY", "update": "max(acc1, x)", "reduce": "max(a, b)"},
    pass2={"init": "0.0f", "update": "acc2 + exp(x - s1)", "reduce": "a + b"},
    write="exp(x - s1) / s2",
    name="my_softmax",
)
y = my_softmax(mx.random.normal((8, 1024)))
```

### 4. Test and benchmark your kernel

```python
import zmlx
import mlx.core as mx

# Verify correctness
zmlx.testing.assert_matches(
    my_softmax, lambda x: mx.softmax(x, axis=-1),
    shapes=[(8, 1024), (32, 4096)],
)

# Benchmark
zmlx.bench.compare(
    {"ZMLX": my_softmax, "MLX": lambda x: mx.softmax(x, axis=-1)},
    shapes=[(1024, 4096), (4096, 4096)],
)
```

### 5. Lower-level building blocks

```python
from zmlx import autograd, elementwise, msl
import mlx.core as mx

# Unary kernel (no gradient)
exp_kern = elementwise.unary(
    name="kk_exp", expr="metal::exp(x)",
    compute_dtype=mx.float32, header=msl.DEFAULT_HEADER,
)

# Binary kernel with custom VJP
mul_op = autograd.binary_from_expr(
    name="safe_mul", fwd_expr="a * b",
    vjp_lhs_expr="g * b", vjp_rhs_expr="g * a",
    compute_dtype=mx.float32,
)
```

---

## Kernel Catalog

ZMLX includes 70+ kernels organized by domain. Some are genuinely useful for custom workloads (loss, GLU fusions, bit ops, MoE gating). Others are **reference implementations** showing codegen patterns — correct but not faster than MLX built-ins for standard transformer shapes.

Full reference: [`docs/KERNELS.md`](docs/KERNELS.md).

| Module | Highlights |
|:---|:---|
| `loss` | `softmax_cross_entropy` — memory-efficient fused loss |
| `transformer` | `swiglu`, `geglu`, `rmsnorm_residual` (with full weight gradients), `dropout` — genuine fusions |
| `bits` | `pack_bits`, `unpack_bits` — no MLX equivalent |
| `moe` | `top2_gating_softmax`, `moe_dispatch`, `moe_combine` — fused expert routing (+36% decode on 30B MoE) |
| `quant` | FP8 (E4M3/E5M2), NF4, int8, int4 dequantization — real bit-manipulation kernels |
| `optimizers` | `adamw_step` — fused AdamW parameter update in a single kernel |
| `scan` | `cumsum_lastdim` — differentiable prefix sum |
| `norms` | `rmsnorm`, `layernorm` — parallel reduction. All norms compute in float32 internally |
| `softmax` | `softmax_lastdim` — map-reduce codegen showcase |
| `rope` | `apply_rope`, `apply_rope_interleaved`, `apply_gqa_rope` |
| `linear` | Reference fused-linear patterns (naive matmul, not for production) |

---

## Architecture

Three-layer design. Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

1. **Metal kernel infrastructure** — `MetalKernel` wrapper, in-process cache, stats tracking
2. **Code generation & helpers** — MSL templates, elementwise/autograd/rowwise APIs, autotuning
3. **Kernel catalog** — domain modules built on layers 1 and 2

---

## Benchmarks

### Op-level (B=16, S=1024, D=1024, float32, M4 Max)

Run `python benchmarks/microbench.py` to reproduce on your hardware.

| Operation | MLX | ZMLX | Speedup |
|:--|--:|--:|:--|
| **SwiGLU** | 0.85 ms | **0.40 ms** | **2.1x** |
| **Dropout** | 3.12 ms | **0.38 ms** | **8.2x** |
| **Top-K** | 1.82 ms | **0.49 ms** | **3.7x** |
| **Gather-Add** | 0.54 ms | **0.41 ms** | **1.3x** |
| Softmax | 0.36 ms | 0.41 ms | 0.90x |
| RMSNorm | 0.37 ms | 0.41 ms | 0.90x |
| Sum | 0.19 ms | 0.36 ms | 0.53x |
| CumSum | 0.30 ms | 0.59 ms | 0.51x |

ZMLX wins big on **fused operations** that MLX doesn't provide as single ops (SwiGLU, fused-RNG dropout, fused gather-add). MLX's built-in operations (`mx.fast.rms_norm`, `mx.softmax`, reductions) are already highly optimized and should not be replaced.

### Model-level inference

All baselines are **unmodified `mlx_lm`** (`mlx_lm.load()` + `mlx_lm.generate()`) — the standard MLX inference stack. ZMLX rows add `patch(model)` (default: `FUSED_ACTIVATIONS`) or explicit `ALL_PATTERNS` on top of that same pipeline. Same model weights, same quantization, same prompt.

LLM inference is **memory-bandwidth-bound**: fused kernels shine on large models where each saved memory round-trip matters. The effect scales with model size — small models see no benefit, large models see significant speedups.

**Qwen3-32B-4bit (dense, 64 layers, ~19 GB)** — M4 Max, 36 GB

| Config | Prompt (tok/s) | Decode (tok/s) | vs Baseline |
|:--|--:|--:|:--|
| Baseline (`mlx_lm`) | 149 | 18.0 | — |
| `patch(model)` (default) | 148 | 17.7 | 0.99x / 0.99x |
| `patch(model, patterns=ALL_PATTERNS)` | 147 | 17.2 | 0.99x / 0.97x |

> Dense batch-1 decode is ~95% weight-reading through quantized matmuls — already at the hardware bandwidth limit. Custom kernels can't beat MLX's built-in `mx.fast.rms_norm`/`mx.fast.rope`/`mx.fast.scaled_dot_product_attention`. ZMLX shines on **MoE models** where fused expert routing eliminates kernel launches that MLX lacks fast paths for.

**Qwen3-30B-A3B-Instruct-2507-4bit (MoE, 48 layers, 3B active/30B total)** — M4 Max, 36 GB

| Config | Prompt (tok/s) | Decode (tok/s) | vs Baseline |
|:--|--:|--:|:--|
| Baseline (`mlx_lm`) | 1,093 | 106 | — |
| `patch(model)` (default) | **1,754** | **138** | **1.60x / 1.31x** |

> **+60% prompt, +31% decode** with default `patch(model)` on MoE — fused gating (`top2_gating_softmax`) and combine (`moe_combine`) kernels eliminate multiple memory round-trips in the expert routing path. No regressions.

**Qwen3-8B-4bit (32 layers, ~5 GB)** — `python benchmarks/inference_benchmark.py --models qwen3-8b --selective`

| Config | Prompt (tok/s) | Decode (tok/s) | vs Baseline |
|:--|--:|--:|:--|
| Baseline (`mlx_lm`) | 676 | 75 | — |
| `patch(model)` (default) | 675 | 76 | 1.00x / 1.01x |

**Llama-3.2-1B-Instruct-4bit (16 layers, ~0.8 GB)** — `python benchmarks/llama_benchmark.py`

| Config | Prompt (tok/s) | Decode (tok/s) | vs Baseline |
|:--|--:|--:|:--|
| Baseline (`mlx_lm`) | 3,913 | 377 | — |
| `patch(model)` (default) | 3,804 | 378 | 0.97x / 1.00x |
| `patch(model, patterns=ALL_PATTERNS)` | 3,705 | 366 | 0.95x / 0.97x |

**GLM-4.7-Flash-4bit (MoE, 47 layers, 30B-A3B, sigmoid+group gating)** — M4 Max, 36 GB

| Config | Prompt (tok/s) | Decode (tok/s) | vs Baseline |
|:--|--:|--:|:--|
| Baseline (`mlx_lm`) | 662 | 74.1 | — |
| `patch(model)` (default) | 672 | 73.6 | 1.01x / 0.99x |

> GLM-4 uses sigmoid + group-based gating with `@mx.compile`, so the gate is already optimized. ZMLX preserves the original gating and only fuses the combine step — not enough to move the needle. **Neutral, no regression.**

**When do patches help?**
- **MoE Models (softmax-gated)**: Qwen3-MoE, Mixtral — fused gating provides 1.3–1.6x. The gate must return raw logits (not pre-computed indices).
- **MoE Models (pre-computed gating)**: GLM-4, DeepSeek-V3 — neutral. Gate is already `@mx.compile`-optimized.
- **Large Dense Models (32B+)**: Neutral. MLX built-ins are already at bandwidth limit.
- **Medium Dense Models (8B)**: Neutral-to-positive, no regressions.
- **Small Models (< 3B)**: Neutral. Use `smart_patch` to be sure.

```python
from zmlx.patch import smart_patch
import mlx.core as mx

# Auto-benchmark each pattern, keep only what helps
sample = mx.array([tokenizer.encode("Hello")])
model = smart_patch(model, sample)
```

Or use mode/presets if you know your workload:

```python
from zmlx.patch import patch

patch(model)                    # inference (safe default — MoE gets 1.3–1.6x, dense neutral)
patch(model, mode="training")   # training: adds norm fusions for backward pass savings

# Or explicit presets for full control:
from zmlx.patch import ALL_PATTERNS, FUSED_ACTIVATIONS, TRAINING_RECOMMENDED
patch(model, patterns=FUSED_ACTIVATIONS)       # same as default
patch(model, patterns=TRAINING_RECOMMENDED)    # same as mode="training"
patch(model, patterns=ALL_PATTERNS)            # WARNING: regresses 3–5% on inference
```

### Smart patching

`smart_patch` applies each candidate pattern, benchmarks the model's forward pass, and **automatically reverts patterns that make things slower**. It supports custom forward functions for realistic benchmarks:

```python
from zmlx.patch import smart_patch

# Basic: benchmark raw forward pass
model = smart_patch(model, sample_input)

# Advanced: benchmark with actual generation
def gen_fn(model, sample):
    return mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=20)

model = smart_patch(model, sample, forward_fn=gen_fn, threshold=0.99)

# Result includes per-pattern speedups
result = model._zmlx_patch_result
print(result.benchmarks)    # {'swiglu_mlp': 1.012, 'residual_norm': 0.971}
print(result.summary())     # what was kept and why
```

### Autotuning

Replacement modules support `threadgroup="auto"` to search for the best threadgroup size on first invocation:

```python
from zmlx.patch import patch
patch(model, threadgroup="auto")  # autotunes each kernel on first call
```

The `map_reduce()` API also supports autotuning:

```python
from zmlx.api import map_reduce
my_softmax = map_reduce(..., threadgroup="auto")  # autotunes per-shape
```

### Where ZMLX genuinely helps

- **MoE model inference (softmax-gated)** — 1.3–1.6x speedup on MoE models whose gate returns raw logits with softmax top-k routing (Qwen3-MoE, Mixtral). Fused gating eliminates 4+ kernel launches per MoE layer. Models with pre-computed gating (GLM-4, DeepSeek-V3 with `@mx.compile`) are neutral — the pattern detects this and only fuses the combine step.
- **Custom ops that MLX doesn't have** — SwiGLU, GeGLU, fused dropout, fused MoE gating, bit packing
- **Training** — fused `softmax_cross_entropy` loss, correct weight gradients for `rmsnorm_residual`
- **Authoring new kernels** — the `elementwise()`, `reduce()`, and `map_reduce()` APIs let you go from math formula to compiled Metal kernel in one line
- **Quantization** — FP8 (E4M3/E5M2), NF4, int8, int4 dequantization with real bit-manipulation kernels

### Where ZMLX won't help

- **Dense model inference** — batch-1 decode is ~95% weight-reading through quantized matmuls, already at hardware bandwidth limit. `patch(model)` is safe (neutral) but won't speed things up.
- **Replacing MLX built-in norms/softmax** — `mx.fast.rms_norm`, `mx.softmax`, `mx.fast.scaled_dot_product_attention` are Apple-optimized fast paths. Custom kernels add dispatch overhead.

---

## Precision

All ZMLX Metal kernels compute internally in **float32** regardless of input dtype. The `compute_dtype` parameter accepted by many kernel functions is **deprecated** and will be removed in a future release. Passing a non-None value will emit a `DeprecationWarning`.

---

## Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 5-minute tutorial
- [`docs/COOKBOOK.md`](docs/COOKBOOK.md) — Recipes for common patterns
- [`docs/KERNELS.md`](docs/KERNELS.md) — Complete kernel catalog reference
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Design philosophy

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and conventions.

---

## License

MIT. See [`LICENSE`](LICENSE).
