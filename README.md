# ZMLX - Triton-style kernels for Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

**A Metal kernel toolkit and upstream incubation lab for [MLX](https://github.com/ml-explore/mlx)** — author custom GPU kernels in Python, apply fused model patches, and prototype C++ Metal primitives for upstream MLX.

ZMLX is **additive** to MLX, not a replacement:
- **Stock MLX**: `pip install mlx` + `pip install zmlx` (pure Python + Metal kernels)
- **Local MLX fork**: build/install `mlx_local/` + `pip install zmlx` (enables fused C++ primitives like `gather_qmm_swiglu`)

Toolkit highlights (available via `pip install zmlx` on stock MLX):
- SwiGLU 2.0x, Dropout 7.5x, Top-K 3.7x in op-level microbenchmarks
- 70+ kernel catalog, autograd, benchmarking utilities, and model patching

> **Validated benchmarks** (M4 Max 36 GB, MLX 0.30.4.dev, Jan 31, 2026):
>
> | Result | Measurement | Notes |
> |:--|:--|:--|
> | **+4% per MoE layer** on LFM2-8B-A1B-4bit (MoE, E=32, K=4) | 293 us -> 282 us at M=1 decode | Kernel-level bench; E2E too noisy to report |
> | **+12% decode** on LFM2-8B-A1B-4bit (default FUSED_ACTIVATIONS) | 223 -> 251 tok/s (prefill neutral) | Token-identical in `zmlx.validate` (200-token greedy) |
> | **Qwen3-30B-A3B-4bit (moe_mlp)** | 110.5 -> 120.7 tok/s | **Not token-identical** (13/200; diverges at token 0) |
> | **Qwen3-4B-4bit (swiglu_mlp default)** | 128.6 -> 129.0 tok/s | **Not token-identical** (21/200; diverges at token 18) |
>
> E2E numbers above are from `python -m zmlx.validate` with a fixed 200-token greedy prompt and are best used for relative comparisons. Fast models can show variance.
>
> Fused SwiGLU requires a [local MLX build](#optimization-lab) with `gather_qmm_swiglu`. On stock MLX, gating+combine-only can be neutral to slightly negative, so use `smart_patch` or a fused build for MoE models and verify token fidelity.
>
> **Token fidelity** (greedy decode vs unpatched `mlx_lm`, Jan 31, 2026):
>
> | Model | Pattern(s) | Tokens matching | Notes |
> |:--|:--|:--|:--|
> | LFM2-8B-A1B-4bit | `moe_mlp`, default FUSED_ACTIVATIONS | 200/200 | Token-identical in `zmlx.validate` (200 tokens) |
> | LFM2-8B-A1B-8bit | `moe_mlp` | 200/200 | Token-identical in `zmlx.validate` (200 tokens) |
> | Qwen3-30B-A3B-4bit | `moe_mlp` | 13/200 | Diverges at token 0; `residual_norm` is 101/200 |
> | GPT-OSS-20B-MXFP4-Q4 | `moe_mlp` | 2/50 | `residual_norm` improves to 29/50 |
> | Qwen3-4B-4bit | `swiglu_mlp` (default) | 21/200 | Diverges at token 18 |
> | Llama-3.2-1B-Instruct-4bit | `residual_norm` | 200/200 | Small decode regression (~0.98x) |
> | Qwen3-8B-4bit | `residual_norm` | 29/200 | Diverges at token 27 |
>
> Results are single-prompt greedy decode checks; see `experiments/gpt_oss_20b_moe.md` for GPT-OSS details.
> If you require strict parity, use baseline `mlx_lm` or `smart_patch` with `moe_mlp` excluded.

> **Benchmarking matrix** (to compare apples-to-apples):
>
> | Label | MLX install | ZMLX | Notes |
> |:--|:--|:--|:--|
> | Baseline | Stock `mlx` | None | Unpatched `mlx_lm` |
> | ZMLX (stock MLX) | Stock `mlx` | `pip install zmlx` | `patch(model)` without fused C++ primitives |
> | ZMLX + fused MLX | Local `mlx_local/` | `pip install zmlx` | `gather_qmm_swiglu` available |
>
> Always report MLX version, ZMLX version, device, and whether `gather_qmm_swiglu` is detected.
> You can check detection with:
>
> ```python
> from zmlx.kernels.fused_moe import has_gather_qmm_swiglu
> print(has_gather_qmm_swiglu())
> ```

```bash
pip install zmlx
```

### LFM2 with ZMLX

Works with both [4bit](https://huggingface.co/mlx-community/LFM2-8B-A1B-4bit) and [8bit](https://huggingface.co/mlx-community/LFM2-8B-A1B-8bit-MLX):

```python
import mlx_lm
from zmlx.patch import patch

model, tokenizer = mlx_lm.load("mlx-community/LFM2-8B-A1B-8bit-MLX")
patch(model)  # patches 22 MoE layers + 2 dense SwiGLU layers

text = mlx_lm.generate(model, tokenizer,
    prompt="Explain mixture-of-experts models in one paragraph.",
    max_tokens=100)
print(text)
```

Output (token-identical to unpatched `mlx_lm`):

```
A mixture-of-experts model is a neural architecture that combines multiple
specialized sub-networks (or "experts") to solve complex tasks by dynamically
routing input data to the most relevant expert based on content, enabling
efficient, scalable, and interpretable reasoning through collaborative
decision-making among diverse specialized components.
```

> **LFM2 speed & fidelity** (M4 Max 36 GB, `python -m zmlx.validate`, Jan 31, 2026):
>
> | | 4bit | 8bit |
> |:--|:--|:--|
> | **Token fidelity** | 200/200 identical | 200/200 identical |
> | **Decode** | 223 -> 251 tok/s (**+12%**) | 153 -> 168 tok/s (**+10%**) |
> | **Prefill** | 734 -> 746 tok/s (neutral) | 552 -> 570 tok/s (neutral) |
> | **Memory** | 5.3 GB | 9.5 GB |
> | **Kernel-level** | +4% per MoE layer (293 -> 282 us, decode) | Same architecture, similar improvement |
>
> Both variants fit on 16 GB M-series Macs. `patch(model)` auto-detects LFM2 and applies the validated pattern set (`moe_mlp` + `swiglu_mlp`).

**Custom kernel example:**

```python
from zmlx.api import elementwise
import mlx.core as mx

# Math formula → compiled Metal kernel → runs on GPU
mish = elementwise("x * tanh(log(1 + exp(x)))", name="mish")
y = mish(mx.random.normal((1024,)))
```

---

## What's New in v0.7.1

### Fused expert SwiGLU (`gather_qmm_swiglu`)

ZMLX prototypes fused C++ Metal primitives in a local MLX fork. The first: `gather_qmm_swiglu` — fuses gate projection + up projection + SwiGLU activation into a single kernel launch, reading the input tensor once instead of twice.

- **+5-8% decode** on Qwen3-30B-A3B-4bit (119-122 vs 113-114 tok/s, E2E)
- **+4% per MoE layer** on LFM2-8B-A1B-4bit (293 us → 282 us at M=1 decode, kernel-level)
- **Neutral (1.01x)** on Qwen3-4B-4bit (dense, E2E)
- Auto-enabled by `patch(model)` when available; falls back to two-pass otherwise
- Threshold-guarded: fused kernel used for M<=32 (decode/small prefill), falls back at large M where it is slower
- Added kernel-level + E2E benchmark scripts plus correctness tests

Requires building MLX from the local fork (`mlx_local/`). See [Optimization Lab](#optimization-lab) for details. Upstream PR to MLX planned.

### Optimization lab

ZMLX is evolving into two things:

1. **A Metal kernel toolkit** (stable) — `elementwise()`, `reduce()`, `map_reduce()`, autograd, model patching. This is what `pip install zmlx` gives you.
2. **An MLX optimization lab** (experimental) — prototyping fused C++ Metal primitives that should eventually be upstreamed to MLX. These require a local MLX build and live in `mlx_local/`.

The lab exists because some fusions (quantized matmul + activation) need access to MLX's internal SIMD helpers (`qdot`, `load_vector`, `QuantizedBlockLoader`) which aren't exposed through the public `metal_kernel` API. The plan is: prototype here, validate with benchmarks, upstream to MLX, then ZMLX auto-detects the primitives when they land.

**Current lab work:**
- `gather_qmm_swiglu` — fused expert gate+up+SwiGLU (done, working, benchmarked)
- `add_rms_norm` — fused residual add + RMSNorm (planned, benefits all models)
- `gather_qmm_combine` — fused down projection + weighted expert sum (planned, MoE-specific)

### Previous highlights (v0.6.x)

- SIMD-group top-k gating, bias-aware fused gating, dynamic `num_experts_per_tok`
- `topk_gating_softmax(x, k)` kernel (fused Metal for k<=8, full-softmax + expert bias)
- LFM2, GPT-OSS, GLM-4, Mixtral model support
- `mode` parameter, validated benchmarks, `router` attribute support

### Previous highlights (v0.4-0.5)

- MoE patch (fused gating + combine), high-level API, JIT compiler
- Smart patching, training pipeline, 70+ kernel catalog

---

## Why ZMLX?

When you need a custom GPU op on Apple Silicon, your options today are:
1. Write raw Metal source strings, manage caching, figure out threadgroups, wire up autodiff manually
2. Use ZMLX

ZMLX wraps `mx.fast.metal_kernel` and `mx.custom_function` to provide **Triton-like ergonomics**:

- **One-line kernel authoring** - define elementwise, reduction, and map-reduce ops from C expressions
- **Automatic gradients** - custom VJP backward passes (themselves Metal kernels) via `mx.custom_function`
- **Define-once caching** - kernels compile once, reused by source hash + config
- **Autotuning** - threadgroup size search with persistent caching
- **Testing & benchmarking** - verify against reference ops, compare timings side-by-side
- **Model patching** - swap MLX layers for fused ZMLX kernels with `patch(model)`

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

# Non-differentiable - just forward pass
fast_exp = elementwise("metal::exp(x)", name="fast_exp")
y = fast_exp(mx.random.normal((1024,)))

# Differentiable - with custom VJP
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

ZMLX includes 70+ kernels organized by domain. Some are genuinely useful for custom workloads (loss, GLU fusions, bit ops, MoE gating). Others are **reference implementations** showing codegen patterns - correct but not faster than MLX built-ins for standard transformer shapes.

Full reference: [`docs/KERNELS.md`](docs/KERNELS.md).

| Module | Highlights |
|:---|:---|
| `loss` | `softmax_cross_entropy` - memory-efficient fused loss |
| `transformer` | `swiglu`, `geglu`, `rmsnorm_residual` (with full weight gradients), `dropout` - genuine fusions |
| `bits` | `pack_bits`, `unpack_bits` - no MLX equivalent |
| `moe` | `topk_gating_softmax`, `moe_dispatch`, `moe_combine` - fused expert routing (k ≤ 8 fused, bias-aware) |
| `quant` | FP8 (E4M3/E5M2), NF4, int8, int4 dequantization - real bit-manipulation kernels |
| `optimizers` | `adamw_step` - fused AdamW parameter update in a single kernel |
| `scan` | `cumsum_lastdim` - differentiable prefix sum |
| `norms` | `rmsnorm`, `layernorm` - parallel reduction. All norms compute in float32 internally |
| `softmax` | `softmax_lastdim` - map-reduce codegen showcase |
| `rope` | `apply_rope`, `apply_rope_interleaved`, `apply_gqa_rope` |
| `linear` | Reference fused-linear patterns (naive matmul, not for production) |

---

## Architecture

Three-layer design. Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

1. **Metal kernel infrastructure** - `MetalKernel` wrapper, in-process cache, stats tracking
2. **Code generation & helpers** - MSL templates, elementwise/autograd/rowwise APIs, autotuning
3. **Kernel catalog** - domain modules built on layers 1 and 2

---

## Benchmarks

Three levels of measurement:
1. **Isolated primitives** — `bench_gather_qmm_swiglu.py`
2. **Single MoE layer** — `bench_moe_layer.py` (500 iters, p50 median, `mx.synchronize()`)
3. **E2E model decode** — `bench_moe_e2e.py`

### Op-level (B=16, S=1024, D=1024, float16, M4 Max)

Run `python benchmarks/microbench.py` to reproduce on your hardware.

| Operation | MLX | ZMLX | Speedup |
|:--|--:|--:|:--|
| **SwiGLU** | 0.87 ms | **0.43 ms** | **2.0x** |
| **Dropout** | 3.08 ms | **0.41 ms** | **7.5x** |
| **Top-K** | 1.81 ms | **0.49 ms** | **3.7x** |
| **Gather-Add** | 0.55 ms | **0.42 ms** | **1.3x** |
| Softmax | 0.45 ms | 0.44 ms | ~1.0x |
| RMSNorm | 0.51 ms | 0.54 ms | 0.95x |
| MoE gating | 0.35 ms | 0.37 ms | 0.94x |
| Sum | 0.22 ms | 0.37 ms | 0.58x |
| CumSum | 0.32 ms | 0.62 ms | 0.52x |

ZMLX is most effective for **fused operations** that MLX does not provide as single ops (SwiGLU, fused-RNG dropout, fused gather-add). MLX built-ins (`mx.fast.rms_norm`, `mx.softmax`, reductions) are already highly optimized and remain the preferred choice for standard transformer shapes.

### Model-level inference (E2E)

All baselines are **unmodified `mlx_lm`**. ZMLX rows add `patch(model)`. Same model weights, same quantization, same prompt. Benchmarks on M4 Max 36 GB, MLX 0.30.4.dev, Jan 31, 2026 unless noted.

#### MoE models

**Qwen3-30B-A3B-4bit** (MoE, 48 layers, E=128, K=8) — `python benchmarks/bench_moe_e2e.py`  
3 runs x 200 tokens, repeated 3 times.

| Config | Decode (tok/s) | vs Baseline |
|:--|--:|:--|
| Baseline (`mlx_lm`) | 113-114 | — |
| `patch(model)` — gating + combine only | 109-111 | 0.95-0.98x |
| `patch(model)` — with fused SwiGLU | 119-122 | **+5-8%** |

> Fused SwiGLU requires a local MLX build. Gating+combine alone measured below baseline on this model.

**LFM2-8B-A1B** (MoE, 24 layers, E=32, K=4) — both [4bit](https://huggingface.co/mlx-community/LFM2-8B-A1B-4bit) and [8bit](https://huggingface.co/mlx-community/LFM2-8B-A1B-8bit-MLX) variants.

| Config | Prefill (tok/s) | Decode (tok/s) | vs Baseline |
|:--|--:|--:|:--|
| Baseline 4bit | 734 | 223 | — |
| `patch(model)` 4bit | 746 | 251 | **+12% decode** |
| Baseline 8bit | 552 | 153 | — |
| `patch(model)` 8bit | 570 | 168 | **+10% decode** |

Prefill is neutral (fused kernel activates at M<=32 only). Output is **token-identical** to unpatched `mlx_lm` (200/200 tokens, greedy). Both variants fit on 16 GB M-series Macs (5.3 GB 4bit / 9.5 GB 8bit). See [example above](#lfm2-with-zmlx).

#### Dense models (neutral — expected)

**Qwen3-4B-4bit** (dense, 36 layers) — `python benchmarks/bench_moe_e2e.py`  
3 runs x 1000 tokens.

| Config | Decode (tok/s) | vs Baseline |
|:--|--:|:--|
| Baseline (`mlx_lm`) | 124.4 | — |
| `patch(model)` | 125.3 | **1.01x (neutral)** |

> Dense decode is bandwidth-bound; patches are safe but not expected to help.

**Overnight suites (sequential, cache on external drive):**

```bash
python benchmarks/bench_moe_suite.py \
  --model-list benchmarks/moe_models.txt \
  --cache-dir /Volumes/VIXinSSD/TEST \
  --runs 3 --max-tokens 200 --resume
```

### Kernel-level MoE layer timing (authoritative for fast models)

`python benchmarks/bench_moe_layer.py` isolates a single MoE layer forward pass, timed with `mx.synchronize()` brackets. 500 iterations, p50 median. Per-measurement stdev: 4-10%.

**LFM2-8B-A1B-4bit** (E=32, K=4, hidden=2048, intermediate=1792):

| seq_len | Baseline | Gating+combine | Fused SwiGLU | Speedup |
|:--|--:|--:|--:|:--|
| 1 (decode) | 293 us | 288 us (1.02x) | 282 us | **1.04x** |
| 4 | 531 us | 500 us (1.06x) | 498 us | **1.07x** |
| 16 | 998 us | 943 us (1.06x) | 949 us | 1.05x |
| 64 | 2291 us | 2480 us (0.92x) | 2452 us | 0.93x |

**Qwen3-30B-A3B-4bit** (E=128, K=8, hidden=2048, intermediate=1024):

| seq_len | Baseline | Gating+combine | Fused SwiGLU | Speedup |
|:--|--:|--:|--:|:--|
| 1 (decode) | 613 us | 621 us (0.99x) | 613 us | **1.00x** |
| 4 | 808 us | 724 us (1.12x) | 756 us | **1.07x** |
| 16 | 1269 us | 1264 us (1.00x) | 1284 us | 0.99x |
| 64 | 3565 us | 3557 us (1.00x) | 3508 us | 1.02x |

**Key finding:** fused SwiGLU saves ~4% per MoE layer on LFM2 at decode, but is neutral on Qwen3-30B at the single-layer level. The +5-8% E2E gain on Qwen3-30B comes from compound system effects: 48 MoE layers x one eliminated kernel dispatch is ~240 us saved per token (~2.7% of 8.85 ms/token), plus reduced intermediate memory pressure and fewer graph nodes.

### Kernel-level primitive microbenchmarks

`python benchmarks/bench_gather_qmm_swiglu.py` — fused vs naive (2x `gather_qmm` + SwiGLU):

| Config | Naive | Fused | Speedup |
|:--|--:|--:|:--|
| M=1, K=512, N=512 | 170 us | 133 us | **1.28x** |
| M=1, K=2048, N=1024 | 144 us | 129 us | **1.11x** |
| M=1, K=2048, N=2048 | 148 us | 134 us | **1.10x** |
| M=16, K=2048, N=1024 | 243 us | 229 us | 1.06x |
| M=64, K=2048, N=1024 | 240 us | 534 us | 0.45x |

> Fused kernel improves small M (decode) and is slower at large M (prefill). The patch auto-selects: fused for M<=32, two-pass fallback otherwise.

### Next steps (roadmap)

- **`add_rms_norm`** — fused residual add + RMSNorm in one kernel (benefits all models, ~20 line Metal diff)
- **`gather_qmm_combine`** — fused down projection + weighted expert sum (MoE-specific, eliminates intermediate tensor)
- **Upstream `gather_qmm_swiglu` to MLX** — PR with benchmarks so everyone benefits via `pip install mlx`
- **Per-device autotune profiles** (better defaults by chip family)

**When do patches help?**
- **LFM2 (MoE, 4-bit/8-bit)**: Best supported model. **+12% decode** on 4-bit, **+10%** on 8-bit, prefill neutral, token-identical. `patch(model)` auto-applies the right patterns.
- **Qwen3-MoE / GPT-OSS**: `moe_mlp` improves speed (+9% on Qwen3-30B) but breaks token fidelity. `patch(model)` auto-excludes these. Fix pending — likely a gating normalization mismatch.
- **Dense models (Qwen3-4B, Llama, etc.)**: Neutral. Decode is bandwidth-bound. `swiglu_mlp` can diverge on Qwen3 quantized variants.
- **GLM-4, DeepSeek-V3**: Neutral. Pre-computed gating is already `@mx.compile`-optimized.

### Tested & validated models

Models below were tested with `python -m zmlx.validate` (200-token greedy decode, M4 Max, Jan 2026). `patch(model)` auto-detects the model family and applies only validated patterns.

| Model | `patch(model)` applies | Decode speedup | Token fidelity |
|:--|:--|:--|:--|
| **LFM2-8B-A1B-4bit** | `moe_mlp` + `swiglu_mlp` | **+12%** decode (223 -> 251 tok/s) | 200/200 |
| **LFM2-8B-A1B-8bit** | `moe_mlp` + `swiglu_mlp` | **+10%** decode (153 -> 168 tok/s) | 200/200 |
| Qwen3-30B-A3B-4bit | *auto-excluded* | — | fails at token 0 |
| Qwen3-4B-4bit | *auto-excluded* | — | fails at token 18 |
| GPT-OSS-20B-MXFP4-Q4 | *auto-excluded* | — | fails at token 1 |
| Llama-3.2-1B-4bit | neutral | 0.98x | 200/200 |

For models not listed here, run `python -m zmlx.validate <model>` before deploying. You can override auto-exclusions with `patch(model, patterns=[...])`.

### Model-aware defaults (v0.7.1)

`patch(model)` detects the model architecture and skips patterns with known fidelity issues:

```python
patch(model)
# LFM2:  applies moe_mlp + swiglu_mlp (+12% decode, token-identical)
# Qwen3: auto-excludes moe_mlp, swiglu_mlp (fidelity issues)
# Other: applies FUSED_ACTIVATIONS defaults
```

To force a specific pattern (at your own risk):

```python
patch(model, patterns=["moe_mlp"])  # prints a fidelity warning for Qwen
```

To auto-benchmark and keep only what speeds things up:

```python
from zmlx.patch import smart_patch
import mlx.core as mx

sample = mx.array([tokenizer.encode("Hello")])
model = smart_patch(model, sample)
```

Or use mode/presets if you know your workload:

```python
from zmlx.patch import patch

patch(model)                    # inference default; use smart_patch on stock MLX for MoE
patch(model, mode="training")   # training: adds norm fusions for backward pass savings

# Or explicit presets for full control:
from zmlx.patch import ALL_PATTERNS, FUSED_ACTIVATIONS, TRAINING_RECOMMENDED
patch(model, patterns=FUSED_ACTIVATIONS)       # same as default
patch(model, patterns=TRAINING_RECOMMENDED)    # same as mode="training"
patch(model, patterns=ALL_PATTERNS)            # WARNING: can be slower on inference
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

- **MoE model inference** — best results with fused expert SwiGLU (`gather_qmm_swiglu`, local MLX build). Qwen3-30B gets **+5-8%** E2E; LFM2 shows **+4% per layer** at decode. On stock MLX, use `smart_patch` to avoid gating+combine slowdowns. Supports Qwen3-MoE, LFM2, Mixtral, GPT-OSS.
- **Prototyping MLX-level optimizations** — ZMLX's optimization lab incubates C++ Metal primitives. Prove value with benchmarks here, then upstream to MLX for everyone.
- **Custom ops that MLX doesn't have** — SwiGLU, GeGLU, fused dropout, fused MoE gating, bit packing
- **Training** — fused `softmax_cross_entropy` loss, correct weight gradients for `rmsnorm_residual`
- **Authoring new kernels** — `elementwise()`, `reduce()`, `map_reduce()` APIs: math formula to compiled Metal kernel in one line
- **Quantization** — FP8 (E4M3/E5M2), NF4, int8, int4 dequantization with real bit-manipulation kernels

### MoE performance notes

- **Expert matmuls are the bottleneck** — for M=1 decode, MoE layers do 3 gather_qmm calls per layer (gate, up, down). Fusing gate+up+SwiGLU into one kernel (`gather_qmm_swiglu`) saves one full read of the input tensor.
- **E2E gains are a compound effect** — on Qwen3-30B, 48 MoE layers x one eliminated dispatch saves ~240 us per token (~2.7% of 8.85 ms/token), plus reduced intermediate memory pressure and fewer graph nodes.
- **Kernel-level timing complements E2E** — LFM2 E2E shows +12% decode (223 -> 251 tok/s, prefill neutral). The kernel-level MoE layer benchmark (+4% per layer at decode, p50 over 500 iterations) is the most stable measurement.

### Where ZMLX won't help

- **Dense model inference** — batch-1 decode is dominated by weight reads and bandwidth-bound. `patch(model)` is generally neutral (Qwen3-4B: 1.01x).
- **Replacing MLX built-in norms/softmax** — `mx.fast.rms_norm`, `mx.softmax`, `mx.fast.scaled_dot_product_attention` are Apple-optimized fast paths. Custom kernels add dispatch overhead.

---

## Optimization Lab

ZMLX includes a local MLX fork (`mlx_local/`) where we prototype fused C++ Metal primitives that need access to MLX's internal quantized matmul infrastructure. These are experimental and require building MLX from source.

### What's in the lab

| Primitive | Status | What it does |
|:--|:--|:--|
| `gather_qmm_swiglu` | Working, benchmarked | Fused gate+up+SwiGLU for MoE experts |
| `add_rms_norm` | Planned | Fused residual add + RMSNorm |
| `gather_qmm_combine` | Planned | Fused down projection + weighted expert sum |

### Building the local MLX fork

```bash
# From the ZMLX repo root
CMAKE_ARGS='-DMETAL_CPP_URL=file:///path/to/ZMLX/mlx_local/third_party/metal-cpp_26.zip \
  -DMLX_METAL_MODULE_CACHE=/tmp/clang_module_cache \
  -DFETCHCONTENT_SOURCE_DIR_JSON=/path/to/ZMLX/mlx_local/third_party/json \
  -DFETCHCONTENT_SOURCE_DIR_FMT=/path/to/ZMLX/mlx_local/third_party/fmt \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND=/path/to/ZMLX/mlx_local/third_party/nanobind \
  -DMLX_BUILD_GGUF=OFF' \
pip install -e mlx_local --no-build-isolation
```

ZMLX auto-detects the fused primitives at runtime. If you're on stock MLX, `patch(model)` uses the standard two-pass path — for MoE models, prefer `smart_patch` to avoid gating+combine slowdowns.

### The plan

Prototype here, validate with benchmarks, upstream to MLX via PR. Once primitives land in stock MLX, ZMLX auto-detects them and everyone benefits via `pip install mlx`. ZMLX is the incubator, MLX is the distribution.

---

## Precision

All ZMLX Metal kernels compute internally in **float32** regardless of input dtype. The `compute_dtype` parameter accepted by many kernel functions is **deprecated** and will be removed in a future release. Passing a non-None value will emit a `DeprecationWarning`.

---

## Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) - 5-minute tutorial
- [`docs/COOKBOOK.md`](docs/COOKBOOK.md) - Recipes for common patterns
- [`docs/KERNELS.md`](docs/KERNELS.md) - Complete kernel catalog reference
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) - Design philosophy

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and conventions.

---

## License

MIT. See [`LICENSE`](LICENSE).
