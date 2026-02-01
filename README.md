# ZMLX — Faster MoE inference on Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

ZMLX patches [MLX](https://github.com/ml-explore/mlx) models with fused Metal kernels for faster Mixture-of-Experts decode on Apple Silicon. No model conversion, no config changes — just `pip install zmlx` and `patch(model)`.

```python
import mlx_lm
from zmlx.patch import patch

model, tokenizer = mlx_lm.load("mlx-community/LFM2-8B-A1B-4bit")
patch(model)

text = mlx_lm.generate(model, tokenizer,
    prompt="Explain mixture-of-experts in one paragraph.",
    max_tokens=200)
```

---

## Two modes

| Mode | What you need | What you get |
|:--|:--|:--|
| **Stable** (default) | `pip install zmlx` + stock MLX | Token-identical output, +5-12% decode on LFM2 |
| **Fast** (opt-in) | Custom MLX fork with `mx.gather_qmm_swiglu` | Additional fused SwiGLU for MoE experts; faster on Qwen3 but may diverge on token fidelity |

`patch()` auto-detects which primitives are available and only enables what is safe. On stock MLX, Qwen3-MoE patterns are auto-excluded to prevent silent correctness loss.

---

## Benchmarks — Stable mode (stock MLX)

### LFM2-8B-A1B on M1 Pro 16 GB

> macOS 14.6.1 · MLX 0.30.4 · ZMLX 0.7.11 · Python 3.10.0 · commit `7de879e`
>
> **Method:** `python -m zmlx.validate` — greedy decode (`temp=0`), fixed prompt, 5 runs, `max_tokens=500` (4-bit runs terminated at 430 tokens due to EOS), median reported. Baseline is unpatched `mlx_lm`; patched adds `patch(model)`.
>
> **Repro capsule:** [`benchmarks/repro_capsules/lfm2_m1pro_20260131.json`](benchmarks/repro_capsules/lfm2_m1pro_20260131.json) · **Print report:** `python -m zmlx.bench.report <capsule.json>`

**4-bit** ([mlx-community/LFM2-8B-A1B-4bit](https://huggingface.co/mlx-community/LFM2-8B-A1B-4bit))

| Metric | Baseline | Patched | Change |
|:--|--:|--:|:--|
| Decode | 105.5 tok/s | 115.3 tok/s | **+9.3%** |
| Prefill | 225.4 tok/s | 227.1 tok/s | +0.8% (neutral) |
| Fidelity | — | 430/430 | token-identical |
| Peak memory | — | 5.3 GB | |

**8-bit** ([mlx-community/LFM2-8B-A1B-8bit-MLX](https://huggingface.co/mlx-community/LFM2-8B-A1B-8bit-MLX))

| Metric | Baseline | Patched | Change |
|:--|--:|--:|:--|
| Decode | 72.8 tok/s | 76.4 tok/s | **+5.0%** |
| Prefill | 180.5 tok/s | 182.8 tok/s | +1.3% (neutral) |
| Fidelity | — | 500/500 | token-identical |
| Peak memory | — | 9.5 GB | |

### LFM2-8B-A1B on M4 Max 36 GB

> macOS 26.1 · MLX 0.30.1 · ZMLX 0.7.11 · Python 3.12 · commit `139993e`
>
> **Method:** `python -m zmlx.validate` — greedy decode (`temp=0`), fixed prompt, 5 runs, `max_tokens=500` (4-bit runs terminated at 430 tokens due to EOS), median reported. Baseline is unpatched `mlx_lm`; patched adds `patch(model)`.
>
> **Repro capsule:** [`benchmarks/repro_capsules/lfm2_m4max_20260131.json`](benchmarks/repro_capsules/lfm2_m4max_20260131.json) · **Print report:** `python -m zmlx.bench.report <capsule.json>`

**4-bit** ([mlx-community/LFM2-8B-A1B-4bit](https://huggingface.co/mlx-community/LFM2-8B-A1B-4bit))

| Metric | Baseline | Patched | Change |
|:--|--:|--:|:--|
| Decode | 223.7 tok/s | 250.3 tok/s | **+11.9%** |
| Prefill | 737.4 tok/s | 755.4 tok/s | +2.4% (neutral) |
| Fidelity | — | 430/430 | token-identical |
| Peak memory | — | 5.30 GB | |

**8-bit** ([mlx-community/LFM2-8B-A1B-8bit-MLX](https://huggingface.co/mlx-community/LFM2-8B-A1B-8bit-MLX))

| Metric | Baseline | Patched | Change |
|:--|--:|--:|:--|
| Decode | 152.5 tok/s | 164.3 tok/s | **+7.7%** |
| Prefill | 557.6 tok/s | 564.4 tok/s | +1.2% (neutral) |
| Fidelity | — | 500/500 | token-identical |
| Peak memory | — | 9.45 GB | |

### Benchmarks — Fast mode (custom MLX fork)

> Requires a local MLX build that exposes `mx.gather_qmm_swiglu`. This fuses the gate+up projections and SwiGLU activation into a single kernel for quantized MoE experts. Check availability: `python -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))"`.

**Qwen3-30B-A3B (max_tokens=500, runs=3):**

| Config | Decode speedup | Fidelity | Notes |
|:--|:--|:--|:--|
| Stock MLX, auto-excluded patches | 1.007x (base) / 0.979x (instruct) | PASS (500/500) | Safe but no MoE gain — `patch()` auto-excludes `moe_mlp` on Qwen3 |
| Dev MLX + `moe_mlp` forced | **+6.9%** (base) / **+8.9%** (instruct) | FAIL | Diverges at token 6 (base) / token 146 (instruct) |

The Qwen3 fidelity failure comes from precision differences in the fused `gather_qmm_swiglu` kernel (likely float16 internal accumulation vs MLX's default primitives). This is a known issue we plan to fix by switching the kernel to float32 accumulation. Until then, Qwen3 gains require explicit opt-in and are not enabled by default.

### Notes on methodology

Prefill throughput is neutral by design — fused kernels are guarded to activate only at sequence length M <= 32 (decode), so prefill takes the standard MLX code path. Raw per-run data is in the repro capsules (`benchmarks/repro_capsules/`).

---

## How It Works

### The problem: dispatch overhead in MoE decode

In Mixture-of-Experts models, each token is routed to a subset of expert networks. During decode (generating one token at a time), the computation per expert is small — a few matrix multiplies on a single row vector. But the standard inference path dispatches multiple Metal kernels per expert per layer:

1. **Gating:** `softmax(logits)` → `argpartition` → `gather` → `normalize` — 4 dispatches
2. **Expert execution:** gate projection, up projection, SwiGLU activation, down projection — per expert
3. **Combine:** element-wise multiply by gating weights → reduce-sum across experts — 2 dispatches

On Apple Silicon, each Metal kernel dispatch has fixed overhead (command buffer encoding, GPU scheduling). When the actual compute per dispatch is small — as it is for M=1 decode — this overhead dominates. The GPU spends more time waiting between kernels than doing math.

### What ZMLX fuses

ZMLX replaces the multi-dispatch sequences with single Metal kernels that do the same math in one pass. All fused kernels are generated from Python via [`mx.fast.metal_kernel`](https://ml-explore.github.io/mlx/build/html/python/fast.html) — no changes to MLX core required.

**Fused top-k gating softmax** (`topk_gating_softmax`):

Replaces the 4-dispatch gating sequence with a single kernel. For small expert counts (D <= 32, common in MoE), the kernel uses SIMD group operations — each row is processed by one SIMD group (32 threads), with `simd_max` and `simd_sum` for the softmax reduction and a register-based insertion sort for top-k selection. For larger D, a threadgroup reduction with shared memory is used. The kernel computes softmax probabilities and selects the top-k experts with their normalized weights in one pass.

**Fused expert combine** (`moe_combine`):

Replaces the separate element-wise multiply and reduce-sum with a single kernel that reads each expert output once, multiplies by its gating weight, and accumulates the weighted sum in float32. Output shape goes directly from `(B, K, D)` to `(B, D)` without materializing the intermediate `weights * expert_outputs` tensor.

**Fused gate+up+SwiGLU** (`gather_qmm_swiglu`, fast mode only):

A C++ Metal primitive (in `mlx_local/`) that fuses the gate projection, up projection, and SwiGLU activation for quantized MoE experts into a single kernel launch. Instead of reading the input vector twice (once for gate, once for up), it reads once and produces the activated output directly. This requires access to MLX's internal quantized matmul infrastructure and is the only ZMLX optimization that needs a custom MLX build.

### Why prefill is unaffected

All fused kernels are guarded with a sequence length check (`M <= 32`). During prefill, M equals the prompt length (typically hundreds or thousands of tokens). At this scale, the compute-to-dispatch ratio is high and the standard MLX path is already efficient. The guards ensure ZMLX never regresses prefill performance.

### Correctness guarantee

Token fidelity is a first-class requirement. `patch()` auto-detects the model family and excludes patterns with known fidelity issues. The fused gating kernel reproduces the exact same top-k selection and softmax normalization as the reference MLX ops. The combine kernel accumulates in float32 (or dtype-matched for Qwen3's `moe_combine_exact`). `python -m zmlx.validate` compares every generated token ID between patched and unpatched models under greedy decoding.

### Patching options

```python
from zmlx.patch import patch, smart_patch

patch(model)                       # auto-detect, apply safe defaults
patch(model, patterns=["moe_mlp"]) # force specific pattern (overrides safety)
patch(model, mode="training")      # add norm fusions for backward pass

# Auto-benchmark: apply only patterns that actually help
sample = mx.array([tokenizer.encode("Hello")])
model = smart_patch(model, sample)
```

---

## Model Support

### Stable

Token-identical output, measurable decode improvement. Safe to use without further validation.

| Model | Decode speedup | Fidelity | Patterns |
|:--|:--|:--|:--|
| **LFM2-8B-A1B-4bit** | **+9-12%** | token-identical | `moe_mlp` + `swiglu_mlp` |
| **LFM2-8B-A1B-8bit** | **+5-8%** | token-identical | `moe_mlp` + `swiglu_mlp` |

### Fast (requires custom MLX)

Requires a local MLX build with `mx.gather_qmm_swiglu`. Speedups exist but token parity is not guaranteed. Auto-excluded by `patch()` defaults — force with `patch(model, patterns=["moe_mlp"])`.

| Model | Decode speedup | Fidelity | Notes |
|:--|:--|:--|:--|
| Qwen3-30B-A3B-4bit | +6.9% | diverges at token 6 | Fused SwiGLU precision mismatch |
| Qwen3-30B-A3B-Instruct-2507-4bit | +8.9% | diverges at token 146 | Fused SwiGLU precision mismatch |

### Tested (no gain)

| Model | Status | Notes |
|:--|:--|:--|
| LFM2.5-1.2B-Thinking-MLX-8bit | 0.997x, PASS | Dense model, no matched MoE patterns |
| Qwen3-4B-4bit (dense) | diverges at token 18 | Dense model, patches not expected to help |
| Llama-3.2-1B-4bit | 0.98x, PASS | Dense model, bandwidth-bound |

For unlisted models: `python -m zmlx.validate <model>`.

---

## Toolkit

ZMLX is also a Metal kernel authoring toolkit for MLX:

- **70+ kernel catalog** — SwiGLU, GeGLU, fused dropout, MoE gating, RMSNorm, RoPE, quantization
- **One-line kernel authoring** — `elementwise("x * tanh(log(1 + exp(x)))")` compiles to Metal
- **Automatic gradients** — custom VJP backward passes as Metal kernels via `mx.custom_function`
- **Benchmarking** — `zmlx.bench.compare()` for side-by-side timing, `zmlx.bench.report` for repro capsules

```python
from zmlx.api import elementwise
import mlx.core as mx

mish = elementwise("x * tanh(log(1 + exp(x)))", name="mish")
y = mish(mx.random.normal((1024,)))
```

### Op-level benchmarks

B=16, S=1024, D=1024, float16, M4 Max. `python benchmarks/microbench.py`:

| Operation | MLX | ZMLX | Speedup |
|:--|--:|--:|:--|
| **SwiGLU** | 0.87 ms | **0.43 ms** | **2.0x** |
| **Dropout** | 3.08 ms | **0.41 ms** | **7.5x** |
| **Top-K** | 1.81 ms | **0.49 ms** | **3.7x** |
| **Gather-Add** | 0.55 ms | **0.42 ms** | **1.3x** |
| Softmax | 0.45 ms | 0.44 ms | ~1.0x |
| RMSNorm | 0.51 ms | 0.54 ms | 0.95x |

ZMLX helps most for **fused operations** that MLX doesn't provide as single ops. MLX built-ins (`mx.fast.rms_norm`, `mx.softmax`) are already highly optimized.

### Kernel catalog

70+ kernels organized by domain. Full reference: [`docs/KERNELS.md`](docs/KERNELS.md).

| Module | Highlights |
|:---|:---|
| `moe` | `topk_gating_softmax`, `moe_dispatch`, `moe_combine` — fused expert routing |
| `transformer` | `swiglu`, `geglu`, `rmsnorm_residual`, `dropout` — genuine fusions |
| `loss` | `softmax_cross_entropy` — memory-efficient fused loss |
| `bits` | `pack_bits`, `unpack_bits` — no MLX equivalent |
| `quant` | FP8, NF4, int8, int4 dequantization |
| `norms` | `rmsnorm`, `layernorm` — float32 internal compute |
| `rope` | `apply_rope`, `apply_rope_interleaved`, `apply_gqa_rope` |
| `optimizers` | `adamw_step` — fused parameter update |

---

## Install

**Requirements**: macOS (Apple Silicon), Python >= 3.10, mlx >= 0.30.0

```bash
pip install zmlx
```

From source:

```bash
git clone https://github.com/Hmbown/ZMLX.git
cd ZMLX
pip install -e ".[dev]"
```

---

## Quick Start

### Custom elementwise kernel

```python
from zmlx.api import elementwise
import mlx.core as mx

# Non-differentiable
fast_exp = elementwise("metal::exp(x)", name="fast_exp")
y = fast_exp(mx.random.normal((1024,)))

# Differentiable with custom VJP
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

### Custom reduction

```python
from zmlx.api import reduce
import mlx.core as mx

my_sum = reduce(init="0.0f", update="acc + v", name="row_sum")
y = my_sum(mx.random.normal((8, 1024)))  # shape (8,)
```

### Two-pass map-reduce (softmax pattern)

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

### Test and benchmark

```python
import zmlx
import mlx.core as mx

zmlx.testing.assert_matches(
    my_softmax, lambda x: mx.softmax(x, axis=-1),
    shapes=[(8, 1024), (32, 4096)],
)

zmlx.bench.compare(
    {"ZMLX": my_softmax, "MLX": lambda x: mx.softmax(x, axis=-1)},
    shapes=[(1024, 4096), (4096, 4096)],
)
```

---

## Optimization Lab

ZMLX includes a local MLX fork (`mlx_local/`) for prototyping fused C++ Metal primitives that need access to MLX internals. These are intended for eventual upstream contribution — see [`UPSTREAM_PLAN.md`](UPSTREAM_PLAN.md).

| Primitive | Status | Description |
|:--|:--|:--|
| `gather_qmm_swiglu` | Working | Fused gate+up+SwiGLU for quantized MoE experts |
| `gather_qmm_combine` | Working | Fused down projection + weighted expert sum |
| `add_rms_norm` | Planned | Fused residual add + RMSNorm |

---

## Precision

All Python-level Metal kernels compute internally in **float32** regardless of input dtype. The C++ `gather_qmm_swiglu` primitive currently uses the default MLX quantized matmul precision, which may differ — this is the source of the Qwen3 fidelity issue and a target for improvement.

---

## Documentation

- [`docs/TOUR.md`](docs/TOUR.md) — Quick walkthrough and orientation
- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 5-minute tutorial
- [`docs/COOKBOOK.md`](docs/COOKBOOK.md) — Recipes for common patterns
- [`docs/KERNELS.md`](docs/KERNELS.md) — Complete kernel catalog
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Design philosophy
- [`UPSTREAM_PLAN.md`](UPSTREAM_PLAN.md) — What belongs upstream in MLX

---

## Acknowledgments

ZMLX is built on [MLX](https://github.com/ml-explore/mlx) by Apple machine learning research. If you use ZMLX in your work, please also cite MLX:

```bibtex
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and conventions.

---

## License

MIT. See [`LICENSE`](LICENSE).
