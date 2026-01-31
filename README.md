# ZMLX — Faster MoE inference on Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

ZMLX patches [MLX](https://github.com/ml-explore/mlx) models with fused Metal kernels for faster Mixture-of-Experts decode. No model conversion, no config changes, token-identical output.

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

## Benchmarks

### LFM2-8B-A1B on M1 Pro 16 GB

> macOS 14.6.1 · MLX 0.30.4 · ZMLX 0.7.11 · Python 3.10.0 · commit `7de879e`
>
> **Method:** `python -m zmlx.validate` — greedy decode (`temp=0`), fixed prompt, 5 runs × 500 tokens, median reported. Baseline is unpatched `mlx_lm`; patched adds `patch(model)`.
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

> macOS 15.x · MLX 0.30.4.dev · ZMLX 0.7.11 · 3 runs × 200 tokens, median reported.

| Variant | Decode baseline | Decode patched | Change | Fidelity |
|:--|--:|--:|:--|:--|
| 4-bit | 223 tok/s | 251 tok/s | **+12%** | 200/200 |
| 8-bit | 153 tok/s | 168 tok/s | **+10%** | 200/200 |

Prefill neutral on both variants (+1.6% and +3.3%, within noise).

### Notes on methodology

Prefill throughput is neutral by design — fused kernels are guarded to activate only at sequence length M <= 32 (decode), so prefill takes the standard MLX code path. An earlier measurement showed an anomalous baseline prefill spike (255+ tok/s on M1 Pro 4bit) that was not reproducible under controlled conditions. Repeated 5-run medians confirmed prefill within noise. Raw per-run data is in the repro capsule.

---

## Correctness

Token fidelity is a first-class requirement, not an afterthought.

**What "PASS" means:** Given the same weights, the same prompt, and greedy decoding (`temp=0`), the patched model produces the exact same token sequence as the unpatched model. `python -m zmlx.validate` compares token IDs one-by-one.

**Auto-exclusion:** When `patch()` detects a model family with known fidelity issues (e.g. Qwen3-MoE diverges at token 0), it auto-excludes the problematic patterns and prints a warning. No silent correctness loss.

**Override with caution:** You can force any pattern with `patch(model, patterns=[...])`, but always validate first:

```bash
python -m zmlx.validate <model> --max-tokens 500 --runs 5
```

---

## Model Support

### Stable

Token-identical output, measurable decode improvement. Safe to use without further validation.

| Model | Decode speedup | Fidelity | Patterns |
|:--|:--|:--|:--|
| **LFM2-8B-A1B-4bit** | **+9-12%** | token-identical | `moe_mlp` + `swiglu_mlp` |
| **LFM2-8B-A1B-8bit** | **+5-10%** | token-identical | `moe_mlp` + `swiglu_mlp` |

### Experimental

Speedups may exist but token parity is not guaranteed. Auto-excluded by `patch()` defaults.

| Model | Status | Notes |
|:--|:--|:--|
| Qwen3-30B-A3B-4bit | diverges at token 0 | Requires local MLX build for fused SwiGLU |
| Qwen3-4B-4bit (dense) | diverges at token 18 | Dense model, patches not expected to help |
| Llama-3.2-1B-4bit | 0.98x (neutral) | Dense model, bandwidth-bound |

For unlisted models: `python -m zmlx.validate <model>`.

---

## How It Works

`patch(model)` walks the model tree, detects the architecture, and replaces matching layers with fused Metal kernel equivalents. No weights are modified — the same quantized matrices are read with fewer kernel dispatches.

```python
patch(model)
# [zmlx.patch] Applying 3 patterns: ['swiglu_mlp', 'geglu_mlp', 'moe_mlp']
# Patched 24 modules:
#   moe_mlp: 22
#   swiglu_mlp: 2
```

The MoE speedup comes from two fused kernels:

1. **Fused gating** — top-k softmax in a single Metal dispatch (replaces softmax + argpartition + gather + normalize)
2. **Fused expert combine** — weighted sum of expert outputs in one pass (replaces element-wise multiply + reduce)

Both kernels are guarded with a sequence length threshold (`M <= 32`). At larger M (prefill), the standard MLX path runs unchanged.

### Patching options

```python
from zmlx.patch import patch, smart_patch

patch(model)                       # auto-detect, apply safe defaults
patch(model, patterns=["moe_mlp"]) # force specific pattern
patch(model, mode="training")      # add norm fusions for backward pass

# Auto-benchmark: apply only patterns that actually help
sample = mx.array([tokenizer.encode("Hello")])
model = smart_patch(model, sample)
```

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
| `gather_qmm_swiglu` | Working | Fused gate+up+SwiGLU for MoE experts |
| `add_rms_norm` | Planned | Fused residual add + RMSNorm |
| `gather_qmm_combine` | Planned | Fused down projection + weighted expert sum |

---

## Precision

All Metal kernels compute internally in **float32** regardless of input dtype.

---

## Documentation

- [`docs/TOUR.md`](docs/TOUR.md) — Quick walkthrough and orientation
- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 5-minute tutorial
- [`docs/COOKBOOK.md`](docs/COOKBOOK.md) — Recipes for common patterns
- [`docs/KERNELS.md`](docs/KERNELS.md) — Complete kernel catalog
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Design philosophy
- [`UPSTREAM_PLAN.md`](UPSTREAM_PLAN.md) — What belongs upstream in MLX

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and conventions.

---

## License

MIT. See [`LICENSE`](LICENSE).
