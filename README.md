# ZMLX - Faster LLM inference on Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

**Speed up MoE model inference on your Mac with one line of code.** ZMLX patches [MLX](https://github.com/ml-explore/mlx) models with fused Metal kernels — no model conversion, no config changes, token-identical output.

```bash
pip install zmlx
```

```python
import mlx_lm
from zmlx.patch import patch

model, tokenizer = mlx_lm.load("mlx-community/LFM2-8B-A1B-4bit")
patch(model)  # +9-12% decode, token-identical output

text = mlx_lm.generate(model, tokenizer,
    prompt="Explain mixture-of-experts in one paragraph.",
    max_tokens=200)
```

---

## LFM2 on Apple Silicon

[LFM2-8B-A1B](https://huggingface.co/LiquidAI/LFM2-8B-A1B) is Liquid AI's 8B-parameter MoE model (32 experts, top-4 routing, 1B active per token). It runs entirely on-device on any 16 GB M-series Mac. ZMLX makes it faster.

### Benchmarks

#### M1 Pro 16 GB

> macOS 14.6.1 · MLX 0.30.4 · ZMLX 0.7.11 · Python 3.10.0 · commit `7de879e`
>
> **Method:** `python -m zmlx.validate` — greedy decode (`temp=0`), fixed prompt, 5 runs × 500 tokens, median reported. Baseline is unpatched `mlx_lm`; patched adds `patch(model)`.
>
> **Repro capsule:** [`benchmarks/results/lfm2_m1pro_20260131.json`](benchmarks/results/lfm2_m1pro_20260131.json)

**LFM2-8B-A1B-4bit** ([mlx-community/LFM2-8B-A1B-4bit](https://huggingface.co/mlx-community/LFM2-8B-A1B-4bit))

| Metric | Baseline | Patched | Change |
|:--|--:|--:|:--|
| Decode | 105.5 tok/s | 115.3 tok/s | **+9.3%** |
| Prefill | 225.4 tok/s | 227.1 tok/s | +0.8% (neutral) |
| Fidelity | — | 430/430 | token-identical |
| Peak memory | — | 5.3 GB | |

**LFM2-8B-A1B-8bit** ([mlx-community/LFM2-8B-A1B-8bit-MLX](https://huggingface.co/mlx-community/LFM2-8B-A1B-8bit-MLX))

| Metric | Baseline | Patched | Change |
|:--|--:|--:|:--|
| Decode | 72.8 tok/s | 76.4 tok/s | **+5.0%** |
| Prefill | 180.5 tok/s | 182.8 tok/s | +1.3% (neutral) |
| Fidelity | — | 500/500 | token-identical |
| Peak memory | — | 9.5 GB | |

#### M4 Max 36 GB

> macOS 15.x · MLX 0.30.4.dev · ZMLX 0.7.11 · 3 runs × 200 tokens, median reported.

**LFM2-8B-A1B-4bit**

| Metric | Baseline | Patched | Change |
|:--|--:|--:|:--|
| Decode | 223 tok/s | 251 tok/s | **+12%** |
| Prefill | 734 tok/s | 746 tok/s | +1.6% (neutral) |
| Fidelity | — | 200/200 | token-identical |

**LFM2-8B-A1B-8bit**

| Metric | Baseline | Patched | Change |
|:--|--:|--:|:--|
| Decode | 153 tok/s | 168 tok/s | **+10%** |
| Prefill | 552 tok/s | 570 tok/s | +3.3% (neutral) |
| Fidelity | — | 200/200 | token-identical |

#### Notes

Both variants fit on 16 GB Macs. Output is **token-identical** to unpatched `mlx_lm`.

Prefill throughput is neutral on both devices. The fused kernels are guarded to activate only at sequence length M<=32 (decode), so prefill takes the standard MLX code path. An earlier single-run measurement showed an anomalous baseline prefill spike (255+ tok/s on M1 Pro 4bit) that was not reproducible under controlled conditions — repeated 5-run medians confirmed prefill within noise of baseline. See the repro capsule for raw per-run data.

### How it works

`patch(model)` walks the model tree, detects the architecture, and replaces matching layers with fused Metal kernel equivalents. No weights are modified — the same quantized matrices are read with fewer kernel dispatches.

```python
patch(model)
# [zmlx.patch] Applying 3 patterns: ['swiglu_mlp', 'geglu_mlp', 'moe_mlp']
# Patched 24 modules:
#   moe_mlp: 22
#   swiglu_mlp: 2
```

### Validate on your hardware

```bash
pip install zmlx
python -m zmlx.validate mlx-community/LFM2-8B-A1B-4bit --max-tokens 500 --runs 5
```

This loads the model twice (baseline then patched), generates tokens with greedy decode, and compares token-for-token fidelity plus throughput.

---

## What else ZMLX does

ZMLX is also a Metal kernel toolkit for MLX:

- **70+ kernel catalog** — SwiGLU, GeGLU, fused dropout, MoE gating, RMSNorm, RoPE, quantization
- **One-line kernel authoring** — `elementwise("x * tanh(log(1 + exp(x)))")` compiles to Metal
- **Automatic gradients** — custom VJP backward passes as Metal kernels via `mx.custom_function`
- **Model patching** — `patch(model)` for MoE models (LFM2, Qwen3-MoE, Mixtral, GPT-OSS)

**Custom kernel example:**

```python
from zmlx.api import elementwise
import mlx.core as mx

mish = elementwise("x * tanh(log(1 + exp(x)))", name="mish")
y = mish(mx.random.normal((1024,)))
```

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

## Kernel Catalog

70+ kernels organized by domain. Full reference: [`docs/KERNELS.md`](docs/KERNELS.md).

| Module | Highlights |
|:---|:---|
| `loss` | `softmax_cross_entropy` — memory-efficient fused loss |
| `transformer` | `swiglu`, `geglu`, `rmsnorm_residual`, `dropout` — genuine fusions |
| `bits` | `pack_bits`, `unpack_bits` — no MLX equivalent |
| `moe` | `topk_gating_softmax`, `moe_dispatch`, `moe_combine` — fused expert routing |
| `quant` | FP8, NF4, int8, int4 dequantization |
| `optimizers` | `adamw_step` — fused parameter update |
| `norms` | `rmsnorm`, `layernorm` — float32 internal compute |
| `rope` | `apply_rope`, `apply_rope_interleaved`, `apply_gqa_rope` |

---

## Benchmarks

### Op-level (B=16, S=1024, D=1024, float16, M4 Max)

`python benchmarks/microbench.py`:

| Operation | MLX | ZMLX | Speedup |
|:--|--:|--:|:--|
| **SwiGLU** | 0.87 ms | **0.43 ms** | **2.0x** |
| **Dropout** | 3.08 ms | **0.41 ms** | **7.5x** |
| **Top-K** | 1.81 ms | **0.49 ms** | **3.7x** |
| **Gather-Add** | 0.55 ms | **0.42 ms** | **1.3x** |
| Softmax | 0.45 ms | 0.44 ms | ~1.0x |
| RMSNorm | 0.51 ms | 0.54 ms | 0.95x |

ZMLX is most effective for **fused operations** that MLX doesn't provide as single ops. MLX built-ins (`mx.fast.rms_norm`, `mx.softmax`) are already highly optimized.

### Model-level (E2E)

Baselines are unpatched `mlx_lm`. ZMLX rows add `patch(model)`. Same weights, same quantization, same prompt.

#### LFM2-8B-A1B (best supported)

See [LFM2 benchmarks above](#benchmarks). Decode: **+9% on M1 Pro, +12% on M4 Max**, prefill neutral, token-identical on both variants.

#### Other MoE models

**Qwen3-30B-A3B-4bit** (E=128, K=8, M4 Max):

| Config | Decode (tok/s) | vs Baseline |
|:--|--:|:--|
| Baseline | 113-114 | — |
| `patch(model)` + fused SwiGLU | 119-122 | **+5-8%** |

> Fused SwiGLU requires a [local MLX build](#optimization-lab). Token fidelity not guaranteed on Qwen3-MoE.

#### Dense models (neutral)

**Qwen3-4B-4bit** (dense): 124.4 -> 125.3 tok/s (1.01x). Dense decode is bandwidth-bound; patches are safe but not expected to help.

### Kernel-level MoE layer timing

`python benchmarks/bench_moe_layer.py` — single MoE layer, 500 iterations, p50 median.

**LFM2-8B-A1B-4bit** (E=32, K=4):

| seq_len | Baseline | Patched | Speedup |
|:--|--:|--:|:--|
| 1 (decode) | 293 us | 282 us | **1.04x** |
| 4 | 531 us | 498 us | **1.07x** |
| 16 | 998 us | 949 us | 1.05x |
| 64 | 2291 us | 2452 us | 0.93x |

Fused kernels help at decode (small M) and regress at prefill (large M). The patch auto-selects: fused for M<=32, standard path otherwise.

---

## Model Support

`patch(model)` auto-detects the model family and applies only validated patterns:

| Model | Patterns applied | Decode speedup | Token fidelity |
|:--|:--|:--|:--|
| **LFM2-8B-A1B-4bit** | `moe_mlp` + `swiglu_mlp` | **+9-12%** | token-identical |
| **LFM2-8B-A1B-8bit** | `moe_mlp` + `swiglu_mlp` | **+5-10%** | token-identical |
| Qwen3-30B-A3B-4bit | *auto-excluded* | — | diverges at token 0 |
| Qwen3-4B-4bit | *auto-excluded* | — | diverges at token 18 |
| Llama-3.2-1B-4bit | neutral | 0.98x | token-identical |

For unlisted models, validate first: `python -m zmlx.validate <model>`.

### Patching options

```python
from zmlx.patch import patch, smart_patch

patch(model)                       # auto-detect model, apply safe defaults
patch(model, patterns=["moe_mlp"]) # force specific pattern
patch(model, mode="training")      # add norm fusions for backward pass

# Auto-benchmark: apply only patterns that actually help
sample = mx.array([tokenizer.encode("Hello")])
model = smart_patch(model, sample)
```

---

## Optimization Lab

ZMLX includes a local MLX fork (`mlx_local/`) for prototyping fused C++ Metal primitives that need access to MLX internals. These require building MLX from source and are intended for eventual upstream contribution.

| Primitive | Status | Description |
|:--|:--|:--|
| `gather_qmm_swiglu` | Working | Fused gate+up+SwiGLU for MoE experts |
| `add_rms_norm` | Planned | Fused residual add + RMSNorm |
| `gather_qmm_combine` | Planned | Fused down projection + weighted expert sum |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for build instructions and design details.

---

## Precision

All Metal kernels compute internally in **float32** regardless of input dtype.

---

## Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md) — 5-minute tutorial
- [`docs/COOKBOOK.md`](docs/COOKBOOK.md) — Recipes for common patterns
- [`docs/KERNELS.md`](docs/KERNELS.md) — Complete kernel catalog
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Design philosophy

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and conventions.

---

## License

MIT. See [`LICENSE`](LICENSE).
