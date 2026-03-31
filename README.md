# ZMLX

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Faster local LLM inference on Apple Silicon.** ZMLX patches [MLX](https://github.com/ml-explore/mlx) models with fused Metal kernels to speed up decode. Two lines of code, token-identical output.

```python
from zmlx.patch import patch
patch(model)  # that's it
```

## Benchmarks

Measured on M4 Max 36GB, greedy decoding. All patched outputs are **token-identical** to unpatched.

### Qwen3.5 (Dense)

| Model | Speedup | Modules Patched |
|:--|--:|:--|
| **Qwen3.5-9B-4bit** | **+7.5%** | 56 (deltanet + swiglu_mlp) |
| Qwen3.5-0.8B-4bit | +5.2% | 42 (deltanet + swiglu_mlp) |
| Qwen3.5-2B-4bit | +4.5% | 42 (deltanet + swiglu_mlp) |
| Qwen3.5-4B-4bit | +2.9% | 56 (deltanet + swiglu_mlp) |
| Qwen3.5-27B-4bit | ~neutral | 112 (deltanet + swiglu_mlp) |

Speedup comes from the **fused post-conv DeltaNet decode kernel**, which fuses qkv split, Q/K RMSNorm, g/beta computation, and recurrent state update into a single Metal dispatch per layer. Enabled by default since v0.11.

### Qwen3.5 (MoE)

| Model | Speedup | Modules Patched |
|:--|--:|:--|
| Qwen3.5-35B-A3B-4bit | ~+2% | 70 (deltanet + moe_mlp) |

### LFM2 (MoE)

| Model | Speedup | Notes |
|:--|--:|:--|
| **LFM2-8B-A1B-4bit** | **+12.8%** | Best overall result. Stock MLX. |
| **LFM2-24B-A2B-4bit** | **+6.0%** | D-SIMD gate kernel. Stock MLX. |

### Other Models

| Model | Speedup | Notes |
|:--|--:|:--|
| GPT-OSS-20B-4bit | +1.0% | Stock MLX. |
| GLM-4.7-Flash-4bit | +6.4% | Requires [custom MLX build](docs/EXPERIMENTAL_MLX.md). |
| Llama (dense) | swiglu_mlp | Dispatch fusion applied. |
| DeepSeek-V3 / Kimi-K2.5 | patterns apply | Untested at full scale. |
| Other models | 0% | Safe no-op. |

Raw data and methodology in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Repro capsules in `benchmarks/repro_capsules/`.

## Install

```bash
pip install "zmlx[lm]"
```

Requires macOS 14+ on Apple Silicon, Python 3.10+, MLX 0.30+.

## Quick Start

```python
import mlx_lm
from zmlx.patch import patch

model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-9B-OptiQ-4bit")
patch(model)

print(mlx_lm.generate(model, tokenizer, prompt="Hello!", max_tokens=200))
```

Verify on your hardware:

```bash
python -m zmlx.validate mlx-community/Qwen3.5-9B-OptiQ-4bit --max-tokens 200 --runs 3
```

## Serving (Ollama Alternative)

ZMLX includes an OpenAI-compatible API server that applies patches automatically. Drop-in replacement for Ollama on Apple Silicon with kernel speedups:

```bash
python integrations/ollama_compat/serve.py \
    --model mlx-community/Qwen3.5-9B-OptiQ-4bit \
    --port 8080
```

Then use any OpenAI-compatible client:

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "default_model", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Works with Open WebUI, Continue, LangChain, and anything else that speaks the OpenAI API. See [`integrations/ollama_compat/README.md`](integrations/ollama_compat/README.md) for setup guides.

## How It Works

LLM decode on Apple Silicon is dispatch-bound: the GPU spends more time launching small Metal kernels than doing compute. ZMLX fuses sequences of operations (conv1d + silu, gating + combine + activation, qkv split + norm + state update) into single dispatches, cutting overhead.

**Fused paths only activate during decode** (sequence length <= 32). Prefill runs the standard MLX code path unchanged. `patch()` is always safe to call -- if no patterns match, the model runs unmodified.

### Pattern Catalog

| Pattern | What It Fuses | Target |
|:--|:--|:--|
| `deltanet` | conv1d+silu, post-conv qkv+norm+state update | Qwen3.5 GatedDeltaNet layers |
| `moe_mlp` | MoE gate+combine+SwiGLU expert dispatch | MoE models (LFM2, Qwen3.5-A3B, GLM) |
| `swiglu_mlp` | Dense SwiGLU (gate+up+activation) | Dense models (Llama, Qwen3.5) |
| `geglu_mlp` | Dense GeGLU activation fusion | GeGLU-based models |
| `rmsnorm` | RMSNorm kernel replacement | All (usually neutral) |
| `layernorm` | LayerNorm kernel replacement | All (usually neutral) |
| `softmax` | Softmax kernel replacement | All (usually neutral) |
| `residual_norm` | Fused residual + norm | All (experimental) |

## Docs

| | |
|:--|:--|
| [Tour](docs/TOUR.md) | Walkthrough and how to verify results |
| [Quickstart](docs/QUICKSTART.md) | 5-minute kernel authoring tutorial |
| [Cookbook](docs/COOKBOOK.md) | Recipes for common patterns |
| [Kernels](docs/KERNELS.md) | Full kernel catalog (70+ kernels) |
| [Benchmarks](docs/BENCHMARKS.md) | Methodology, raw data, repro capsules |
| [Architecture](docs/ARCHITECTURE.md) | Design philosophy |

---

<details>
<summary>Kernel authoring API</summary>

Beyond model patching, ZMLX provides a Python-first API for writing Metal kernels:

```python
from zmlx.api import elementwise
import mlx.core as mx

mish = elementwise("x * tanh(log(1 + exp(x)))", name="mish")
y = mish(mx.random.normal((1024,)))
```

Also available: `reduce()`, `map_reduce()`, `@zmlx.jit`, and autograd support. See [`docs/QUICKSTART.md`](docs/QUICKSTART.md).

</details>

<details>
<summary>exo integration</summary>

ZMLX works with [exo](https://github.com/exo-explore/exo) for distributed inference. From a ZMLX checkout:

```bash
bash setup_zmlx.sh
bash exo/run_zmlx.sh
```

Or if exo is already installed: `pip install zmlx && zmlx-exo`

See [`docs/EXO.md`](docs/EXO.md) for details.

</details>

<details>
<summary>Custom MLX primitive (advanced)</summary>

For GLM-4.7-Flash and Qwen3-30B-A3B, an optional custom C++ Metal primitive (`gather_qmm_swiglu`) fuses quantized expert projections into a single GPU dispatch. This is **not part of released MLX** and requires building a local fork.

On stock MLX, these models are auto-skipped (0 modules patched, no regressions). `patch()` is always safe to call.

Build instructions: [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md).

</details>

<details>
<summary>Patching controls</summary>

```python
from zmlx.patch import patch, smart_patch

patch(model)                        # auto-detect, safe defaults
patch(model, patterns=["moe_mlp"])  # override safety; validate first

# Auto-benchmark: apply only patterns that actually help on your sample
sample = mx.array([tokenizer.encode("Hello")])
model = smart_patch(model, sample)
```

Environment variables:
- `ZMLX_DELTANET_FUSED_POSTCONV=0` — disable fused post-conv DeltaNet kernel
- `ZMLX_DELTANET_FUSED_POSTCONV_STATE_FP32=1` — FP32 state buffering for long decodes
- `ZMLX_DELTANET_FUSED_POSTCONV_RESYNC_INTERVAL=N` — resync every N tokens

</details>

<details>
<summary>Troubleshooting</summary>

| Symptom | Fix |
|:--|:--|
| `No module named 'mlx'` | Requires Apple Silicon Mac. Not supported on Intel or Linux. |
| `No module named 'mlx_lm'` | `pip install "zmlx[lm]"` |
| `patch()` shows 0 modules patched | Model may not match any patterns, or auto-skipped for safety. Run `python -m zmlx.validate <model>`. |
| Model downloads fill disk | Set `HF_HOME=/path/to/large/drive` before downloading. |

</details>

## Contributing

```bash
git clone https://github.com/Hmbown/ZMLX.git && cd ZMLX
pip install -e ".[dev]" && pytest
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgments

Built on [MLX](https://github.com/ml-explore/mlx) by Apple. Please cite MLX if you use ZMLX in your work.

## License

MIT
