# ZMLX — Metal kernels and model patching for MLX on Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

ZMLX extends [MLX](https://github.com/ml-explore/mlx) with a Python-first Metal kernel toolkit and optional, model-aware patching for faster MoE decode on Apple Silicon.

**Why ZMLX**

- **Speed (evidence-based):** reduces Metal dispatch overhead during MoE decode. Benchmarked on LFM2-8B-A1B with token-identical output (see [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).
- **Metal kernels from Python:** author custom ops from expressions (and optional custom VJPs for training), or use the 70+ kernel catalog.
- **Tooling you can verify:** `python -m zmlx.validate` for token-fidelity + throughput, plus repro capsules for repeatable benchmarks.

## Quick Start

**Requirements:** macOS 14+ (Apple Silicon), Python >= 3.10, `mlx>=0.30.0`

1. Install (patching examples use `mlx-lm`):

```bash
pip install "zmlx[train]"    # includes mlx-lm for model patching
# pip install zmlx            # kernel authoring only
```

2. Patch a model and generate (no weight conversion; patches apply in-place):

```python
import mlx_lm
from zmlx.patch import patch

model, tokenizer = mlx_lm.load("mlx-community/LFM2-8B-A1B-4bit")
patch(model)  # safe inference defaults for supported model families

print(
    mlx_lm.generate(
        model,
        tokenizer,
        prompt="Explain mixture-of-experts in one paragraph.",
        max_tokens=200,
    )
)
```

3. Verify token fidelity + throughput on your hardware:

```bash
python -m zmlx.validate mlx-community/LFM2-8B-A1B-4bit --max-tokens 200 --runs 3
```

Tip: large model downloads use the Hugging Face cache; set `HF_HOME` to control its location.

## What's Inside

- **Model patching:** `zmlx.patch.patch()` (preset-based) and `zmlx.patch.smart_patch()` (auto-benchmark patterns).
- **Kernel authoring:** `zmlx.api.elementwise()`, `reduce()`, `map_reduce()`, and `@zmlx.jit`.
- **Autograd support:** optional custom VJP paths via MLX custom functions.
- **Benchmarking:** `zmlx.bench.compare()` and `python -m zmlx.bench.report` (repro capsules in `benchmarks/repro_capsules/`).
- **Training CLI (optional):** `zmlx train`.
- **Experimental / opt-in:** a custom MLX build for some Qwen3 paths in `mlx_local/`.

## Docs & Benchmarks

| Doc | What's inside |
|:--|:--|
| [`docs/TOUR.md`](docs/TOUR.md) | Quick walkthrough and how to verify results |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | 5-minute kernel authoring tutorial |
| [`docs/COOKBOOK.md`](docs/COOKBOOK.md) | Recipes for common patterns |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Kernel catalog (by module/domain) |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Benchmark methodology + raw data |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Roadmap and priorities |
| [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md) | Optional custom-MLX experiments |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Design philosophy |
| [`UPSTREAM_PLAN.md`](UPSTREAM_PLAN.md) | What belongs upstream in MLX |

## Contributing / Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and conventions.

```bash
git clone https://github.com/Hmbown/ZMLX.git
cd ZMLX
pip install -e ".[dev]"
pytest
```

---

<details>
<summary>Benchmarks (repro capsules)</summary>

Full methodology and raw data: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

ZMLX's default inference patching is designed to keep prefill neutral (guards only enable fused kernels at small sequence lengths).

| Model | Hardware | Decode (baseline -> patched) | Change | Fidelity | Capsule |
|:--|:--|--:|--:|:--|:--|
| LFM2-8B-A1B-4bit | M1 Pro 16 GB | 105.5 tok/s -> 115.3 tok/s | +9.3% | token-identical | [`benchmarks/repro_capsules/lfm2_m1pro_20260131.json`](benchmarks/repro_capsules/lfm2_m1pro_20260131.json) |
| LFM2-8B-A1B-8bit | M1 Pro 16 GB | 72.8 tok/s -> 76.4 tok/s | +5.0% | token-identical | [`benchmarks/repro_capsules/lfm2_m1pro_20260131.json`](benchmarks/repro_capsules/lfm2_m1pro_20260131.json) |
| LFM2-8B-A1B-4bit | M4 Max 36 GB | 223.7 tok/s -> 250.3 tok/s | +11.9% | token-identical | [`benchmarks/repro_capsules/lfm2_m4max_20260131.json`](benchmarks/repro_capsules/lfm2_m4max_20260131.json) |
| LFM2-8B-A1B-8bit | M4 Max 36 GB | 152.5 tok/s -> 164.3 tok/s | +7.7% | token-identical | [`benchmarks/repro_capsules/lfm2_m4max_20260131.json`](benchmarks/repro_capsules/lfm2_m4max_20260131.json) |

To print a report from a capsule:

```bash
python -m zmlx.bench.report benchmarks/repro_capsules/<capsule>.json
```

</details>

<details>
<summary>Model support notes</summary>

ZMLX only patches what it recognizes. When a pattern is known to break token fidelity on a model family, `patch()` auto-excludes it.

**Default patching (stock MLX)**

Token-identical output with measurable decode improvement (on models where the patterns match).

| Model | Typical decode change | Fidelity |
|:--|:--|:--|
| LFM2-8B-A1B (4-bit / 8-bit) | +5-12% | token-identical |
| GPT-OSS-20B-MXFP4-Q4 | ~+1% | token-identical |

**Custom MLX kernel (opt-in)**

Some Qwen3 speedups require the custom MLX build in `mlx_local/`.

| Model | Typical decode change | Fidelity |
|:--|:--|:--|
| Qwen3-30B-A3B-Instruct-2507-4bit | +5-7% | token-identical |

For unlisted models, run: `python -m zmlx.validate <model>`.

Patching controls:

```python
import mlx.core as mx
from zmlx.patch import patch, smart_patch

patch(model)                      # inference defaults
patch(model, mode="training")     # training preset (adds norms/residual fusions)
patch(model, patterns=["moe_mlp"])  # override safety; validate first

# Auto-benchmark: apply only patterns that actually help on your sample
sample = mx.array([tokenizer.encode("Hello")])
model = smart_patch(model, sample)
```

</details>

<details>
<summary>How patching works (MoE decode)</summary>

MoE decode is often dominated by Metal kernel dispatch overhead (many small ops per token).

ZMLX targets the multi-op sequences that show up during decode:

- **Gating:** top-k softmax selection fused into one kernel (`topk_gating_softmax`).
- **Combine:** weight-and-reduce across experts fused into one kernel (`moe_combine`).
- **Guards:** fused paths only activate at small sequence lengths (decode), keeping prefill throughput neutral.

Deeper dives:

- Walkthrough: [`docs/TOUR.md`](docs/TOUR.md)
- Design notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

</details>

<details>
<summary>Kernel authoring (very short example)</summary>

ZMLX can compile small Python expressions into Metal kernels via MLX's `mx.fast.metal_kernel`:

```python
from zmlx.api import elementwise
import mlx.core as mx

mish = elementwise("x * tanh(log(1 + exp(x)))", name="mish")
y = mish(mx.random.normal((1024,)))
mx.eval(y)
```

Next steps:

- 5-minute tutorial: [`docs/QUICKSTART.md`](docs/QUICKSTART.md)
- Recipes: [`docs/COOKBOOK.md`](docs/COOKBOOK.md)
- Catalog: [`docs/KERNELS.md`](docs/KERNELS.md)

</details>

<details>
<summary>Custom MLX kernel (opt-in, Qwen3)</summary>

Qwen3 speedups require the custom MLX kernel patch in `mlx_local/` (opt-in).

Build (one-time):

```bash
cd mlx_local
# Limit CPU usage during build if desired:
# CMAKE_BUILD_PARALLEL_LEVEL=4 python3 setup.py build_ext --inplace
python3 setup.py build_ext --inplace
```

Use:

```bash
export PYTHONPATH=<REPO_ROOT>/mlx_local/python:<REPO_ROOT>/src:$PYTHONPATH
python3 -m zmlx.validate mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --max-tokens 1000 --runs 15
```

Remove `mlx_local/python` from `PYTHONPATH` to revert to stock MLX.

</details>

<details>
<summary>Troubleshooting</summary>

| Symptom | Fix |
|:--|:--|
| `ModuleNotFoundError: No module named 'mlx'` | Requires Apple Silicon macOS. ZMLX does not support Intel Macs or Linux. |
| `ModuleNotFoundError: No module named 'mlx_lm'` | Install with `pip install "zmlx[train]"` for model patching examples. |
| Model downloads fill disk | Set `HF_HOME` to a larger drive before running. |
| `patch()` shows 0 modules patched | The model may not match any patterns. Run `python -m zmlx.validate <model>` to verify. |

</details>

<details>
<summary>Precision note</summary>

Most kernels compute internally in **float32** regardless of input dtype. The exception is `moe_combine_exact`, which accumulates in the input dtype to match MLX's bfloat16 semantics for Qwen3.

</details>

---

## Acknowledgments

Built on [MLX](https://github.com/ml-explore/mlx) by Apple machine learning research. If you use ZMLX in your work, please also cite MLX:

```bibtex
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```

## License

MIT. See [`LICENSE`](LICENSE).
