# ZMLX — Metal kernels and model patching for MLX on Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

ZMLX extends [MLX](https://github.com/ml-explore/mlx) with a Python-first Metal kernel toolkit and model-aware patching for faster MoE decode on Apple Silicon.

**What ZMLX does**

- **Metal kernels from Python:** write `elementwise("x * tanh(log(1 + exp(x)))")` and get a compiled Metal kernel with caching, autograd support, and the 70+ kernel catalog.
- **Model patching:** `patch(model)` replaces MoE gating/combine/activation sequences with fused Metal kernels, reducing dispatch overhead during decode. Token-identical output; verify with `python -m zmlx.validate`.
- **Optional custom primitive (GLM/Qwen3):** build the custom `gather_qmm_swiglu` primitive to fuse quantized expert projections for GLM-4.7-Flash and Qwen3-30B-A3B (~+8% / ~+6% decode). On stock MLX these models auto-skip safely. See [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md) and [`docs/EXO.md`](docs/EXO.md).
- **Proven on stock MLX:** LFM2-8B-A1B shows **+5-12% decode** on released MLX with no custom builds needed. These gains come from ZMLX's own Metal kernels for fused gating, combine, and SwiGLU activation.
- **Next test target:** Qwen3-80B Coder (planned).

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
- **Custom MLX primitive (opt-in):** build a custom MLX with `gather_qmm_swiglu` (see [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md); patch lives in `integrations/mlx_local_integration/`).

## exo Integration

ZMLX works with [exo](https://github.com/exo-explore/exo) for faster GLM-4.7-Flash and Qwen3-30B-A3B decode in distributed inference clusters. Setup is automated:

```bash
git clone https://github.com/Hmbown/ZMLX.git
cd ZMLX
bash setup_zmlx.sh        # one-time setup (creates ./exo + ./exo/run_zmlx.sh)
bash exo/run_zmlx.sh      # launch exo with ZMLX
```

When GLM loads, ZMLX fuses all 46 MoE layers + 1 dense SwiGLU (~8% faster decode, token-identical) when the custom MLX primitive is available. See [`docs/EXO.md`](docs/EXO.md) for the full guide.

## Docs

| Doc | What's inside |
|:--|:--|
| [`docs/TOUR.md`](docs/TOUR.md) | Quick walkthrough and how to verify results |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | 5-minute kernel authoring tutorial |
| [`docs/COOKBOOK.md`](docs/COOKBOOK.md) | Recipes for common patterns |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Kernel catalog (by module/domain) |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Benchmark methodology + raw data |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Design philosophy |
| [`docs/EXO.md`](docs/EXO.md) | exo integration guide (GLM/Qwen3) |
| [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md) | Custom MLX primitive details |
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
<summary>Benchmarks (stock MLX — works with pip install mlx)</summary>

These results use **released MLX** (`pip install mlx`). The speedup comes from ZMLX's own Python-level Metal kernels (fused gating, combine, SwiGLU activation) — no custom C++ or MLX fork required.

Full methodology and raw data: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

| Model | Hardware | Decode (baseline -> patched) | Change | Fidelity | Capsule |
|:--|:--|--:|--:|:--|:--|
| LFM2-8B-A1B-4bit | M4 Max 36 GB | 223.5 tok/s -> 249.4 tok/s | **+11.6%** | token-identical | [`benchmarks/repro_capsules/lfm2_m4max_20260131.json`](benchmarks/repro_capsules/lfm2_m4max_20260131.json) |
| LFM2-8B-A1B-8bit | M4 Max 36 GB | 152.5 tok/s -> 164.3 tok/s | +7.7% | token-identical | [`benchmarks/repro_capsules/lfm2_m4max_20260131.json`](benchmarks/repro_capsules/lfm2_m4max_20260131.json) |
| LFM2-8B-A1B-4bit | M1 Pro 16 GB | 105.5 tok/s -> 115.3 tok/s | +9.3% | token-identical | [`benchmarks/repro_capsules/lfm2_m1pro_20260131.json`](benchmarks/repro_capsules/lfm2_m1pro_20260131.json) |
| LFM2-8B-A1B-8bit | M1 Pro 16 GB | 72.8 tok/s -> 76.4 tok/s | +5.0% | token-identical | [`benchmarks/repro_capsules/lfm2_m1pro_20260131.json`](benchmarks/repro_capsules/lfm2_m1pro_20260131.json) |
| GPT-OSS-20B-4bit | M4 Max 36 GB | 121.8 tok/s -> 122.9 tok/s | +1.0% | token-identical | — |

To print a report from a capsule:

```bash
python -m zmlx.bench.report benchmarks/repro_capsules/<capsule>.json
```

</details>

<details>
<summary>Benchmarks (custom MLX primitive — requires building mlx_local/)</summary>

GLM-4.7-Flash and Qwen3-30B-A3B gains come from `gather_qmm_swiglu`, a **custom C++ Metal primitive we wrote** (~800 lines of C++/Metal). It fuses gate projection + up projection + SwiGLU activation for quantized MoE experts into a single GPU dispatch. This primitive is not part of released MLX — build it by applying the patch described in [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md).

ZMLX provides the model-side integration: auto-detecting MoE architectures, rewiring forward passes to use the fused primitive, and a deterministic no-FMA combine kernel to preserve token fidelity on GLM.

**On stock MLX (released 0.30.4/0.30.5), ZMLX auto-skips these models** (0 modules patched, 0% change) to avoid regressions. `patch()` is always safe to call.

| Model | Hardware | Decode (baseline -> patched) | Change | Fidelity |
|:--|:--|--:|--:|:--|
| GLM-4.7-Flash-4bit | M4 Max 36 GB | 85.8 tok/s -> 92.8 tok/s | **+8.1%** | 128/128 identical |
| Qwen3-30B-A3B-4bit | M4 Max 36 GB | 117 tok/s -> 123 tok/s | +5.5% | 128/128 identical |

See [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md) for build instructions.

</details>

<details>
<summary>Model support summary</summary>

| Model | Stock MLX | + Custom primitive | What ZMLX does |
|:--|:--|:--|:--|
| LFM2-8B-A1B | **+5-12% decode** | same | ZMLX Metal kernels: fused MoE gating + combine + SwiGLU |
| GLM-4.7-Flash | 0% (auto-skipped) | **+8% decode** | ZMLX patching + custom `gather_qmm_swiglu` primitive |
| Qwen3-30B-A3B | 0% (auto-skipped) | **+6% decode** | ZMLX patching + custom `gather_qmm_swiglu` primitive |
| GPT-OSS-20B | ~+1% | same | ZMLX Metal kernel: fused SwiGLU activation |
| Other models | safe no-op | same | `patch()` returns unchanged if no patterns match |

All results are token-identical under greedy decoding. Verify on your hardware with `python -m zmlx.validate <model>`.

Patching controls:

```python
import mlx.core as mx
from zmlx.patch import patch, smart_patch

patch(model)                      # inference defaults (auto-skips unsafe patterns)
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
- **Expert SwiGLU (when available):** gate+up projection+SwiGLU fused into one dispatch via custom `gather_qmm_swiglu` primitive.
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
<summary>Troubleshooting</summary>

| Symptom | Fix |
|:--|:--|
| `ModuleNotFoundError: No module named 'mlx'` | Requires Apple Silicon macOS. ZMLX does not support Intel Macs or Linux. |
| `ModuleNotFoundError: No module named 'mlx_lm'` | Install with `pip install "zmlx[train]"` for model patching examples. |
| Model downloads fill disk | Set `HF_HOME` to a larger drive before running. |
| `patch()` shows 0 modules patched | The model may not match any patterns, or ZMLX auto-skipped them for safety. Run `python -m zmlx.validate <model>` to verify. |
| GLM/Qwen shows 0 modules patched | Expected on stock MLX. Requires building the custom `gather_qmm_swiglu` primitive in `mlx_local/` (see [docs](docs/EXPERIMENTAL_MLX.md)). |

</details>

<details>
<summary>Precision note</summary>

Most kernels compute internally in **float32** regardless of input dtype. The exception is `moe_combine_exact`, which accumulates in the input dtype to match MLX's bfloat16 semantics for Qwen3. GLM uses `moe_combine_no_fma` to disable FMA contraction and match MLX's non-fused multiply-then-sum reduction order.

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
