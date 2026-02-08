# ZMLX — Metal kernels and model patching for MLX on Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

ZMLX extends [MLX](https://github.com/ml-explore/mlx) with a Python-first Metal kernel toolkit and model-aware patching for faster MoE decode on Apple Silicon.

**What ZMLX does**

- **Metal kernels from Python:** write `elementwise("x * tanh(log(1 + exp(x)))")` and get a compiled Metal kernel with caching, autograd support, and the 70+ kernel catalog.
- **Model patching:** `patch(model)` replaces MoE gating/combine/activation sequences with fused Metal kernels, reducing dispatch overhead during decode. Token-identical output; verify with `python -m zmlx.validate`.
- **Optional custom primitive (GLM/Qwen3):** build the custom `gather_qmm_swiglu` primitive to fuse quantized expert projections for GLM-4.7-Flash and Qwen3-30B-A3B. See the GLM-4.7-Flash stress benchmark results below + [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md). On stock MLX these models auto-skip safely.
- **Proven on stock MLX:** LFM2-8B-A1B shows **+8-13% decode** on released MLX with no custom builds needed. These gains come from ZMLX's own Metal kernels for fused gating, combine, and SwiGLU activation.
- **Next release prep:** Qwen3.5 support scaffolding (pending official `Qwen/*` release on Hugging Face).

## GLM-4.7-Flash — Stress-Benchmark-Verified Decode Speedups (Custom Primitive)

ZMLX's flagship result is **token-identical decode speedups** on `mlx-community/GLM-4.7-Flash-4bit` when running with a custom MLX build that includes `gather_qmm_swiglu` (see [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md)).

**Stress benchmark protocol:** 5 prompts × 3 generation lengths × 5 runs (15 configs), greedy decode, token-by-token fidelity across configs. The benchmark runner is [`benchmarks/bench_glm_stress.py`](benchmarks/bench_glm_stress.py).

**Result (Apple M4 Max 36 GB, MLX `0.30.4.dev20260204+2f324cc`):** 66.3 → 70.7 tok/s average decode throughput (**+6.6%**, mean of per-config **median** tok/s), **15/15 configs token-identical**. Capsule: [`benchmarks/repro_capsules/glm_stress_m4_20260205_rerun_mlx0304dev2f324cc.json`](benchmarks/repro_capsules/glm_stress_m4_20260205_rerun_mlx0304dev2f324cc.json).

Prior rerun capsule (same machine + MLX): [`benchmarks/repro_capsules/glm_stress_m4_20260205_d17ab1b.json`](benchmarks/repro_capsules/glm_stress_m4_20260205_d17ab1b.json).

**Speedup vs length (avg across prompts)**
| Length | Avg Baseline | Avg Patched | Avg Speedup |
|:--|--:|--:|--:|
| 256 | 70.2 | 73.6 | 1.049x |
| 1024 | 65.0 | 70.3 | **1.081x** |
| 2048 | 63.8 | 68.1 | 1.068x |

**Speedup vs prompt type (avg across lengths)**
| Prompt | Avg Baseline | Avg Patched | Avg Speedup |
|:--|--:|--:|--:|
| english_technical | 66.2 | 69.9 | 1.055x |
| chinese | 66.7 | 68.8 | 1.031x |
| code | 64.6 | 69.6 | **1.078x** |
| math_reasoning | 66.7 | 69.0 | 1.035x |
| creative | 67.3 | 76.2 | **1.133x** |

Reproduce on your machine (writes a new capsule + log):

```bash
source .venv/bin/activate

python benchmarks/bench_glm_stress.py \
  --prompts english_technical,chinese,code,math_reasoning,creative \
  --lengths 256,1024,2048 \
  --runs 5 \
  --json-out benchmarks/repro_capsules/glm_stress_<your_machine>_<date>.json
```

### What We Learned (Hypotheses)

- **Prompt-dependent speedups:** the stress test shows larger gains on some prompt types (e.g. `code`) than others (e.g. `english_technical`). A working hypothesis is that **expert routing distributions** differ across prompt styles (hot experts vs high-entropy routing), which changes how much overhead the fused expert path saves.
- **Benchmarking needs diversity:** single-prompt validation can over/under-estimate performance; the 15-config stress protocol catches these differences and is the recommended regression gate for GLM work.

### Next Steps

- Add an **opt-in routing histogram** mode (log Top‑K expert IDs/weights during the stress run) to correlate routing entropy with speedups and identify “hot” experts worth special-casing.
- Prioritize **GLM KV-cache experiments with delayed quantization**: `kv_bits=4, quantized_kv_start=128` reached `72.9 -> 78.9 tok/s` (**+8.2%**) at 1024 tokens in quick reruns (`benchmarks/repro_capsules/glm47_flash_kv4_t1024_s128_m4max_20260205_rerun_mlx0304dev2f324cc.json`).
- Keep **shared-expert overlap** disabled for now: `shared_experts_overlap_streams2` regressed to `0.734x` in reruns (`benchmarks/repro_capsules/glm47_flash_shared_overlap_m4max_20260205_rerun2_mlx0304dev2f324cc.json`).
- Keep **`residual_norm` disabled** on GLM: still fails greedy token fidelity (`1/200` identical) in reruns (`benchmarks/repro_capsules/glm47_flash_residual_norm_m4max_20260205_rerun2_mlx0304dev2f324cc.json`).
- Treat **`glm47_rope` as low-priority**: currently modest decode gain (`1.013x`) in quick reruns (`benchmarks/repro_capsules/glm47_flash_rope_m4max_20260205_rerun2_mlx0304dev2f324cc.json`).
- For **Qwen3-30B-A3B**, current best decode uplift is `96.5 -> 104.3 tok/s` (**+8.1%**) with `patch(model, profile="qwen3")` (`benchmarks/repro_capsules/qwen3_a3b_profile_qwen3_m4max_20260205_rerun2_mlx0304dev2f324cc.json`).
- For **Qwen3**, keep no-KV as the performance baseline for now: quick 1024-token reruns with `kv_bits=4, quantized_kv_start=128` did not beat no-KV absolute decode tok/s (`benchmarks/repro_capsules/qwen3_a3b_nokv_t1024_m4max_20260205_rerun_mlx0304dev2f324cc.json` vs `benchmarks/repro_capsules/qwen3_a3b_kv4_t1024_s128_m4max_20260205_rerun_mlx0304dev2f324cc.json`).
- 2026-02-08 sanity checks on current MLX showed smaller relative gains on default patching:
  - GLM-4.7-Flash-4bit: `1.048x` (200 tok), `1.037x` (2000 tok)
  - Qwen3-30B-A3B-4bit: `1.022x` (200 tok), `1.034x` (2000 tok)
  These are still token-identical; benchmark deltas vary with decode length, thermals, and MLX baseline changes.
- Prepare **Qwen3.5** integration in advance (once official checkpoints are published under the `Qwen` org):
  - Add matrix catalog aliases in `src/zmlx/matrix/models.py` for new model ID patterns.
  - Add/adjust KVTC geometry presets in `src/zmlx/kvtc/presets.py` if architecture differs from current Qwen3 MoE.
  - Validate with `python -m zmlx.validate <model> --max-tokens 200 --runs 3` first, then long-run checks (1000/2000 tokens).

### Qwen3.5 Watchlist (Hugging Face)

As of 2026-02-08, there are no official `Qwen/Qwen3.5*` model IDs in the Hugging Face API; community repos with `qwen-3.5` names exist but are not authoritative release targets.

Quick checks:

```bash
curl -s "https://huggingface.co/api/models?author=Qwen&search=Qwen3&limit=50"
curl -s "https://huggingface.co/api/models?author=Qwen&search=3.5&limit=20"
```

## DeepSeek-V3.2 + Kimi-K2.5 Experiments (Experimental)

DeepSeek-V3.2 and Kimi-K2.5 are **DeepSeek-style MoE** variants. ZMLX provides
an **opt-in** fused router (`deepseek_router`) plus existing MoE combine/SwiGLU
fusions (`moe_mlp`, `swiglu_mlp`) that may apply depending on your MLX/MLX-LM
build.

**Hardware validation needed:** we have not yet run full fidelity + throughput
validation on actual DeepSeek-V3.2 / Kimi-K2.5 weights in this repo due to
memory constraints. If you can load these models, community benchmarking would
help confirm behavior and performance.

Suggested validation (greedy token fidelity + throughput):

```bash
source .venv/bin/activate

python -m zmlx.validate <model_id> \
  --patterns deepseek_router moe_mlp swiglu_mlp \
  --runs 3 --max-tokens 200
```

Notes:
- `deepseek_router` is intentionally opt-in and only changes expert routing.
- Please share repro capsules under `benchmarks/repro_capsules/` if you record
  performance results.
- For exo users, see the quickstart in [`docs/HANDOFF_DEEPSEEK_KIMI.md`](docs/HANDOFF_DEEPSEEK_KIMI.md).

## Quick Start

**Requirements:** macOS 14+ (Apple Silicon), Python >= 3.10, `mlx>=0.30.0`

1. Install (patching examples use `mlx-lm`):

```bash
pip install "zmlx[lm]"       # includes mlx-lm for model patching
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
- **Custom MLX primitive (opt-in):** build a custom MLX with `gather_qmm_swiglu` (see [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md); patch lives in `integrations/mlx_local_integration/`).

## exo Integration

ZMLX works with [exo](https://github.com/exo-explore/exo) for faster GLM-4.7-Flash and Qwen3-30B-A3B decode. No source patching needed:

```bash
bash setup_zmlx.sh
bash exo/run_zmlx.sh
```

ZMLX hooks into exo's model loading at runtime — when GLM/Qwen3 load with the custom MLX primitive, MoE expert dispatch is fused. Measured speedups vary by prompt/length; see [`docs/EXO.md`](docs/EXO.md) and repro capsules in `benchmarks/repro_capsules/`.

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
| LFM2-8B-A1B-4bit | M4 Max 36 GB | 197.8 tok/s -> 223.2 tok/s | **+12.8%** | token-identical | [`benchmarks/repro_capsules/lfm2_m4max_20260205_rerun_mlx0304dev2f324cc.json`](benchmarks/repro_capsules/lfm2_m4max_20260205_rerun_mlx0304dev2f324cc.json) |
| LFM2-8B-A1B-8bit | M4 Max 36 GB | 134.0 tok/s -> 145.4 tok/s | +8.5% | token-identical | [`benchmarks/repro_capsules/lfm2_m4max_20260205_rerun_mlx0304dev2f324cc.json`](benchmarks/repro_capsules/lfm2_m4max_20260205_rerun_mlx0304dev2f324cc.json) |
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

ZMLX provides the model-side integration: auto-detecting MoE architectures, rewiring forward passes to use the fused primitive, and using native MLX combine ops on GLM/Qwen3 for fidelity and lower dispatch overhead.

**On stock MLX (released 0.30.4/0.30.5), ZMLX auto-skips these models** (0 modules patched, 0% change) to avoid regressions. `patch()` is always safe to call.

| Model | Hardware | Decode (baseline -> patched) | Change | Fidelity | Capsule |
|:--|:--|--:|--:|:--|:--|
| GLM-4.7-Flash-4bit | M4 Max 36 GB | 86.6 tok/s -> 92.4 tok/s | **+6.7%** | 200/200 tokens identical | [`benchmarks/repro_capsules/glm47_flash_control_m4max_20260205.json`](benchmarks/repro_capsules/glm47_flash_control_m4max_20260205.json) |
| Qwen3-30B-A3B-4bit | M4 Max 36 GB | 106.6 tok/s -> 115.0 tok/s | **+7.9%** | 200/200 tokens identical | [`benchmarks/repro_capsules/qwen3_a3b_moe_mlp_m4max_20260205.json`](benchmarks/repro_capsules/qwen3_a3b_moe_mlp_m4max_20260205.json) |

For the full GLM-4.7-Flash stress benchmark protocol + tables, see the “GLM-4.7-Flash — Stress-Benchmark-Verified Decode Speedups” section above.

Capsules and logs:
- Historical full stress run: [`benchmarks/repro_capsules/glm_stress_m4_20260204.json`](benchmarks/repro_capsules/glm_stress_m4_20260204.json) (log under `benchmarks/results/glm_stress/`)
- Latest re-run using [`benchmarks/bench_glm_stress.py`](benchmarks/bench_glm_stress.py): [`benchmarks/repro_capsules/glm_stress_m4_20260205_rerun_mlx0304dev2f324cc.json`](benchmarks/repro_capsules/glm_stress_m4_20260205_rerun_mlx0304dev2f324cc.json)

See [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md) for build instructions.

</details>

<details>
<summary>Model support summary</summary>

| Model | Stock MLX | + Custom primitive | What ZMLX does |
|:--|:--|:--|:--|
| LFM2-8B-A1B | speedup (see stock MLX table) | same | ZMLX Metal kernels: fused MoE gating + combine + SwiGLU |
| GLM-4.7-Flash | 0% (auto-skipped) | speedup (see custom primitive table) | ZMLX patching + custom `gather_qmm_swiglu` primitive |
| Qwen3-30B-A3B | 0% (auto-skipped) | speedup (see custom primitive table) | ZMLX patching + custom `gather_qmm_swiglu` primitive |
| GPT-OSS-20B | fused SwiGLU activation | same | ZMLX Metal kernel: fused SwiGLU activation |
| Other models | safe no-op | same | `patch()` returns unchanged if no patterns match |

All results are token-identical under greedy decoding. Verify on your hardware with `python -m zmlx.validate <model>`.

Patching controls:

```python
import mlx.core as mx
from zmlx.patch import patch, smart_patch

patch(model)                      # inference defaults (auto-skips unsafe patterns)
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
| `ModuleNotFoundError: No module named 'mlx_lm'` | Install with `pip install "zmlx[lm]"` for model patching examples. |
| Model downloads fill disk | Set `HF_HOME` to a larger drive before running. |
| `patch()` shows 0 modules patched | The model may not match any patterns, or ZMLX auto-skipped them for safety. Run `python -m zmlx.validate <model>` to verify. |
| GLM/Qwen shows 0 modules patched | Expected on stock MLX. Requires building the custom `gather_qmm_swiglu` primitive in `mlx_local/` (see [docs](docs/EXPERIMENTAL_MLX.md)). |

</details>

<details>
<summary>Precision note</summary>

Most kernels compute internally in **float32** regardless of input dtype. The exception is `moe_combine_exact`, which accumulates in the input dtype to match MLX's bfloat16 semantics. GLM and Qwen3 use native MLX ops for the combine step (`(y * scores[..., None]).sum(axis=-2)`) to match the original model code exactly and avoid custom-kernel dispatch overhead.

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
