# ZMLX — Metal kernel toolkit for MLX on Apple Silicon

[![PyPI](https://img.shields.io/pypi/v/zmlx.svg)](https://pypi.org/project/zmlx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey.svg)](https://github.com/ml-explore/mlx)

ZMLX extends [MLX](https://github.com/ml-explore/mlx) with custom Metal kernels for Apple Silicon. Author GPU kernels from Python expressions, use the 70+ kernel catalog, or call `patch(model)` for MoE decode speedups — +5-12% on LFM2-8B-A1B, token-identical, stock MLX. No model conversion, no config changes.

**Quick start** (after `pip install "zmlx[train]"`):

```python
import mlx_lm
from zmlx.patch import patch

model, tokenizer = mlx_lm.load("mlx-community/LFM2-8B-A1B-4bit")
patch(model)

text = mlx_lm.generate(model, tokenizer,
    prompt="Explain mixture-of-experts in one paragraph.",
    max_tokens=200)
```

Verify token fidelity on your hardware:

```bash
python -m zmlx.validate mlx-community/LFM2-8B-A1B-4bit --max-tokens 200 --runs 3
```

---

## Install

**Requirements:** macOS 14+ (Apple Silicon), Python >= 3.10, MLX >= 0.30.0

```bash
pip install "zmlx[train]"    # includes mlx-lm for model patching
pip install zmlx              # kernel authoring only
```

Large model downloads use the Hugging Face cache — set `HF_HOME` to control the location.

**From source:**

```bash
git clone https://github.com/Hmbown/ZMLX.git
cd ZMLX
pip install -e ".[dev]"
```

## Default mode (stock MLX)

`patch()` auto-detects which patterns are safe for your model family. In the default **Stable** mode, you get token-identical output with real decode speedups using stock MLX — nothing else required. Optional custom-MLX builds for Qwen3 are documented below.

Model-family safeguards:
- Qwen: `moe_mlp` is disabled by default due to recent decode regressions. Override with `ZMLX_PATCH_ALLOW_PERF_RISK=1`.
- LFM: TG progressive is denied by default via `ZMLX_FUSED_QSWIGLU_TG_DENY_FAMILY=lfm`.

---

## Benchmarks — Stable mode (stock MLX)

LFM2-8B-A1B: **+5-12% decode throughput**, token-identical, measured on M1 Pro and M4 Max. Prefill is neutral by design — the fused SwiGLU dispatch only activates at M <= 32.

<details>
<summary>LFM2-8B-A1B on M1 Pro 16 GB</summary>

> macOS 14.6.1 · MLX 0.30.4 · ZMLX 0.7.12 · Python 3.10.0 · commit `7de879e`
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
| Peak memory | — | 9.45 GB | |

</details>

<details>
<summary>LFM2-8B-A1B on M4 Max 36 GB</summary>

> macOS 26.1 · MLX 0.30.1 · ZMLX 0.7.12 · Python 3.12 · commit `139993e`
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

</details>

<details>
<summary>Validation update — 2026-02-01 (M4 Max 36 GB)</summary>

> macOS 26.1 · MLX 0.30.4 · ZMLX 0.7.12 · Python 3.14.2 · Apple M4 Max
>
> 1000 tokens, runs=15, greedy decoding (`python -m zmlx.validate`)

**Stock MLX (ZMLX only)**

| Model | Base prompt | Patched prompt | Base decode | Patched decode | Decode speedup | Fidelity |
|:--|--:|--:|--:|--:|--:|:--|
| LFM2-8B-A1B-4bit | 742.8 tok/s | 756.7 tok/s | 223.5 tok/s | 249.4 tok/s | 1.116x | PASS |
| LFM2-8B-A1B-8bit-MLX | 555.3 tok/s | 552.1 tok/s | 151.8 tok/s | 162.5 tok/s | 1.071x | PASS |
| GPT-OSS-20B-MXFP4-Q4 | 320.1 tok/s | 317.9 tok/s | 121.8 tok/s | 122.9 tok/s | 1.008x | PASS |

**Custom MLX kernel (opt-in)**

| Model | Base prompt | Patched prompt | Base decode | Patched decode | Decode speedup | Fidelity |
|:--|--:|--:|--:|--:|--:|:--|
| Qwen3-30B-A3B-Instruct-2507-4bit | 333.2 tok/s | 333.3 tok/s | 111.2 tok/s | 116.8 tok/s | 1.050x | PASS |

</details>

Full methodology and raw data: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

---

## SLIME/TPU⁰ Status (Feb 3, 2026)

Recent findings (decode focus, runs=5 unless noted):
- LFM2‑8B‑A1B (eps=10, TG=64, progressive): **+8.0% decode**, prompt ‑2.2%.
- LFM2.5‑1.2B (eps=10, TG=64, progressive): **+0.1% decode**, prompt +2.0% (neutral).
- Qwen3‑30B‑A3B (local MLX, runs=5): **regressed** with `moe_mlp` (‑4.5% decode), fused SwiGLU **much worse** (‑70% decode).
- Default safety: Qwen `moe_mlp` remains disabled; fused SwiGLU max tokens is **1** (decode‑only).

Next experiments:
- Add refine‑rate instrumentation to tighten bounds vs skip cost.
- Re‑evaluate Qwen3 MoE only at M=1 with an even tighter fused‑SwiGLU gate or keep it disabled.
- Try Qwen3‑Coder‑Next MLX quantizations, including `NexVeridian/Qwen3-Coder-Next-3bit` (not yet benchmarked).

---

## Toolkit

Beyond model patching, ZMLX is a Metal kernel authoring toolkit for MLX:

- **70+ kernel catalog** — SwiGLU, GeGLU, fused dropout, MoE gating, RMSNorm, RoPE, quantization
- **One-line kernel authoring** — `elementwise("x * tanh(log(1 + exp(x)))")` compiles to Metal
- **Automatic gradients** — custom VJP backward passes as Metal kernels via `mx.custom_function`
- **Benchmarking** — `zmlx.bench.compare()` for side-by-side timing, `zmlx.bench.report` for repro capsules
- **Training** — `zmlx train` CLI for fine-tuning with fused kernels (opt-in)
- **Zig frontend** — optional Zig API via C++ shim (see `docs/ARCHITECTURE.md`)

```python
from zmlx.api import elementwise
import mlx.core as mx

mish = elementwise("x * tanh(log(1 + exp(x)))", name="mish")
y = mish(mx.random.normal((1024,)))
```

<details>
<summary>Op-level microbenchmarks</summary>

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

</details>

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

## Kernel Authoring

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

<details>
<summary>More examples: reductions, softmax, testing</summary>

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

</details>

## How It Works

### The problem: dispatch overhead in MoE decode

In Mixture-of-Experts models, each token is routed to a subset of expert networks. During decode (one token at a time), the computation per expert is small — a few matrix multiplies on a single row vector. But the standard path dispatches multiple Metal kernels per expert per layer:

1. **Gating:** `softmax(logits)` → `argpartition` → `gather` → `normalize` — 4 dispatches
2. **Expert execution:** gate projection, up projection, SwiGLU activation, down projection — per expert
3. **Combine:** element-wise multiply by gating weights → reduce-sum across experts — 2 dispatches

On Apple Silicon, each Metal kernel dispatch has fixed overhead (command buffer encoding, GPU scheduling). When the actual compute per dispatch is small — as it is for M=1 decode — this overhead dominates. The GPU spends more time waiting between kernels than doing math.

<details>
<summary>What ZMLX fuses</summary>

ZMLX replaces the multi-dispatch sequences with single Metal kernels that do the same math in one pass. All fused kernels are generated from Python via [`mx.fast.metal_kernel`](https://ml-explore.github.io/mlx/build/html/python/fast.html) — no changes to MLX core required.

**Fused top-k gating softmax** (`topk_gating_softmax`):

Replaces the 4-dispatch gating sequence with a single kernel. For small expert counts (D <= 32, common in MoE), the kernel uses SIMD group operations — each row is processed by one SIMD group (32 threads), with `simd_max` and `simd_sum` for the softmax reduction and a register-based insertion sort for top-k selection. For larger D, a threadgroup reduction with shared memory is used. The kernel computes softmax probabilities and selects the top-k experts with their normalized weights in one pass.

**Fused expert combine** (`moe_combine`):

Replaces the separate element-wise multiply and reduce-sum with a single kernel that reads each expert output once, multiplies by its gating weight, and accumulates the weighted sum in float32. Output shape goes directly from `(B, K, D)` to `(B, D)` without materializing the intermediate `weights * expert_outputs` tensor.

### Why prefill is unaffected

The fused SwiGLU expert dispatch is guarded with a sequence length check (`M <= 1`). During prefill, M equals the prompt length (typically hundreds or thousands of tokens). At this scale, the compute-to-dispatch ratio is high and the standard MLX path is already efficient. The guard ensures ZMLX never regresses prefill performance.

### Correctness guarantee

Token fidelity is a first-class requirement. `patch()` auto-detects the model family and excludes patterns with known fidelity issues. The fused gating kernel reproduces the exact same top-k selection and softmax normalization as the reference MLX ops. The combine kernel accumulates in float32 (or dtype-matched for Qwen3's `moe_combine_exact`). `python -m zmlx.validate` compares every generated token ID between patched and unpatched models under greedy decoding.

</details>

### Patching options

```python
from zmlx.patch import patch, smart_patch

patch(model)                       # auto-detect, apply safe defaults
patch(model, patterns=["moe_mlp"]) # force specific pattern (overrides safety)
patch(model, mode="training")      # add norm fusions for backward pass
patch(model, profile="qwen3")      # Qwen3 minimal (moe-only)

# Auto-benchmark: apply only patterns that actually help
sample = mx.array([tokenizer.encode("Hello")])
model = smart_patch(model, sample)
```

---

## Custom MLX kernel (opt-in)

Qwen3 speedups require the custom MLX kernel patch in `mlx_local/` (opt-in).

**Build (one-time)**

```bash
cd mlx_local
# Limit CPU usage during build if desired:
# CMAKE_BUILD_PARALLEL_LEVEL=4 python3 setup.py build_ext --inplace
python3 setup.py build_ext --inplace
```

**Use**

```bash
# From the ZMLX repo root:
export PYTHONPATH=$(pwd)/mlx_local/python:$(pwd)/src:$PYTHONPATH
python3 -m zmlx.validate mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit --max-tokens 1000 --runs 15
```

Keep `mlx_local/python` before `src` in `PYTHONPATH` so the custom MLX build is used.

**Safety**

- The **build step** is the only CPU-heavy operation; cap it with `CMAKE_BUILD_PARALLEL_LEVEL`.
- Runtime uses the same MLX threading behavior as stock; remove `mlx_local/python` from `PYTHONPATH` to revert.

---

## Model Support

### Stable (stock MLX)

Token-identical output, measurable decode improvement. Safe to use as-is.

| Model | Decode speedup | Fidelity | Patterns |
|:--|:--|:--|:--|
| **LFM2-8B-A1B-4bit** | **+9-12%** | token-identical | `moe_mlp` + `swiglu_mlp` |
| **LFM2-8B-A1B-8bit** | **+5-8%** | token-identical | `moe_mlp` + `swiglu_mlp` |
| **GPT-OSS-20B-MXFP4-Q4** | **+1%** | token-identical | `moe_mlp` |

### Custom MLX kernel (opt-in)

Requires the custom MLX kernel build (see `mlx_local/`). Token-identical, validated at ~1.050-1.063x with some run-to-run variance.

| Model | Decode speedup | Fidelity | Patterns |
|:--|:--|:--|:--|
| **Qwen3-30B-A3B-Instruct-2507-4bit** | **+5-7%** | token-identical | `moe_mlp` |

### Tested (no gain, stock MLX)

| Model | Status | Notes |
|:--|:--|:--|
| GLM-4.7-Flash-4bit | 0.955x, FAIL | `@mx.compile` gating; auto-excluded |
| Nemotron-3-Nano-30B-A3B-NVFP4 | 0.999x, PASS | Hybrid Mamba-MoE, bandwidth-limited at 19.4 GB |
| LFM2.5-1.2B-Thinking-MLX-8bit | 0.997x, PASS | Dense model, no matched MoE patterns |
| Qwen3-30B-A3B-Instruct-2507-4bit | 0.98x, PASS | No gain on stock MLX; custom kernel required for speedup |
| Qwen3-4B-4bit (dense) | diverges at token 18 | Dense model, patches not expected to help |
| Llama-3.2-1B-4bit | 0.98x, PASS | Dense model, bandwidth-bound |

For unlisted models: `python -m zmlx.validate <model>`.

---

## Troubleshooting

| Symptom | Fix |
|:--|:--|
| `ModuleNotFoundError: No module named 'mlx'` | Requires Apple Silicon macOS. ZMLX does not support Intel Macs or Linux. |
| `ModuleNotFoundError: No module named 'mlx_lm'` | Install with `pip install "zmlx[train]"` for model patching. |
| Model downloads fill disk | Set `HF_HOME` to a larger drive before running. |
| `patch()` shows 0 modules patched | The model may not match any patterns. Run `python -m zmlx.validate <model>` to verify. |

---

## Future Projects

- **Forced MoE parallelization** — explore true multi‑queue GPU overlap (per‑stream command queues + MTLEvent sync) for concurrent expert execution.
- **Combine kernel tuning** — specialize `moe_combine_fp32` for K=8 and threadgroup sizes on M‑series GPUs.
- **Prefill optimizations** — large‑M attention/GEMM fusions and compile‑graph experiments.

---

## Precision

Most kernels compute internally in **float32** regardless of input dtype. The exception is `moe_combine_exact`, which accumulates in the input dtype to match MLX's bfloat16 semantics for Qwen3.

---

## Documentation

| Doc | What's inside |
|:--|:--|
| [`docs/TOUR.md`](docs/TOUR.md) | Quick walkthrough and orientation |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | 5-minute tutorial |
| [`docs/COOKBOOK.md`](docs/COOKBOOK.md) | Recipes for common patterns |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Complete kernel catalog |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | Benchmark methodology and raw data |
| [`docs/EXPERIMENTAL_MLX.md`](docs/EXPERIMENTAL_MLX.md) | Optional custom-MLX experiments |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Design philosophy |
| [`UPSTREAM_PLAN.md`](UPSTREAM_PLAN.md) | What belongs upstream in MLX |

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

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and conventions.

---

## License

MIT. See [`LICENSE`](LICENSE).
