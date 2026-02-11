# CLAUDE.md

Context for Claude Code when working in this repository.

## What ZMLX Is

ZMLX is an open-source toolkit that extends [MLX](https://github.com/ml-explore/mlx) with custom Metal kernels for Apple silicon. It provides a kernel authoring API, automatic gradient support, a catalog of over 70 ready-to-use kernels, and a model patching system that replaces MLX model layers with fused kernel equivalents. Think "Triton for MLX": author kernels in Python, compile to Metal, and wire them into real models.

**The big win:** ZMLX achieves measurable, token-identical decode speedups on real models shipping today:

| Model | Hardware | Speedup | MLX Required |
|:--|:--|--:|:--|
| LFM2-8B-A1B-4bit | M4 Max 36GB | **+11.6%** | stock (pip install) |
| LFM2-8B-A1B-4bit | M1 Pro 16GB | **+9.3%** | stock (pip install) |
| GLM-4.7-Flash-4bit | M4 Max 36GB | **+8.5%** | custom `gather_qmm_swiglu` |
| Qwen3-30B-A3B-4bit | M4 Max 36GB | **+5.5%** | custom `gather_qmm_swiglu` |

All results verified token-identical under greedy decoding. Repro capsules in `benchmarks/repro_capsules/`.

### How It Works

The kernel authoring API compiles Metal Shading Language directly from Python expressions. A call like `elementwise("x * tanh(log(1 + exp(x)))")` generates a Metal kernel, compiles it, caches it, and returns a callable that behaves like any other MLX op. Gradient support is built on `mx.custom_function`, where the backward passes are themselves Metal kernels rather than autodiff tape operations.

The primary use case is faster MoE decode inference. In MoE models, each token is routed to a subset of expert networks, and the standard path dispatches multiple Metal kernels per expert per layer. ZMLX fuses several of these into single dispatches, reducing command buffer overhead. The fused SwiGLU expert dispatch activates only when M <= 32 (decode); during prefill the standard MLX code path runs unchanged.

### Custom `gather_qmm_swiglu` Primitive

For GLM and Qwen3, we wrote a custom C++ Metal primitive (~800 lines) that fuses gate projection + up projection + SwiGLU activation for quantized MoE experts into a single GPU dispatch. This primitive is **not part of released MLX** — it lives in `mlx_local/` and is applied via the patch in `integrations/mlx_local_integration/`. On stock MLX, ZMLX auto-detects its absence and safely skips these models (0 modules patched, no regressions).

## Dev Environment

The project uses a venv at `.venv/`. Always use the venv Python — never the system Python.

```bash
source .venv/bin/activate                    # activate before any work
pip install -e ".[dev]"                      # install editable (already done)
```

All commands below assume the venv is active. If you see `ModuleNotFoundError: No module named 'zmlx'`, you forgot to activate.

## Build and Test

```bash
pytest -q                                    # full test suite (~670 tests)
pytest tests/test_kernels_catalog.py -q      # kernel correctness only
pytest -k "test_softmax" -q                  # single test by name

ruff check .                                 # lint
mypy src                                     # type check
```

End-to-end model validation compares patched and unpatched models under greedy decoding:

```bash
python -m zmlx.validate <model> --max-tokens 500 --runs 5
```

Repro capsules capture raw per-run data with full environment metadata:

```bash
python -m zmlx.bench.report benchmarks/repro_capsules/lfm2_m4max_20260131.json
```

Tests auto-skip on platforms without Apple silicon or Metal GPU support (see `tests/conftest.py`).

## Test Matrix

The matrix module (`src/zmlx/matrix/`) provides structured model catalog, test runner, and reporting across all Exo-supported models:

```bash
python -m zmlx.matrix catalog               # 58 models with metadata
python -m zmlx.matrix run <model> --runs 3   # validate one model
python -m zmlx.matrix run --all --runs 3     # all models that fit in RAM
python -m zmlx.matrix report                 # terminal heatmap table
python -m zmlx.matrix csv                    # CSV export
python -m zmlx.matrix html > matrix.html     # self-contained HTML
```

The catalog ingests Exo's TOML model cards plus manual LFM2 entries. Each model has inferred architecture (MoE/dense), expected ZMLX patterns, and exclusion reasons.

## Custom MLX Primitive (optional)

Some MoE fusions require a local MLX fork that exposes `mx.gather_qmm_swiglu`. ZMLX auto-detects this at runtime and only enables those paths when present.

```bash
python -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))"
```

Build instructions: `docs/EXPERIMENTAL_MLX.md`. The patch: `integrations/mlx_local_integration/gather_qmm_swiglu.patch`.

## Architecture

The codebase is organized under `src/zmlx/`:

### Metal kernel infrastructure
`metal.py` wraps `mx.fast.metal_kernel` with in-process caching keyed on source hash and configuration. `cache.py` manages the global compilation cache. `_compat.py` detects the platform and guards imports on non-Apple-silicon systems.

### Code generation
`codegen.py` provides template generators for elementwise, rowwise reduction, and two-pass map-reduce patterns. `elementwise.py` exposes `unary()`, `binary()`, and `map()` builders. `autograd.py` creates differentiable ops via `mx.custom_function`. `autotune.py` searches threadgroup size candidates.

### Kernel catalog (`kernels/`)
70+ kernels across 19 modules: activations, attention, bits, fused, fused_moe, image, indexing, linear, loss, moe, norms, optimizers, quant, reductions, rope, scan, softmax, transformer, vlsp.

### Patch system (`patch/`)
Walks a model tree, detects the architecture, replaces matching layers with fused kernel equivalents. Each pattern is a single file in `patch/patterns/`:

- `moe_mlp.py` — fused MoE expert dispatch (the main performance win)
- `swiglu_mlp.py` — dense SwiGLU layer fusion
- `geglu_mlp.py` — GeGLU activation fusion
- `rmsnorm.py`, `layernorm.py` — norm kernel replacements
- `softmax.py` — softmax kernel replacement
- `residual_norm.py` — fused residual + norm

Model-aware safety: `_FIDELITY_EXCLUDES` and `_PERF_EXCLUDES` in `patch/__init__.py` auto-skip patterns with known issues per model family (Qwen: swiglu/residual fidelity; GLM/Qwen: moe_mlp perf risk on stock MLX).

### Test matrix (`matrix/`)
Model catalog from Exo TOML cards, JSONL ledger for results, terminal/CSV/HTML reports. See `python -m zmlx.matrix --help`.

### Foundry module (`foundry/`)
Kernel template evaluation and dataset generation. Generates Metal kernel variants across 16 ops (9 kernel classes), evaluates correctness and performance, and exports training data for kernel-writing SFT.

CLI: `python -m zmlx.foundry {run, report, export, export-sft, list}`.

Key submodules: `ops/` (op registry), `templates/` (Metal template discovery), `harness/` (compile + benchmark orchestrator), `sampling/` (candidate generation), `scheduler.py` (curriculum staging), `export/` (JSONL + chat SFT export), `workers.py` (multi-process execution).

### Discover module (`discover/`)
LLM-guided PUCT tree search for Metal kernel optimization. Uses LLM backends (Claude, OpenAI, mock) to propose kernel variants, evaluates them against baselines, and tracks the search tree.

CLI: `python -m zmlx.discover {search, autorun, compare, report, list, export}`.

### Training module (`train/`)
LoRA fine-tuning CLI: `zmlx train`. Config, runner, callbacks, export.

### Integrations
- `exo.py` + `_exo_bootstrap/` — exo distributed inference integration
- `mlx_lm_compat.py` — compatibility layer for mlx-lm API differences
- `kv_cache.py` — quantized KV cache support

## Model Support

| Family | ZMLX Family Key | Architecture | Stock MLX | + Custom Primitive |
|:--|:--|:--|:--|:--|
| LFM2 | `lfm` | MoE | **+5-12% decode** | same |
| GLM-4.7 | `glm` | MoE | 0% (auto-skipped) | **+8% decode** |
| Qwen3-MoE | `qwen` | MoE | 0% (auto-skipped) | **+6% decode** |
| GPT-OSS | `gpt_oss` | MoE | ~+1% | same |
| DeepSeek-V3 | `deepseek` | MoE | patterns apply | untested (needs >300GB) |
| Kimi-K2.5 | `deepseek` | MoE | patterns apply | untested (needs >300GB) |
| Llama | `llama` | Dense | swiglu_mlp | same |
| Other | `unknown` | varies | safe no-op | same |

`patch()` is always safe to call. It returns unchanged if no patterns match or if the model family has known issues.

## CLI Entry Points

| Command | Purpose |
|:--|:--|
| `python -m zmlx.validate <model>` | Fidelity + throughput validation |
| `python -m zmlx.matrix {catalog,run,report,csv,html}` | Test matrix across models |
| `python -m zmlx.bench.report <capsule.json>` | Print benchmark report from capsule |
| `python -m zmlx.foundry {run,report,export,export-sft,list}` | Kernel template evaluation + dataset generation |
| `python -m zmlx.discover {search,autorun,compare,report,list,export}` | LLM-guided kernel optimization search |
| `zmlx train` | LoRA fine-tuning CLI |

## Design Decisions

MLX caches compiled Metal programs by source string. ZMLX generates MSL deterministically, so the same kernel configuration always reuses the compiled binary. No separate compilation cache needed.

In MoE decode, sequence length M is typically 1. At this scale, kernel dispatch overhead dominates over compute. The fused SwiGLU path activates only when M <= 32. At larger M (prefill), the standard MLX code path runs unchanged — prefill throughput is neutral.

Binary elementwise ops require matching shapes. Deliberate choice to avoid silent shape-dependent bugs from broadcasting.

## Conventions

Linting: `ruff check .` — line-length 100, target py310, rules `[E, F, I, UP, B]`, E501 ignored.
Type checking: `mypy src` — `ignore_missing_imports = true`, `warn_return_any = true`.

Every kernel needs a correctness test against the MLX reference op (see `tests/test_kernels_catalog.py`).
Performance claims need a repro capsule in `benchmarks/repro_capsules/` with hardware, OS, MLX version, raw per-run data.
New patch patterns go in `src/zmlx/patch/patterns/` as a single file, registered in `patterns/__init__.py`.

## Release Policy

- **Stable (default)**: stock MLX, only token-identical patches enabled.
- **Fast (opt-in)**: custom MLX allowed; experimental kernels opt-in, must be validated.
- **Edge (opt-in)**: nightly/dev MLX + experimental kernels for local testing only.

## Model Storage

HuggingFace models resolve through `HF_HOME`. Set it to a directory with enough space:

```
export HF_HOME=/path/to/your/hf_cache
```

## Key Files

| File | Purpose |
|:--|:--|
| `src/zmlx/patch/__init__.py` | Main `patch()` function, safety excludes, model family detection |
| `src/zmlx/patch/patterns/moe_mlp.py` | Fused MoE pattern (the main performance win) |
| `src/zmlx/patch/patterns/swiglu_mlp.py` | Dense SwiGLU fusion |
| `src/zmlx/kernels/fused_moe.py` | `gather_qmm_swiglu` detection and wrapper |
| `src/zmlx/kernels/moe.py` | MoE combine kernels (`moe_combine_no_fma`, etc.) |
| `src/zmlx/validate.py` | Fidelity + throughput validation CLI |
| `src/zmlx/matrix/` | Test matrix: catalog, runner, reports |
| `src/zmlx/api.py` | Public kernel authoring API |
| `src/zmlx/mlx_lm_compat.py` | mlx-lm version compatibility |
| `src/zmlx/foundry/__main__.py` | Foundry CLI entry point |
| `src/zmlx/foundry/ops/` | 16 registered ops across 9 kernel classes |
| `src/zmlx/foundry/export/sft.py` | Chat SFT dataset export |
| `src/zmlx/discover/__main__.py` | Discover CLI entry point |
| `integrations/mlx_local_integration/` | Custom MLX primitive patch |

## Documentation

| File | Contents |
|:--|:--|
| `README.md` | Benchmarks, usage, API overview, model support |
| `docs/TOUR.md` | Quick walkthrough for new contributors |
| `docs/ARCHITECTURE.md` | Design philosophy and multi-frontend vision |
| `docs/KERNELS.md` | Complete kernel catalog reference |
| `docs/QUICKSTART.md` | 5-minute kernel authoring tutorial |
| `docs/COOKBOOK.md` | Recipes for common patterns |
| `docs/BENCHMARKS.md` | Benchmark methodology and raw data |
| `docs/EXO.md` | exo integration guide (GLM/Qwen3) |
| `docs/EXPERIMENTAL_MLX.md` | Custom `gather_qmm_swiglu` primitive build instructions |
| `docs/ROADMAP.md` | Feature roadmap with priority matrix |
| `docs/FOUNDRY.md` | Foundry module: kernel template evaluation and SFT export |
| `docs/DEVELOPMENT.md` | Development areas and backlog |
| `UPSTREAM_PLAN.md` | What belongs upstream in MLX vs stays in ZMLX |
