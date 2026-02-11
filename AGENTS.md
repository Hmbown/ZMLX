# AGENTS.md

Guidance for AI coding agents working in this repository.

## Rules

- Do **not** include machine-specific absolute paths (e.g., `/Volumes/VIXinSSD/...`) in README, docs, or user-facing text. Use placeholders like `<REPO_ROOT>`, `$HF_HOME`, or repository-relative paths.
- Do **not** fabricate benchmark numbers. All performance claims must come from actual measurements with repro capsules in `benchmarks/repro_capsules/`.
- Do **not** modify `mlx_local/` or `exo/` — these are external projects cloned locally and gitignored.
- Always activate the venv (`source .venv/bin/activate`) before running any Python commands.
- Run `ruff check .` and `pytest -q` before considering any code change complete.

## Project Overview

ZMLX is a Metal kernel toolkit for MLX on Apple Silicon. It provides:

1. **Kernel authoring** — `elementwise("x * tanh(log(1 + exp(x)))")` compiles to Metal
2. **Model patching** — `patch(model)` fuses MoE expert dispatch for faster decode
3. **70+ kernel catalog** — activations, attention, norms, MoE, quant, loss, etc.
4. **Custom C++ primitive** — `gather_qmm_swiglu` for GLM/Qwen3 (optional, ~800 lines Metal/C++)

### What Actually Works (Proven Results)

| Model | Speedup | Requires |
|:--|--:|:--|
| LFM2-8B-A1B-4bit | +11.6% decode | stock MLX |
| GLM-4.7-Flash-4bit | +8.5% decode | custom `gather_qmm_swiglu` |
| Qwen3-30B-A3B-4bit | +5.5% decode | custom `gather_qmm_swiglu` |

All token-identical under greedy decoding.

## Key Commands

```bash
pytest -q                                    # ~670 tests
ruff check .                                 # lint
python -m zmlx.validate <model> --runs 3     # fidelity + throughput
python -m zmlx.matrix catalog                # 58 models with metadata
python -m zmlx.matrix report                 # test matrix heatmap
```

## File Layout

```
src/zmlx/
  patch/                  # model patching (the main win)
    patterns/moe_mlp.py   # fused MoE expert dispatch
    patterns/swiglu_mlp.py # dense SwiGLU fusion
    __init__.py           # patch(), safety excludes
  kernels/                # 70+ Metal kernels (19 modules)
  matrix/                 # test matrix: catalog, runner, reports
  foundry/                # kernel template evaluation + SFT dataset export
  discover/               # LLM-guided PUCT kernel optimization search
  train/                  # LoRA training CLI
  validate.py             # fidelity + throughput validation
  api.py                  # kernel authoring API
  metal.py                # Metal kernel wrapper
tests/                    # ~670 tests
benchmarks/               # benchmark scripts + repro capsules
configs/                  # training/foundry config YAML files
docs/                     # user-facing documentation
integrations/             # custom MLX primitive patch
```

## Model-Aware Safety

`patch()` auto-detects model family and skips patterns with known issues:
- **Qwen**: `swiglu_mlp` and `residual_norm` break fidelity
- **GLM/Qwen on stock MLX**: `moe_mlp` regresses (needs custom primitive)
- **Mixtral**: `moe_mlp` breaks fidelity

See `_FIDELITY_EXCLUDES` and `_PERF_EXCLUDES` in `src/zmlx/patch/__init__.py`.
