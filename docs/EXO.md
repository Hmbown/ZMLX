# Using ZMLX with exo (GLM-4.7-Flash / Qwen3-30B-A3B)

ZMLX integrates with [exo](https://github.com/exo-explore/exo) to speed up MoE decode for GLM-4.7-Flash and Qwen3-30B-A3B. The speedup comes from `gather_qmm_swiglu`, a fused Metal primitive that replaces multiple kernel launches per MoE expert per layer with a single dispatch.

**What to expect:** speedups vary by prompt/length/hardware; see the repro capsules referenced below. Output should remain token-identical under greedy decoding (verify with `python -m zmlx.validate`).

## Quick start

In a **Python 3.13+** environment (exo requires >= 3.13):

```bash
# From a ZMLX checkout (recommended):
bash setup_zmlx.sh
bash exo/run_zmlx.sh

# If `exo` is already installed in your environment:
#   pip install zmlx
#   zmlx-exo
```

Then open `http://localhost:52416` in your browser and select GLM-4.7-Flash or Qwen3-30B-A3B.

## Which models benefit?

| Model | With ZMLX in exo | Notes | Capsule |
|:--|:--|:--|:--|
| GLM-4.7-Flash-4bit | Yes (custom primitive) | MoE expert SwiGLU fused via `gather_qmm_swiglu` | [`benchmarks/repro_capsules/glm_stress_m4_20260204.json`](../benchmarks/repro_capsules/glm_stress_m4_20260204.json) |
| Qwen3-30B-A3B-4bit | Yes (custom primitive) | MoE expert SwiGLU fused via `gather_qmm_swiglu` | [`benchmarks/repro_capsules/qwen3_a3b_moe_mlp_m4max_20260205.json`](../benchmarks/repro_capsules/qwen3_a3b_moe_mlp_m4max_20260205.json) |
| LFM2-8B-A1B-4bit | Yes (stock MLX) | No custom MLX build needed | [`benchmarks/repro_capsules/lfm2_m4max_20260131.json`](../benchmarks/repro_capsules/lfm2_m4max_20260131.json) |
| Other models | No change | `patch()` auto-skips; safe no-op | — |

GLM and Qwen3 require the custom MLX primitive (`gather_qmm_swiglu`). Without it, ZMLX auto-skips these models — no regressions, but no speedup either.

## Prerequisites

- **macOS 14+** on Apple Silicon (M1 or later)
- **Python 3.13+** (exo requires >= 3.13). If you build a custom MLX, it must match the Python minor version you run exo with.
- **uv** (recommended) — `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Custom MLX primitive (optional, for GLM/Qwen3)

GLM and Qwen3 gains require `mx.gather_qmm_swiglu`, which is not in released MLX. See [`docs/EXPERIMENTAL_MLX.md`](EXPERIMENTAL_MLX.md) for full details.

Recommended:

```bash
bash integrations/mlx_local_integration/setup_mlx_local.sh
```

Verify the custom primitive is active:

```bash
python -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))"  # should print True
```

To run exo with the custom build without replacing your default MLX install, prepend `mlx_local/python` on `PYTHONPATH`:

```bash
export PYTHONPATH=<REPO_ROOT>/mlx_local/python:$PYTHONPATH
zmlx-exo
```

Remove `mlx_local/python` from `PYTHONPATH` to revert to stock MLX.

## How it works

The launcher (`zmlx-exo` / `python -m zmlx.exo`) installs a runtime hook on exo's MLX model loading path (`exo.worker.engines.mlx.utils_mlx.load_mlx_items`). After exo loads an MLX model, the hook calls `zmlx.patch()` which:

1. Detects the model family (GLM, Qwen3, LFM2, etc.)
2. Checks which patterns are safe for this architecture
3. Replaces matching MoE layers with fused kernel equivalents
4. Returns the patched model — all subsequent generation uses fused paths

The hook is controlled by the `EXO_ZMLX` environment variable:
- `EXO_ZMLX=1` — enable patching
- Unset or `0` — exo runs normally, no ZMLX code is imported

Additional env vars:
- `EXO_ZMLX_VERBOSE=1` — log every patched module
- `EXO_ZMLX_PATTERNS=moe_mlp,swiglu_mlp` — override auto-detected patterns
- `EXO_ZMLX_EXCLUDE=moe_mlp` — exclude specific patterns

## Measured results

| Model | Hardware | Baseline | Patched | Change | Capsule |
|:--|:--|--:|--:|--:|:--|
| GLM-4.7-Flash-4bit | M4 Max 36 GB | 86.6 tok/s | 92.4 tok/s | +6.7% | [`benchmarks/repro_capsules/glm47_flash_control_m4max_20260205.json`](../benchmarks/repro_capsules/glm47_flash_control_m4max_20260205.json) |
| Qwen3-30B-A3B-4bit | M4 Max 36 GB | 106.6 tok/s | 115.0 tok/s | +7.9% | [`benchmarks/repro_capsules/qwen3_a3b_moe_mlp_m4max_20260205.json`](../benchmarks/repro_capsules/qwen3_a3b_moe_mlp_m4max_20260205.json) |

All results token-identical under greedy decoding.

## Multi-device notes

| Mode | MoE fusions | SwiGLU fusions | Notes |
|:--|:--|:--|:--|
| Single device | Yes | Yes | Best case — all patterns active |
| Pipeline parallel | Yes | Yes | Each rank patches its own layers |
| Tensor parallel | **No** (auto-excluded) | Yes | exo's ShardedMoE handles distribution; ZMLX would bypass the all-reduce |

## Troubleshooting

| Symptom | Fix |
|:--|:--|
| No `[zmlx.exo]` log lines after model load | Use `zmlx-exo` / `python -m zmlx.exo` (it installs the hook). If launching exo directly, ZMLX will not be loaded. |
| `Patched 0 modules` on GLM/Qwen | Custom MLX not active. Run `python -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))"` — should print `True`. |
| `ModuleNotFoundError: No module named 'zmlx'` | ZMLX not installed in the exo environment. Run `pip install zmlx` (or use an editable install). |
| `Error: couldn't import exo.main` | `exo` may not be installed, or your current directory contains an `exo/` folder shadowing the Python package. Run from a different directory or install exo into the current environment. |
| Port 52415 in use | Another process (IDE, previous exo) is using the port. The launcher uses 52416 by default, or pass `--api-port <N>`. |
| Line continuation breaks on paste | Prefer `zmlx-exo` over long multi-line commands. |

## Verify token fidelity

```bash
python -m zmlx.validate mlx-community/GLM-4.7-Flash-4bit --max-tokens 128 --runs 5
```

This compares patched vs unpatched output token-by-token and reports throughput.
