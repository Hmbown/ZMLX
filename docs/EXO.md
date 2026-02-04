# Using ZMLX with exo (GLM-4.7-Flash / Qwen3-30B-A3B)

ZMLX integrates with [exo](https://github.com/exo-explore/exo) to speed up MoE decode for GLM-4.7-Flash and Qwen3-30B-A3B. The speedup comes from `gather_qmm_swiglu`, a fused Metal primitive that replaces multiple kernel launches per MoE expert per layer with a single dispatch.

**What to expect:** ~8% faster GLM decode, ~6% faster Qwen3 decode, token-identical output.

## Which models benefit?

| Model | With ZMLX in exo | Notes |
|:--|:--|:--|
| GLM-4.7-Flash-4bit | **+8% decode** | 46 MoE layers + 1 dense SwiGLU fused |
| Qwen3-30B-A3B-4bit | **+6% decode** | MoE expert dispatch fused |
| LFM2-8B-A1B-4bit | **+5-12% decode** | Works on stock MLX too (no custom build needed) |
| Other models | No change | `patch()` auto-skips; safe no-op |

GLM and Qwen3 require the custom MLX primitive (`gather_qmm_swiglu`). Without it, ZMLX auto-skips these models — no regressions, but no speedup either.

## Prerequisites

- **macOS 14+** on Apple Silicon (M1 or later)
- **Python 3.13+** (exo requires >= 3.13). If you build a custom MLX, it must match the Python minor version you run exo with.
- **uv** (recommended) — `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Setup (one command)

```bash
git clone https://github.com/Hmbown/ZMLX.git
cd ZMLX
bash setup_zmlx.sh
```

The script:
1. Clones exo into `exo/` (local-only; gitignored)
2. Applies a small hook patch so exo calls `zmlx.patch()` after model load
3. Creates a Python venv in `exo/.venv/`
4. Installs exo and ZMLX (editable)
5. If `mlx_local/python` exists, wires it into the exo venv via a `.pth` file
6. Creates `exo/run_zmlx.sh` launcher and verifies the install

### Build the custom MLX first (if not already built)

GLM and Qwen3 gains require `mx.gather_qmm_swiglu`, which is not in released MLX.

- Recommended (creates `mlx_local/`, applies the patch, and builds):

```bash
bash integrations/mlx_local_integration/setup_mlx_local.sh
```

- Manual: see [`docs/EXPERIMENTAL_MLX.md`](EXPERIMENTAL_MLX.md).

After building, re-run `bash setup_zmlx.sh` so exo's venv picks up `mlx_local/python`.

## Launch

```bash
bash exo/run_zmlx.sh
```

Then open `http://localhost:52416` in your browser and select GLM-4.7-Flash or Qwen3-30B-A3B.

When the model loads, the terminal should show:

```
[zmlx] Applying fused-kernel patches (distributed=False, parallelism=none)...
[zmlx] Patched 47 modules in 0.02s: {'swiglu_mlp': 1, 'moe_mlp': 46}
```

## How it works

The integration is a single hook in exo's model loading path (`src/exo/worker/engines/mlx/utils_mlx.py`). After exo loads an MLX model, the hook calls `zmlx.patch()` which:

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

| Model | Hardware | Baseline | Patched | Change |
|:--|:--|--:|--:|--:|
| GLM-4.7-Flash-4bit | M4 Max 36 GB | 85.8 tok/s | 92.8 tok/s | **+8.1%** |
| GLM-4.7-Flash-4bit | M4 Mac Studio | ~77 tok/s | ~83 tok/s | **+8%** |
| Qwen3-30B-A3B-4bit | M4 Max 36 GB | 117 tok/s | 123 tok/s | **+5.5%** |

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
| No `[zmlx]` log lines after model load | `EXO_ZMLX=1` not set. Use the `run_zmlx.sh` script or `export EXO_ZMLX=1` before launching. |
| `Patched 0 modules` on GLM/Qwen | Custom MLX not active. Run `python -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))"` — should print `True`. |
| `ModuleNotFoundError: No module named 'zmlx'` | ZMLX not installed in the exo venv. Re-run `setup_zmlx.sh`. |
| Port 52415 in use | Another process (IDE, previous exo) is using the port. The script uses 52416 instead, or pass `--api-port <N>`. |
| Line continuation breaks on paste | Use the `run_zmlx.sh` script instead of pasting multi-line commands. |

## Verify token fidelity

From the ZMLX venv (not exo's):

```bash
cd ZMLX
source .venv/bin/activate
python -m zmlx.validate mlx-community/GLM-4.7-Flash-4bit --max-tokens 128 --runs 5
```

This compares patched vs unpatched output token-by-token and reports throughput.
