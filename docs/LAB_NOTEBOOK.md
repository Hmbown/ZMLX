# ZMLX Lab Notebook

Purpose: maintain a complete, professional, reproducible engineering log for
all benchmark and kernel work.

## Notebook Standards

- Record every experimental run that informs decisions.
- Use objective language only; no hype, no speculative claims without data.
- Attach exact commands, model IDs, token counts, run counts, and environment.
- Link evidence: `benchmarks/matrix.jsonl`, repro capsules, and commit IDs.
- If a run is discarded (OOM, concurrent contamination, bad config), record why.
- If fidelity fails, record it explicitly and do not present speed numbers as wins.
- Keep entries append-only; corrections are new entries referencing prior ones.

## Entry Template

```md
## YYYY-MM-DD HH:MM UTC — Title
- Goal:
- Code/Config changed:
- Environment:
  - macOS:
  - MLX:
  - Python:
  - Custom MLX primitive:
- Commands:
  - `...`
- Results:
  - Model:
  - Fidelity:
  - Baseline decode:
  - Patched decode:
  - Speedup:
- Evidence:
  - `benchmarks/matrix.jsonl` timestamp(s):
  - `benchmarks/repro_capsules/...json`
  - commit:
- Decision:
```

## 2026-02-08 22:35 UTC — README benchmark cleanup and documentation alignment

- Goal:
  - Align top-level README with current MLX behavior and latest matrix-backed
    measurements.
- Code/Config changed:
  - Updated `README.md` benchmark sections and removed stale exploratory blocks.
  - Added this notebook and logged the change.
- Environment:
  - macOS: 26.1
  - MLX: `0.30.7.dev20260207+8fe1d092` (from matrix entries used)
  - Python: 3.14.2
  - Custom MLX primitive: enabled for GLM/Qwen3 benchmark rows cited.
- Commands:
  - `source .venv/bin/activate && python - <<'PY' ...` (parsed latest matrix rows)
  - `source .venv/bin/activate && ruff check . && pytest -q`
- Results:
  - README now surfaces 2026-02-08 4-bit sequential MoE snapshot:
    - LFM2-8B-A1B-4bit: `209.79 -> 235.68` (`1.123x`, PASS)
    - GLM-4.7-Flash-4bit: `74.54 -> 78.57` (`1.054x`, PASS)
    - Qwen3-30B-A3B-4bit: `103.27 -> 106.26` (`1.029x`, PASS)
  - GLM 200-token revalidation kept:
    - `82.23 -> 89.63` (`1.090x`, PASS)
  - Test/lint status:
    - `ruff check .`: PASS
    - `pytest -q`: `846 passed, 74 skipped, 3 xfailed`
- Evidence:
  - `benchmarks/matrix.jsonl` timestamps:
    - `2026-02-08T22:16:36Z`
    - `2026-02-08T22:22:53Z`
    - `2026-02-08T22:24:10Z`
    - `2026-02-08T22:25:19Z`
- Decision:
  - Keep stress-benchmark numbers as historical context only.
  - Keep top-level claims tied to latest matrix rows.

## 2026-02-08 23:10 UTC — Kimi K2.5 readiness update

- Goal:
  - Ensure DeepSeek/Kimi router fusion path is current for Kimi K2.5 and
    version-tracked with reproducible references.
- Code/Config changed:
  - Updated `src/zmlx/patch/patterns/deepseek_router.py` matcher to include
    `mlx_lm.models.kimi_k25` module path.
  - Fixed decimal quant parsing in `src/zmlx/matrix/models.py` (`3.6bit`,
    `4.2bit`).
  - Updated Kimi/DeepSeek docs:
    - `docs/DEEPSEEK_KIMI_ROUTER_FUSION.md`
    - `docs/HANDOFF_DEEPSEEK_KIMI.md`
  - Added optional Kimi entry to `benchmarks/moe_models.txt`.
  - Added regression tests:
    - `tests/test_deepseek_router_pattern.py`
    - `tests/test_matrix_models.py`
- Environment:
  - Local `mlx-lm`: `0.30.6`
  - Local `mlx`: `0.30.4.dev20260204+2f324cc`
  - Verified upstream package latest on PyPI (2026-02-08):
    - `mlx-lm==0.30.6`
    - `mlx==0.30.6`
- External model checks (HF API):
  - `moonshotai/Kimi-K2.5` last modified `2026-02-05`
  - `mlx-community/Kimi-K2.5` last modified `2026-01-27`
- Commands:
  - `python - <<'PY' ...` (HF API + PyPI metadata checks)
  - `python -m zmlx.matrix catalog | rg kimi`
- Decision:
  - Treat `mlx-community/Kimi-K2.5` as the default MLX validation target for
    K2.5 in current docs.
  - Keep `deepseek_router` opt-in until model-level fidelity/throughput runs are
    captured in matrix/capsules.

## 2026-02-09 00:20 UTC — Custom MLX 0.30.6 bring-up + GLM/Qwen smoke validation

- Goal:
  - Move custom `gather_qmm_swiglu` path to MLX 0.30.6 and confirm current
    decode fidelity/perf on GLM-4.7 and Qwen3-30B-A3B.
- Code/Config changed:
  - Updated `integrations/mlx_local_integration/setup_mlx_local.sh` default
    `MLX_REF` to `185b06d9efc1c869540eccfb5baff853fff3659d` (`v0.30.6`).
  - Re-based patch check against `v0.30.6` and rebuilt custom MLX from source.
  - Installed editable custom MLX into `.venv` from `/tmp/mlx_0306_update`.
- Environment:
  - Python: 3.14.2
  - `mlx-lm`: 0.30.6
  - Active MLX core: `0.30.6.dev20260208+185b06d9`
  - `mx.gather_qmm_swiglu`: present (`True`)
- Commands:
  - `python -m zmlx.validate mlx-community/GLM-4.7-Flash-4bit --max-tokens 200 --runs 3`
  - `python -m zmlx.validate mlx-community/Qwen3-30B-A3B-4bit --max-tokens 200 --runs 3`
- Results:
  - GLM-4.7-Flash-4bit:
    - Fidelity: `200/200` PASS
    - Decode: `83.1 -> 89.8 tok/s` (`1.081x`)
  - Qwen3-30B-A3B-4bit:
    - Fidelity: `200/200` PASS
    - Decode: `116.1 -> 118.4 tok/s` (`1.020x`)
- Notes:
  - Pip reports an exo pin mismatch (`exo 0.3.0` requires `mlx==0.30.4`). This
    does not affect direct `zmlx.validate` runs, but exo env pinning must be
    updated separately for 0.30.6 custom MLX.
- Decision:
  - Custom MLX 0.30.6 path is functional and beneficial on both target models.
  - Proceed to keep 0.30.6 as the default custom-MLX setup ref.
