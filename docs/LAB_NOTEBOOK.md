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
