# Next-AI Prompt (Isolation-First + Custom-MLX Pivot)

Use this prompt verbatim with the next coding agent.

---

You are working in the ZMLX repo. Continue the custom-MLX-first pivot, but use
strict isolation-first benchmarking for GLM/Qwen3 to avoid stale conclusions and
OOM-prone methodology.

## Mission
Produce high-confidence, reproducible benchmark evidence for:
- `mlx-community/GLM-4.7-Flash-4bit-mxfp4`
- `mlx-community/Qwen3-30B-A3B-4bit`

without fidelity regressions.

## Hard constraints
- Do not fabricate benchmark numbers.
- Every claim must be backed by repro capsules in `benchmarks/repro_capsules/`.
- Use in-repo tracking:
  - notebook: `benchmarks/LAB_NOTEBOOK.md`
  - matrix ledger: `benchmarks/matrix.jsonl`
- Do not edit external clones under `mlx_local/` or `exo/`.
- Run with venv active:
  - `source .venv/bin/activate`
- Before finalizing any change set, run:
  - `ruff check .`
  - `pytest -q`

## Current benchmark truth to preserve
- GLM:
  - active default path: `glm_combine_fp32_no_fma`
  - status: `promote_candidate` (not fully promoted)
- Qwen:
  - benchmark anchor: `control_patterns_moe_mlp`
  - no custom-kernel variant is currently promoted
- Rejected variants to keep rejected unless new evidence reverses them:
  - `qwen_combine_exact`
  - `qwen_router_argpartition_logits_topk_combine_exact`
  - `glm_combine_fp32_no_fma_shared_overlap_streams2`
  - `glm_combine_fp32_no_fma_glm47_rope`

## Benchmark methodology (mandatory)
- Use `benchmarks/bench_iso_variant_sweep.py` for GLM/Qwen3 variant sweeps.
- One variant per subprocess only.
- Use AB/BA replicate blocks with cooldown windows for GLM consistency checks.
- For matrix controls, run `python -m zmlx.matrix run ... --patterns none`.
- Do not reintroduce multi-variant single-process sweeps for GLM/Qwen3.

## Campaign command pattern
Run one phase at a time (recommended):

```bash
source .venv/bin/activate
bash benchmarks/run_3hr_benchmark_campaign.sh quick
bash benchmarks/run_3hr_benchmark_campaign.sh glm_abba_200
bash benchmarks/run_3hr_benchmark_campaign.sh glm_abba_1024
bash benchmarks/run_3hr_benchmark_campaign.sh qwen_isolation_200
bash benchmarks/run_3hr_benchmark_campaign.sh qwen_isolation_1024
bash benchmarks/run_3hr_benchmark_campaign.sh glm_stress
bash benchmarks/run_3hr_benchmark_campaign.sh matrix_controls
```

## Custom-MLX kernel priorities (C++/Metal-first)
1. `moe_router_argpartition_logits_topk`
- Goal: fuse top-k selection + normalization for Qwen-style routing.
- Guardrail: exact semantic parity for selected indices and normalized weights.

2. `moe_gather_qmm_swiglu_downproj_combine`
- Goal: fuse gather -> qmm swiglu -> downproj -> weighted combine.
- Guardrail: deterministic accumulation mode option (`fp32_no_fma`).

3. `moe_combine_weighted_sum_fp32_no_fma` specialization
- Goal: optimize decode-typical small-k combine.
- Guardrail: strict token-level fidelity checks against control.

## Deliverables
- Ranked variant table with decode delta, prefill delta, fidelity verdict, memory delta.
- Explicit recommendation for GLM and Qwen (promote/hold/reject).
- Reject list with concrete failure reasons and capsule paths.
- Updated `benchmarks/LAB_NOTEBOOK.md` entry with exact commands and artifacts.

Proceed immediately and execute, not just plan.

---
