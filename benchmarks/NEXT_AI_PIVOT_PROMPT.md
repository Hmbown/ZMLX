# Next-AI Pivot Prompt (Custom MLX First)

Use this prompt verbatim with the next coding agent.

---

You are working in the ZMLX repo. We need to pivot from patch-heavy ZMLX experiments to custom MLX kernel authoring (C++/Metal) first, then thin ZMLX wiring.

## Mission
Deliver the highest-confidence decode throughput gains for:
- Qwen3-30B-A3B-4bit
- GLM-4.7-Flash-4bit

without fidelity regressions.

## Hard constraints
- Do not fabricate benchmark numbers.
- Every claim must be backed by repro capsules under `benchmarks/repro_capsules/`.
- Use in-repo notebook/matrix tracking:
  - notebook: `benchmarks/LAB_NOTEBOOK.md`
  - matrix ledger: `benchmarks/matrix.jsonl`
- Do not edit external clones under `mlx_local/` or `exo/` directly unless required by the integration workflow.
- Run with venv active:
  - `source .venv/bin/activate`
- Before finalizing any change set, run:
  - `ruff check .`
  - `pytest -q`

## Existing findings to preserve
- Qwen best validated combo currently includes combine-exact path (see latest qwen v6 confirm capsules).
- New GLM override work shows:
  - `ZMLX_GLM_COMBINE_MODE=fp32_no_fma` is fidelity-safe and beneficial.
  - `exact` and `fp32` combine modes fail fidelity for GLM.

## Where custom MLX lives
- Patch source: `integrations/mlx_local_integration/gather_qmm_swiglu.patch`
- Setup/build workflow: `integrations/mlx_local_integration/setup_mlx_local.sh`
- Workflow docs: `docs/EXPERIMENTAL_MLX.md`

## Priority kernel targets (C++/Metal-first)
1. `moe_router_argpartition_logits_topk`
- Goal: fuse top-k selection on logits + normalization over selected logits for Qwen-style routing.
- Why: router overhead is still a bottleneck at decode batch sizes 1-4.
- Fidelity guardrail: preserve argpartition/top-k semantics exactly.

2. `moe_gather_qmm_swiglu_downproj_combine`
- Goal: one primitive for gather -> qmm swiglu -> downproj -> weighted combine.
- Why: remove intermediate allocations and launch overhead.
- Fidelity guardrail: deterministic accumulation mode (`fp32_no_fma`) option.

3. `moe_combine_weighted_sum_fp32_no_fma` specialization
- Goal: optimize small-k combine for decode (k=8 common case).
- Why: this appears portable across Qwen and GLM.
- Fidelity guardrail: strict token-level comparison vs baseline.

## Execution plan
1. Implement one kernel at a time in custom MLX (C++/Metal), wire capability detection.
2. Keep ZMLX-side change minimal and opt-in with env flags for A/B safety.
3. Benchmark each variant isolated, then in controlled combos.
4. For each run, write:
- repro capsule JSON in `benchmarks/repro_capsules/`
- matrix entry in `benchmarks/matrix.jsonl`
- short experiment note in `benchmarks/LAB_NOTEBOOK.md`
5. Promote only variants that satisfy both:
- token fidelity pass at configured token budget
- repeatable decode speedup across short and long context settings

## Benchmark minimums
For every candidate kernel and combo:
- short decode check: `tokens=200`, `runs>=3`
- long decode check: `tokens=1024`, `runs>=2`
- capture and compare `peak_mem_gb`
- include baseline and patched runs on same hardware session

## Deliverables
- Ranked table of tested variants by decode speedup and fidelity status.
- Explicit "best combo" recommendation for Qwen3 and GLM (may differ).
- Clear reject list with failure reason (fidelity, perf regression, or memory regression).
- Updated notebook + matrix + repro capsules.

Proceed immediately; do not stop at planning.

---
