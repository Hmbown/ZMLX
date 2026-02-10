# ZMLX Lab Notebook

This notebook records kernel experiments, decisions, and pivot notes inside the ZMLX repo.

## 2026-02-10: Custom-MLX Location Map

- Custom MLX patch source: `integrations/mlx_local_integration/gather_qmm_swiglu.patch`
- Custom MLX setup/build script: `integrations/mlx_local_integration/setup_mlx_local.sh`
- Custom MLX workflow docs: `docs/EXPERIMENTAL_MLX.md`
- Optional local checkout target (gitignored): `<REPO_ROOT>/mlx_local`
- Runtime probe for custom primitive availability:
  - `python -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))"`
- Current setup script MLX pin:
  - `MLX_REF` in `integrations/mlx_local_integration/setup_mlx_local.sh`

## EXP-20260210T233500Z-BEST-V8-PROMOTION

Question:
What is the best combined custom-MLX + ZMLX kernel stack to promote after v7/v8 retests?

Code change promoted:
- `src/zmlx/kernels/moe.py`
  - Added threadgroup-staged weight loads for `moe_combine_fp32_no_fma`.
  - Reverted the same staging from `moe_combine`, `moe_combine_no_fma`, `moe_combine_exact`, and `moe_combine_fp32`.
  - Rationale: this preserved Qwen exact-combine behavior while improving GLM fp32-no-fma combine.

Primary v8 capsules:
- `benchmarks/repro_capsules/qwen3_a3b_combo_v8_fp32nofmaonly_t200_r2_summary.json`
- `benchmarks/repro_capsules/qwen3_a3b_combo_v8_fp32nofmaonly_t1024_r2_summary.json`
- `benchmarks/repro_capsules/glm47_combo_v8_fp32nofmaonly_t200_r2_summary.json`
- `benchmarks/repro_capsules/glm47_combo_v8_fp32nofmaonly_t1024_r2_summary.json`

Promoted results (all fidelity PASS):
- Qwen3-30B-A3B-4bit:
  - `qwen_combine_exact`: `1.0549x` @200, `1.0551x` @1024
- GLM-4.7-Flash-4bit:
  - `glm_combine_fp32_no_fma`: `1.0619x` @200, `1.0667x` @1024

Memory notes:
- Qwen remained ~`17.24 GB` (@200) and ~`17.33/17.34 GB` (@1024).
- GLM remained ~`16.91 GB` (@200) and ~`16.94/16.95 GB` (@1024).

Promotion decision:
- Promote v8 kernel behavior as the new best-known cross-model setting:
  - custom MLX primitive: `gather_qmm_swiglu` (from `integrations/mlx_local_integration/`)
  - ZMLX MoE combine optimization: threadgroup-staged weights only for `fp32_no_fma` combine.

## EXP-20260210T145900Z-PIVOT-NOTE

Observation:
- We have been iterating heavily in ZMLX patch logic.
- For larger gains and cleaner upstreamability, next effort should prioritize custom MLX primitive/kernel work (C++/Metal) and then thin ZMLX integration, not the other way around.

Pivot direction:
- Focus on custom MLX kernel authoring first.
- Keep ZMLX changes minimal: capability detection + model-family-safe wiring.

## EXP-20260210T160100Z-MEMORY-BENCH-COVERAGE

What current benchmark scripts record:
- `benchmarks/bench_qwen3_a3b_experiments.py` and `benchmarks/bench_glm47_flash_experiments.py` include `peak_mem_gb` in per-variant outputs and summary artifacts.
- `benchmarks/bench_iso_variant_sweep.py` records baseline/patched memory deltas from `peak_mem_gb`.
- `benchmarks/matrix.jsonl` receives memory fields for each run (through benchmark writer).

What this does and does not capture:
- Captures:
  - process-level peak memory during the benchmark window
  - per-variant baseline vs patched peak-memory comparison
- Does not yet capture:
  - transient per-kernel temporary buffer footprints at kernel-dispatch granularity
  - unified memory page-fault pressure and migration count
  - KV-cache fragmentation behavior over very long contexts

Decision:
- Keep `peak_mem_gb` as the default matrix metric.
- Add targeted microbench/profiler passes when a variant shows speedup with suspicious memory behavior.

## EXP-20260210T160300Z-NEXT-KERNELS-FROM-SCRATCH

Highest-impact next custom kernels to try (C++/Metal-first):
- `moe_router_argpartition_logits_topk`:
  - fuse: select top-k logits + normalize selected logits (Qwen path)
  - expected gain: lower router overhead at decode batch size 1-4
  - risk: index-order semantics; must preserve Qwen fidelity behavior
- `moe_gather_qmm_swiglu_downproj_combine`:
  - single primitive for gather -> qmm swiglu -> downproj -> weighted combine
  - expected gain: remove intermediate writes and launch overhead
  - risk: numeric drift from accumulation order; keep fp32/no_fma modes
- `moe_combine_weighted_sum_fp32_no_fma` specialization:
  - optimized combine kernel for small-k (k=8) decode with deterministic accumulation path
  - expected gain: portable benefit for Qwen and GLM
  - risk: tune thresholds by model shape to avoid regressions

See handoff prompt for implementation workflow:
- `benchmarks/NEXT_AI_PIVOT_PROMPT.md`

## EXP-20260210T182700Z-V9-REPRO-CONTROLS

Question:
Can we reproduce current v8 control/candidate behavior with isolated per-variant runs?

Primary capsules:
- `benchmarks/repro_capsules/qwen3_combo_v9_repro_t200_r3_summary.json`
- `benchmarks/repro_capsules/qwen3_combo_v9_repro_t1024_r2_summary.json`
- `benchmarks/repro_capsules/glm47_combo_v9_repro_t200_r3_summary.json`
- `benchmarks/repro_capsules/glm47_combo_v9_repro_t1024_r2_summary.json`

Key results (all fidelity PASS):
- Qwen (`control_patterns_moe_mlp` vs `qwen_combine_exact`)
  - t=200 r=3: `1.0345x` vs `1.0383x`
  - t=1024 r=2: `1.0232x` vs `1.0717x`
- GLM (`control_swiglu_moe` vs `glm_combine_fp32_no_fma`)
  - t=200 r=3: `1.0374x` vs `1.0327x`
  - t=1024 r=2: `1.0388x` vs `1.0508x`

Memory notes:
- Qwen stayed at ~`17.24 GB` (t=200) and ~`17.33/17.34 GB` (t=1024).
- GLM stayed at ~`16.91 GB` (t=200) and ~`16.94/16.95 GB` (t=1024).

Interpretation:
- Short-run medians show expected decode variance; v8 promoted capsules remain the
  canonical promotion reference for now.

## EXP-20260210T183900Z-KERNEL1-ROUTER-ARGPARTITION-LOGITS-TOPK

Question:
Does a fused `argpartition(logits) -> top-k softmax` router path improve Qwen
decode when combined with the current best `qwen_combine_exact`?

Code changes:
- `src/zmlx/kernels/moe.py`
  - added `router_argpartition_logits_topk()` and fused Metal kernel
    (`kk_router_argpartition_logits_topk_D{D}_K{K}`)
- `src/zmlx/patch/patterns/moe_mlp.py`
  - added env-gated Qwen router branch:
    - `ZMLX_QWEN_ROUTER_ARGPARTITION_LOGITS_TOPK=1`
    - requires existing `ZMLX_QWEN_ROUTER_ARGPARTITION_LOGITS=1`
- `benchmarks/bench_qwen3_a3b_experiments.py`
  - added variants:
    - `qwen_router_argpartition_logits_topk`
    - `qwen_router_argpartition_logits_topk_combine_exact`

Primary capsules:
- `benchmarks/repro_capsules/qwen3_combo_v9_routertopk_t200_r3_summary.json`
- `benchmarks/repro_capsules/qwen3_combo_v9_routertopk_t1024_r2_summary.json`

Key results (all fidelity PASS):
- t=200 r=3:
  - control: `1.0060x` (`117.4 -> 118.1`)
  - prior best in-suite (`qwen_combine_exact`): `1.0442x` (`117.7 -> 122.9`)
  - new candidate (`qwen_router_argpartition_logits_topk_combine_exact`):
    `1.0544x` (`117.6 -> 124.0`)
  - incremental vs control: `+0.0484x` speedup points
  - incremental vs prior best in-suite: `+0.0102x`
- t=1024 r=2:
  - control: `1.0222x` (`112.7 -> 115.2`)
  - prior best in-suite (`qwen_combine_exact`): `1.0460x` (`113.1 -> 118.3`)
  - new candidate: `1.0408x` (`112.8 -> 117.4`)
  - incremental vs control: `+0.0186x`
  - vs prior best in-suite: `-0.0052x` (regression)

Memory notes:
- No measurable memory regression (`17.24 GB` at 200; `17.33/17.34 GB` at 1024).

Decision:
- Keep kernel #1 path experimental and env-gated only.
- Do not promote over `qwen_combine_exact` yet due long-context regression.

## EXP-20260210T184800Z-KERNEL1-LONG-CONFIRM

Question:
Is the long-context (`1024`) gap between `qwen_combine_exact` and the new
router-topk candidate just run noise?

Primary capsule:
- `benchmarks/repro_capsules/qwen3_combo_v9_routertopk_t1024_r3_confirm_summary.json`

Results (all fidelity PASS, t=1024 r=3):
- control: `1.0341x` (`111.3 -> 115.1`)
- `qwen_combine_exact`: `1.0614x` (`112.4 -> 119.3`)
- `qwen_router_argpartition_logits_topk_combine_exact`: `1.0495x` (`113.1 -> 118.7`)

Interpretation:
- The candidate remains below `qwen_combine_exact` on long decode in the
  confirm run (`-0.0119x` speedup points in-suite).
- Keep the new router-topk path experimental only; no promotion yet.
