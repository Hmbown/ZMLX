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

## EXP-20260210T203517Z-TAKEOVER-COMBO-MAX

Hypothesis:
- Existing Qwen/GLM variants might still yield additive decode gains in isolated
  sweeps, and missing GLM additive combos (`fp32_no_fma + overlap`, `fp32_no_fma
  + glm47_rope`) could improve on current controls.

Code delta:
- `benchmarks/bench_glm47_flash_experiments.py`
  - Added variants:
    - `glm_combine_fp32_no_fma_shared_overlap_streams2`
    - `glm_combine_fp32_no_fma_glm47_rope`
  - Updated `--variants` help text with new names.

Primary artifacts:
- `benchmarks/repro_capsules/qwen3_takeover_combo_t200_r3_summary.json`
- `benchmarks/repro_capsules/qwen3_takeover_combo_t1024_r2_summary.json`
- `benchmarks/repro_capsules/glm47_takeover_combo_t200_r3_summary.json`
- `benchmarks/repro_capsules/glm47_takeover_combo_t1024_r2_summary.json`
- `benchmarks/repro_capsules/glm47_takeover_additive_combo_t200_r3_summary.json`
- `benchmarks/repro_capsules/glm47_takeover_additive_combo_t1024_r2_summary.json`

Results (all listed variants fidelity PASS, no material memory regression):

- Qwen takeover sweep:
  - t=200 r=3:
    - `control_patterns_moe_mlp`: `0.9719x`
    - `qwen_combine_exact`: `0.9992x`
    - `qwen_router_argpartition_logits_topk_combine_exact`: `1.0051x`
  - t=1024 r=2:
    - `control_patterns_moe_mlp`: `0.9884x`
    - `qwen_combine_exact`: `1.0071x`
    - `qwen_router_argpartition_logits_topk_combine_exact`: `1.0071x`

- GLM takeover sweep:
  - t=200 r=3:
    - `control_swiglu_moe`: `0.9660x`
    - `glm_combine_fp32_no_fma`: `1.0206x`
    - `shared_experts_overlap_streams2`: `0.6827x`
    - `glm47_rope`: `0.9776x`
  - t=1024 r=2:
    - `control_swiglu_moe`: `1.0368x`
    - `glm_combine_fp32_no_fma`: `1.0369x`
    - `shared_experts_overlap_streams2`: `0.6947x`
    - `glm47_rope`: `0.9834x`

- GLM additive follow-up sweep:
  - t=200 r=3:
    - `control_swiglu_moe`: `1.0177x`
    - `glm_combine_fp32_no_fma_shared_overlap_streams2`: `0.6959x`
    - `glm_combine_fp32_no_fma_glm47_rope`: `1.0047x`
  - t=1024 r=2:
    - `control_swiglu_moe`: `1.0282x`
    - `glm_combine_fp32_no_fma_shared_overlap_streams2`: `0.7070x`
    - `glm_combine_fp32_no_fma_glm47_rope`: `0.9784x`

Decision:
- Qwen:
  - `qwen_combine_exact`: keep experimental in this cycle (no promote).
  - `qwen_router_argpartition_logits_topk_combine_exact`: keep experimental in
    this cycle (no promote).
  - Reason: despite positive incremental deltas vs in-run control, both are far
    below current promoted long-context reference (`1.0551x`).
- GLM:
  - `glm_combine_fp32_no_fma`: keep current promoted reference unchanged (no
    new promote in this cycle).
  - `shared_experts_overlap_streams2`: reject (large short/long regressions).
  - `glm47_rope`: reject (regression vs control at long context).
  - `glm_combine_fp32_no_fma_shared_overlap_streams2`: reject.
  - `glm_combine_fp32_no_fma_glm47_rope`: reject.
  - Reason: no candidate exceeds current promoted long-context reference
    (`1.0667x`) and overlap-based paths are consistently negative.

Next hypothesis:
- Focus on deterministic combine-path kernel improvements (custom-MLX primitive
  and/or foundry-generated combine candidates) while avoiding shared-overlap
  concurrency for GLM decode.

## EXP-20260210T235900Z-RECAL-AND-FOUNDRY-MOE-COMBINE

Question:
- After takeover sweeps, do promoted references still hold under fresh controls,
  and can we bring `moe_combine` online in foundry with first templates?

Code delta:
- `src/zmlx/foundry/ops/moe_combine.py`
  - Added foundry template support for `moe_combine`:
    - templates: `t0_basic`, `t1_k8_unrolled`
    - knob space + validation (`tg_size`, `unroll`, `fast_math`)
    - `extra_shape_dims` for `n_experts`/`k` sampling
    - bytes/FLOPs estimates
- `src/zmlx/foundry/templates/moe_combine/t0_basic.metal`
- `src/zmlx/foundry/templates/moe_combine/t1_k8_unrolled.metal`
- `src/zmlx/foundry/harness/evaluate.py`
  - Added numpy-reference fallback for ops without `reference_mlx`.
  - Preserved integer input buffers in MLX execution path.
  - Cast MLX outputs to float32 before numpy conversion in fallback metrics path.

Primary recalibration artifacts:
- `benchmarks/repro_capsules/qwen3_recal_t200_r3_summary.json`
- `benchmarks/repro_capsules/qwen3_recal_t1024_r2_summary.json`
- `benchmarks/repro_capsules/glm47_recal_t200_r3_summary.json`
- `benchmarks/repro_capsules/glm47_recal_t1024_r2_summary.json`

Recalibration results (all fidelity PASS):
- Qwen:
  - t=200 r=3:
    - `control_patterns_moe_mlp`: `0.9704x`
    - `qwen_combine_exact`: `1.0401x`
  - t=1024 r=2:
    - `control_patterns_moe_mlp`: `0.9964x`
    - `qwen_combine_exact`: `0.9884x`
- GLM:
  - t=200 r=3:
    - `control_swiglu_moe`: `1.0400x`
    - `glm_combine_fp32_no_fma`: `1.0376x`
  - t=1024 r=2:
    - `control_swiglu_moe`: `1.0238x`
    - `glm_combine_fp32_no_fma`: `1.0719x`

Foundry bring-up artifacts:
- `sessions/foundry_moe_combine_smoke_20260210d/attempts.ndjson`
- `sessions/foundry_moe_combine_smoke_20260210_random/attempts.ndjson`

Foundry bring-up outcomes:
- `python -m zmlx.foundry list` now shows:
  - `moe_combine ... t0_basic, t1_k8_unrolled`
- Coverage smoke (`n=8`) confirmed expected harness behavior:
  - `7/8` build-ok, `6/8` correctness-ok, `6/8` bench-ok
  - one intentional compile-error injection case
  - one low-shape correctness outlier for `t1_k8_unrolled` observed under
    coverage-mode knobs
- Random smoke (`n=6`, no fault injection) was clean:
  - `6/6` build-ok, `6/6` correctness-ok, `6/6` bench-ok
  - best p50:
    - `t0_basic`: `0.1556 ms`
    - `t1_k8_unrolled`: `0.2275 ms`

Decision:
- Keep promoted Qwen/GLM references unchanged pending larger-sample recalibration.
- Accept foundry `moe_combine` bring-up as operational (template discovery,
  compile, correctness, and bench path all working in clean random-mode smoke).

## EXP-20260210T213122Z-FOUNDRY-MOE-COMBINE-TARGETED-PAIRWISE-RANKING

Question:
- Between `t0_basic` and `t1_k8_unrolled`, which `moe_combine` template is
  more stable/faster under practical decode-oriented shapes when tested
  pairwise on identical candidates?

Method:
- Unconstrained `mix` (`n=120`) was started and then stopped after observing
  heavy-tail runtime from extreme shapes:
  - partial artifact: `sessions/foundry_moe_combine_mix_20260210_s1/attempts.ndjson`
- Switched to a controlled, larger matched-pair campaign:
  - seeds: `20260210`, `20260211`
  - cases per seed: `72`
  - total attempts: `288` (`2 templates x 72 x 2 seeds`)
  - backend: MLX, correctness tests: `2`, warmup: `3`, repeats: `12`
  - shapes: decode-centric (`batch` in `{1,2,4,8}`, `seq` in `{1,2,4,8,16,32}`,
    hidden in `{768,1024,1536,2048,3072,4096}`, `k` in `{2,4}`)
  - artifacts:
    - `sessions/foundry_moe_combine_targeted_20260210/attempts.ndjson`
    - `sessions/foundry_moe_combine_targeted_20260211/attempts.ndjson`
    - `sessions/foundry_moe_combine_targeted_20260210_combined/attempts.ndjson`
    - `sessions/foundry_moe_combine_targeted_20260210_combined/summary.json`
    - `sessions/foundry_moe_combine_targeted_20260210_combined/report.md`

Results:
- Pairwise valid: `144/144` matched cases.
- Reliability:
  - `t0_basic`: bench-ok `144/144` (`100%`)
  - `t1_k8_unrolled`: bench-ok `144/144` (`100%`)
- Overall latency:
  - `t0_basic` median p50: `1.011396 ms` (IQR `0.023104 ms`)
  - `t1_k8_unrolled` median p50: `1.007125 ms` (IQR `0.026604 ms`)
- Pairwise wins:
  - `t0_basic`: `58` wins (`40.3%`)
  - `t1_k8_unrolled`: `86` wins (`59.7%`)
  - median ratio `t1/t0`: `0.998185` (p10 `0.982510`, p90 `1.019989`)
- Segment behavior:
  - Small/medium tokens (`1..32`): `t1_k8_unrolled` leads.
  - Larger tokens (`33..256`): slight edge to `t0_basic`
    (median `t1/t0 = 1.000436`).

Decision:
- Current decode-oriented default preference: `t1_k8_unrolled` (higher pairwise
  win rate, slightly lower global median p50).
- Keep `t0_basic` as a strong fallback for larger token regimes where unroll-8
  is not consistently beneficial.

## EXP-20260210T214600Z-FOUNDRY-MOE-COMBINE-T2-CONFOUND-CHECK

Question:
- Can we improve on `t0_basic`/`t1_k8_unrolled` with a template that is better
  for medium/large token regimes, and do results hold after order-bias controls?

Code delta:
- `src/zmlx/foundry/ops/moe_combine.py`
  - Added `t2_row_tile` to template list.
- `src/zmlx/foundry/templates/moe_combine/t2_row_tile.metal`
  - New threadgroup-tiled variant that stages `(packed_token_ids, packed_weights)`
    per p-tile to reduce repeated global reads across columns.

Method (scientific controls):
- Added `t2_row_tile` and first ran a 3-template targeted campaign.
- Detected confound risk: fixed per-case template order can bias latency.
- Re-ran with stronger controls:
  1. balanced rotating order per case
  2. random order per case (seeded/reproducible)
- All campaigns kept identical shape/dtype regime and candidate pairing.

Artifacts:
- initial 3-template run:
  - `sessions/foundry_moe_combine_targeted3_20260210_combined/report.md`
- balanced-order run:
  - `sessions/foundry_moe_combine_targeted3_balanced_20260210_combined/report.md`
  - `sessions/foundry_moe_combine_targeted3_balanced_20260210_combined/summary.json`
- random-order run (primary decision basis):
  - `sessions/foundry_moe_combine_targeted3_randorder_20260210_combined/report.md`
  - `sessions/foundry_moe_combine_targeted3_randorder_20260210_combined/summary.json`

Random-order headline results (432 attempts, 144 matched cases):
- Overall median p50:
  - `t0_basic`: `0.311792 ms`
  - `t1_k8_unrolled`: `0.323479 ms`
  - `t2_row_tile`: `0.232416 ms`
- 3-way winner rates:
  - `t0_basic`: `18.1%`
  - `t1_k8_unrolled`: `11.1%`
  - `t2_row_tile`: `70.8%`
- Pairwise wins (n=144):
  - `t0_basic` vs `t2_row_tile`: `40` vs `104`, median ratio `t2/t0=0.807621`
  - `t1_k8_unrolled` vs `t2_row_tile`: `35` vs `109`, median ratio `t2/t1=0.781027`
  - `t0_basic` vs `t1_k8_unrolled`: `116` vs `28`
- Sign-test p-values:
  - `t2` vs `t0`: `9.45e-08`
  - `t2` vs `t1`: `4.86e-10`
  - `t0` vs `t1`: `6.30e-14`

Token-bucket behavior:
- `tokens 33..256`: `t2` wins `100%` (32/32)
- `tokens 9..32`: `t2` wins `95.6%` (43/45)
- `tokens 1..8`: mixed; `t2` wins `40.3%`, `t0` `35.8%`, `t1` `23.9%`
- Threshold probe (`t0` for small tokens, `t2` otherwise):
  - `tokens<=1` gives slight median/mean gain vs always-`t2`
  - broader small-token thresholds (`<=8`, `<=16`) regress median/mean

Decision:
- Promote foundry preference to `t2_row_tile` as the primary `moe_combine`
  search winner under practical decode regimes.
- Keep `t0_basic` as a micro-regime fallback candidate for ultra-small token
  cases (`tokens==1`) in future dispatch-policy experiments.
- Deprecate `t1_k8_unrolled` as default candidate (consistently behind `t0` and `t2`).

Replication extension (holdout seeds):
- Added additional random-order seeds (`20260212`, `20260213`) and aggregated
  all random-order runs (4 seeds total):
  - `sessions/foundry_moe_combine_targeted3_randorder_aggregate_4seeds/report.md`
  - `sessions/foundry_moe_combine_targeted3_randorder_aggregate_4seeds/summary.json`
- Aggregate totals:
  - attempts: `864`
  - matched cases: `288`
- Aggregate winner rates:
  - `t0_basic`: `17.0%` (49/288)
  - `t1_k8_unrolled`: `11.5%` (33/288)
  - `t2_row_tile`: `71.5%` (206/288)
- Aggregate median p50:
  - `t0_basic`: `0.317875 ms`
  - `t1_k8_unrolled`: `0.331270 ms`
  - `t2_row_tile`: `0.250750 ms`
- Aggregate pairwise significance (sign test):
  - `t2` vs `t0`: p=`2.70e-14`
  - `t2` vs `t1`: p=`2.37e-19`
  - `t0` vs `t1`: p=`1.20e-23`
- Token segments (aggregate):
  - `33..256`: `t2` wins `100%`
  - `9..32`: `t2` wins `94.4%`
  - `1..8`: mixed, `t2` still plurality (`44.2%`)

Final interpretation:
- `t2_row_tile` is now strongly replicated as the best default foundry candidate
  across practical decode regimes, with statistically significant wins.

## EXP-20260210T215255Z-COMBINE-MICROBENCH-TO-E2E-TRANSFER

Question:
- Do production `moe_combine*` microbench wins transfer to model-level decode,
  and which combine modes remain fidelity-safe?

Microbench method:
- Synthetic benchmark across production combine kernels:
  - functions: `moe_combine`, `moe_combine_no_fma`, `moe_combine_exact`,
    `moe_combine_fp32`, `moe_combine_fp32_no_fma`
  - grid: `B in {1,2,4,8,16,32,64}`, `K in {2,4}`, `D in {1024,2048,4096}`
  - result artifact: `sessions/moe_combine_microbench_20260210/results.json`

Microbench summary (42 configs):
- median ratio `no_fma/base`: `0.9917` (slightly faster)
- median ratio `exact/base`: `1.0036` (rough parity)
- median ratio `fp32/base`: `1.0231` (slower)
- median ratio `fp32_no_fma/fp32`: `0.9581` (faster)
- wins:
  - `no_fma` faster than base in `26/42`
  - `exact` faster than base in `19/42`
  - `fp32_no_fma` faster than `fp32` in `36/42`

Model-level probe method:
- A full Qwen multi-variant sweep (`runs=2`, `tokens=200`) hit Metal OOM.
- Switched to safer single-variant probes (`runs=1`, `tokens=128`), one variant
  per process:
  - `benchmarks/repro_capsules/qwen3_fp32_probe_t128_r1_summary.json`
  - `benchmarks/repro_capsules/qwen3_fp32_nofma_probe_t128_r1_summary.json`
  - `benchmarks/repro_capsules/glm47_fp32_probe_t128_r1_summary.json`
  - `benchmarks/repro_capsules/glm47_fp32_nofma_probe_t128_r1_summary.json`

Probe results:
- Qwen:
  - `qwen_combine_fp32`: fidelity `FAIL` (`93/128`), decode speedup `0.36x`,
    memory up (`17.24 -> 17.71 GB`)
  - `qwen_combine_fp32_no_fma`: fidelity `FAIL` (`93/128`), decode speedup
    `0.3614x`, memory up (`17.24 -> 17.71 GB`)
- GLM:
  - `glm_combine_fp32`: fidelity `FAIL` (`2/128`), decode speedup `0.9892x`
  - `glm_combine_fp32_no_fma`: fidelity `PASS` (`128/128`), decode speedup
    `1.0161x`

Decision:
- Keep Qwen off fp32 combine modes (`fp32`, `fp32_no_fma`) due simultaneous
  fidelity and performance failure in probe.
- Keep GLM on `fp32_no_fma` path as the only fp32-family mode that remains
  fidelity-safe in probe.
- Continue using fidelity-first gating: microbench improvements alone are not
  promotion criteria.

## EXP-20260210T221022Z-REPRO-AUTOMATION-ENTRYPOINTS

Question:
- Can we make the foundry campaign + transfer-probe workflow fully reproducible
  from single entrypoint scripts, with quick smoke validation?

Code delta:
- `benchmarks/run_moe_combine_foundry_campaign.py`
  - Hardened report rendering when stats are missing (`n/a` fallback).
  - Added template validation against discovered `MoECombineOp().templates()`.
  - Added CLI guard requiring at least two unique template IDs.
- `benchmarks/run_moe_combine_transfer_probes.py` (new)
  - Runs isolated Qwen/GLM fp32-family combine probes via
    `benchmarks/bench_iso_variant_sweep.py`.
  - Emits a rollup summary with per-variant fidelity/speed/memory deltas.
  - Supports `--dry-run`, `--skip-qwen`, `--skip-glm`, configurable
    `--runs/--max-tokens/--ledger`.

Validation artifacts:
- Foundry script mock smoke:
  - command:
    - `python benchmarks/run_moe_combine_foundry_campaign.py --backend mock --seeds 20260210 --cases-per-seed 1 --correctness-tests 1 --warmup 1 --repeats 2 --out-dir sessions/foundry_moe_combine_campaign_smoke_scriptcheck`
  - outputs:
    - `sessions/foundry_moe_combine_campaign_smoke_scriptcheck/combined/attempts.ndjson`
    - `sessions/foundry_moe_combine_campaign_smoke_scriptcheck/combined/summary.json`
    - `sessions/foundry_moe_combine_campaign_smoke_scriptcheck/combined/report.md`
  - smoke winner counts: `{'t0_basic': 0, 't1_k8_unrolled': 0, 't2_row_tile': 1}`
- Transfer script dry-run:
  - command:
    - `python benchmarks/run_moe_combine_transfer_probes.py --dry-run --runs 1 --max-tokens 128 --prefix moe_combine_transfer_probe_dryrun_20260210`
  - output:
    - `benchmarks/repro_capsules/moe_combine_transfer_probe_dryrun_20260210_summary.json`

Decision:
- Adopt both scripts as canonical reproducibility entrypoints for this phase.
- Use real (non-dry) transfer probe runs as the next execution step before any
  further combine-mode promotion changes.

## EXP-20260210T221834Z-TRANSFER-PROBE-RERUN-VIA-ENTRYPOINT

Question:
- When executed through the new one-command transfer entrypoint, do fp32-family
  combine outcomes replicate prior conclusions, and is there any new signal?

Method:
- Command:
  - `python benchmarks/run_moe_combine_transfer_probes.py --runs 1 --max-tokens 128 --prefix moe_combine_transfer_probe_t128_r1_20260210`
- Notes:
  - Initial rollup printed `0/0` fidelity due key mismatch in parser
    (`matched_tokens/total_tokens` vs `matched/total`).
  - Parser fixed in `benchmarks/run_moe_combine_transfer_probes.py`, then rollup
    was regenerated from child summaries for this exact run.

Artifacts:
- Rollup:
  - `benchmarks/repro_capsules/moe_combine_transfer_probe_t128_r1_20260210_summary.json`
- Child summaries:
  - `benchmarks/repro_capsules/moe_combine_transfer_probe_t128_r1_20260210_qwen3_summary.json`
  - `benchmarks/repro_capsules/moe_combine_transfer_probe_t128_r1_20260210_glm47_summary.json`
- Per-variant capsules:
  - `benchmarks/repro_capsules/moe_combine_transfer_probe_t128_r1_20260210_qwen3_qwen_combine_fp32.json`
  - `benchmarks/repro_capsules/moe_combine_transfer_probe_t128_r1_20260210_qwen3_qwen_combine_fp32_no_fma.json`
  - `benchmarks/repro_capsules/moe_combine_transfer_probe_t128_r1_20260210_glm47_glm_combine_fp32.json`
  - `benchmarks/repro_capsules/moe_combine_transfer_probe_t128_r1_20260210_glm47_glm_combine_fp32_no_fma.json`

Results:
- Qwen:
  - `qwen_combine_fp32`: fidelity `FAIL` (`93/128`), decode speedup `0.3585x`,
    memory delta `+0.47 GB`
  - `qwen_combine_fp32_no_fma`: fidelity `FAIL` (`9/128`), decode speedup
    `0.3586x`, memory delta `+0.47 GB`
- GLM:
  - `glm_combine_fp32`: fidelity `FAIL` (`2/128`), decode speedup `1.0117x`,
    memory delta `+0.00 GB`
  - `glm_combine_fp32_no_fma`: fidelity `PASS` (`128/128`), decode speedup
    `1.0531x`, memory delta `+0.00 GB`

Interpretation:
- Qualitative ranking is unchanged vs earlier probes:
  - Qwen fp32-family combine modes remain non-viable (fidelity failure +
    severe decode regression).
  - GLM keeps the same split: `fp32` fails fidelity, `fp32_no_fma` passes and
    remains positive in decode.
- The `qwen_combine_fp32_no_fma` mismatch count changed materially (`9/128` vs
  prior `93/128`), but verdict stays `FAIL`; this suggests unstable failure
  manifestation under an already-invalid mode, not a promotion candidate.

Decision:
- No promotion changes from this rerun.
- Keep Qwen fp32-family modes rejected.
- Keep GLM `fp32_no_fma` as the only fidelity-safe fp32-family combine mode.

Next hypothesis:
- Move to long-context confirmation for the same pass/fail boundaries:
  - GLM `fp32_no_fma` at `max_tokens=1024` (`runs=2`) against control in isolated
    subprocesses.
  - Qwen fp32-family modes can be deprioritized unless there is a new kernel
    design change.

## EXP-20260210T222429Z-GLM-FP32-NOFMA-LONGCONFIRM

Question:
- Does `glm_combine_fp32_no_fma` remain fidelity-safe and beneficial at long
  context (`1024`) in an isolated rerun against control?

Method:
- Command:
  - `python benchmarks/bench_iso_variant_sweep.py --suite glm47 --variants control_swiglu_moe glm_combine_fp32_no_fma --runs 2 --max-tokens 1024 --prefix glm47_fp32_no_fma_longconfirm_t1024_r2`

Artifacts:
- `benchmarks/repro_capsules/glm47_fp32_no_fma_longconfirm_t1024_r2_summary.json`
- `benchmarks/repro_capsules/glm47_fp32_no_fma_longconfirm_t1024_r2_control_swiglu_moe.json`
- `benchmarks/repro_capsules/glm47_fp32_no_fma_longconfirm_t1024_r2_glm_combine_fp32_no_fma.json`

Results:
- `control_swiglu_moe`:
  - fidelity `PASS` (`1024/1024`)
  - decode speedup vs baseline: `1.0442x`
  - prefill change: `-0.21%`
  - memory: `16.94 -> 16.95 GB`
- `glm_combine_fp32_no_fma`:
  - fidelity `PASS` (`1024/1024`)
  - decode speedup vs baseline: `1.0575x`
  - prefill change: `+2.62%`
  - memory: `16.94 -> 16.95 GB`

Interpretation:
- Both variants preserve greedy token fidelity at long context in this run.
- `glm_combine_fp32_no_fma` remains positive and shows an incremental decode
  gain over control in the same campaign (patched decode `82.7` vs `80.3` tok/s,
  ~`+3.0%`).
- This result supports keeping `glm_combine_fp32_no_fma` as the best current
  long-context-safe fp32-family combine mode.

Decision:
- Keep `glm_combine_fp32_no_fma` as long-context-safe and still favored over
  control in this confirmation run.
- No Qwen fp32-family re-entry; those modes remain rejected.

Next hypothesis:
- Test whether foundry-discovered `t2_row_tile` behavior can transfer into a
  production combine path (env-gated) and outperform `glm_combine_fp32_no_fma`
  without fidelity regressions.

## Benchmark-vs-Baseline Snapshot (2026-02-10)

Question:
- Where are current custom-kernel variants truly better than baseline on decode and prefill, with fidelity safety as a hard gate?

Protocol:
- Isolated sweeps via `benchmarks/bench_iso_variant_sweep.py`.
- Suites: `qwen3`, `glm47`.
- Token lengths: `200` and `1024`.
- Replicates: `repA`, `repB`.
- Runs per variant: `3`.
- Prefill sign convention in this section: positive means patched prefill is slower.

Artifacts:
- Aggregate snapshot: `benchmarks/repro_capsules/benchmark_vs_baseline_snapshot_20260210.json`.
- Source summaries (8):
  - `benchmarks/repro_capsules/glm47_benchmark_vs_baseline_t1024_r3_repA_20260210_summary.json`
  - `benchmarks/repro_capsules/glm47_benchmark_vs_baseline_t1024_r3_repB_20260210_summary.json`
  - `benchmarks/repro_capsules/glm47_benchmark_vs_baseline_t200_r3_repA_20260210_summary.json`
  - `benchmarks/repro_capsules/glm47_benchmark_vs_baseline_t200_r3_repB_20260210_summary.json`
  - `benchmarks/repro_capsules/qwen3_benchmark_vs_baseline_t1024_r3_repA_20260210_summary.json`
  - `benchmarks/repro_capsules/qwen3_benchmark_vs_baseline_t1024_r3_repB_20260210_summary.json`
  - `benchmarks/repro_capsules/qwen3_benchmark_vs_baseline_t200_r3_repA_20260210_summary.json`
  - `benchmarks/repro_capsules/qwen3_benchmark_vs_baseline_t200_r3_repB_20260210_summary.json`

Environment notes:
- Short cooldown intervals were inserted between major sweep blocks to reduce thermal carryover.
- First GLM sweep incurred one-time model download/warm cache effects for mlx-community/GLM-4.7-Flash-4bit-mxfp4.

Per-row comparative view:
- Full row-level report is in `rows` within `benchmarks/repro_capsules/benchmark_vs_baseline_snapshot_20260210.json` (`40` rows total, each with fidelity/decode/prefill/memory deltas and capsule path).

Per-variant aggregate (across `t=200` + `t=1024`, both replicates):

| Model | Variant | Decode speedup mean / median | Decode delta vs baseline (pp, mean) | Prefill change (slower+, mean / median) | Prefill delta vs baseline (pp, mean) | Stability | Verdict |
|---|---|---:|---:|---:|---:|---|---|
| Qwen3-30B-A3B-4bit | `qwen_combine_exact` | `0.9859x / 0.9864x` | `-0.41` | `+0.28% / +1.09%` | `+1.26` | `high_variance` | `reject` |
| Qwen3-30B-A3B-4bit | `qwen_router_argpartition_logits` | `0.9867x / 0.9866x` | `-0.33` | `-0.78% / -0.97%` | `+0.20` | `high_variance` | `reject` |
| Qwen3-30B-A3B-4bit | `qwen_router_argpartition_logits_topk_combine_exact` | `1.0001x / 0.9991x` | `+1.01` | `+0.61% / +0.84%` | `+1.59` | `high_variance` | `hold` |
| GLM-4.7-Flash-4bit-mxfp4 | `glm_combine_fp32_no_fma` | `0.9555x / 0.9551x` | `+0.76` | `+2.08% / +1.08%` | `+0.31` | `high_variance` | `hold` |
| GLM-4.7-Flash-4bit-mxfp4 | `shared_experts_overlap_streams2` | `0.7003x / 0.7026x` | `-24.76` | `+11.53% / +11.50%` | `+9.76` | `high_variance` | `reject` |
| GLM-4.7-Flash-4bit-mxfp4 | `glm47_rope` | `0.9359x / 0.9354x` | `-1.19` | `+0.47% / +0.71%` | `-1.29` | `high_variance` | `reject` |
| GLM-4.7-Flash-4bit-mxfp4 | `glm_combine_fp32_no_fma_shared_overlap_streams2` | `0.7072x / 0.7086x` | `-24.07` | `+10.87% / +11.07%` | `+9.11` | `high_variance` | `reject` |
| GLM-4.7-Flash-4bit-mxfp4 | `glm_combine_fp32_no_fma_glm47_rope` | `0.9229x / 0.9254x` | `-2.50` | `+1.60% / +1.64%` | `-0.17` | `high_variance` | `reject` |

Top candidates from this snapshot:
- Qwen3-30B-A3B-4bit decode-first candidate: `qwen_router_argpartition_logits_topk_combine_exact` (mean decode delta `+1.01` pp; verdict `hold`).
- Qwen3-30B-A3B-4bit balanced candidate: `qwen_router_argpartition_logits` (mean decode delta `-0.33` pp, mean prefill delta `+0.20` pp; verdict `reject`).
- GLM-4.7-Flash-4bit-mxfp4 decode-first candidate: `glm_combine_fp32_no_fma` (mean decode delta `+0.76` pp; verdict `hold`).
- GLM-4.7-Flash-4bit-mxfp4 balanced candidate: `glm_combine_fp32_no_fma` (mean decode delta `+0.76` pp, mean prefill delta `+0.31` pp; verdict `hold`).

Promotion decision:
- No candidate satisfies full promote policy (fidelity pass + consistent decode win + non-material prefill impact + stable replicates + stable memory).
- Promotion outcome for this snapshot: `hold` overall.

Default recommendation (current state):
- Qwen: keep baseline control behavior (`control_patterns_moe_mlp`) as benchmark truth anchor; do not promote a new custom-kernel default from this snapshot.
- GLM: keep `fp32_no_fma` as current patch-path default, but retain `hold` status for benchmark-vs-control truth until consistency improves in additional controlled replicates.

Next targeted experiments:
- 1. GLM `glm_combine_fp32_no_fma` consistency confirmation: run `runs=5`, fixed ABBA order per replicate block, and explicit cooldown windows to resolve the `t=200` sign flip vs control.
- 2. Qwen router/combine isolation matrix: run single-toggle sweeps (`qwen_combine_exact`, `qwen_router_argpartition_logits`, and topk+combine) with `runs=5` at both lengths and pinned process ordering to separate thermal/order effects from true kernel signal.
## Benchmark-vs-Baseline Follow-up (2026-02-11)

Question:
- Do higher-run AB/BA replication sweeps convert any candidates into clear benchmark-vs-baseline wins?

Protocol:
- Isolated sweeps with `benchmarks/bench_iso_variant_sweep.py`.
- `runs=5`, token lengths `200` and `1024`.
- Replicate order control for GLM (`AB` and `BA`) and Qwen (ordered repA + reversed repB).
- Prefill sign convention here: positive means patched prefill is slower.

Artifacts:
- `benchmarks/repro_capsules/glm47_consistency_abba_t1024_r5_repA_20260211_summary.json`
- `benchmarks/repro_capsules/glm47_consistency_abba_t1024_r5_repB_20260211_summary.json`
- `benchmarks/repro_capsules/glm47_consistency_abba_t200_r5_repA_20260211_summary.json`
- `benchmarks/repro_capsules/glm47_consistency_abba_t200_r5_repB_20260211_summary.json`
- `benchmarks/repro_capsules/qwen3_isolation_ordered_t1024_r5_repA_20260211_summary.json`
- `benchmarks/repro_capsules/qwen3_isolation_ordered_t1024_r5_repB_20260211_summary.json`
- `benchmarks/repro_capsules/qwen3_isolation_ordered_t200_r5_repA_20260211_summary.json`
- `benchmarks/repro_capsules/qwen3_isolation_ordered_t200_r5_repB_20260211_summary.json`
- Aggregate follow-up snapshot: `benchmarks/repro_capsules/benchmark_vs_baseline_followup_20260211.json`
- Long-context confirmation: `benchmarks/repro_capsules/glm47_final_longconfirm_t1024_r5_20260211_summary.json`

Aggregate candidate deltas vs baseline variant:

| Model | Variant | Decode delta vs baseline (pp mean / min..max) | Prefill delta vs baseline (pp mean / min..max) | Fidelity | Verdict |
|---|---|---:|---:|---|---|
| GLM-4.7-Flash-4bit-mxfp4 | `glm_combine_fp32_no_fma` | `+2.31 / +0.31..+6.73` | `-0.88 / -2.94..+1.85` | `PASS` | `promote_candidate` |
| Qwen3-30B-A3B-4bit | `qwen_combine_exact` | `-2.40 / -6.00..-0.54` | `+0.89 / -0.95..+2.55` | `PASS` | `reject` |
| Qwen3-30B-A3B-4bit | `qwen_router_argpartition_logits` | `-0.21 / -0.65..+0.66` | `-0.68 / -2.05..+0.30` | `PASS` | `reject` |
| Qwen3-30B-A3B-4bit | `qwen_router_argpartition_logits_topk_combine_exact` | `-1.66 / -5.71..+0.53` | `+1.85 / +0.37..+5.70` | `PASS` | `reject` |

Interpretation:
- GLM `glm_combine_fp32_no_fma` improved materially vs control in this follow-up (decode delta positive in all 4 blocks), with mixed prefill deltas but net prefill mean improvement.
- Qwen candidates remained non-promotable:
  - `qwen_combine_exact` is consistently decode-negative.
  - `qwen_router_argpartition_logits` is near-neutral/negative decode with mixed prefill.
  - `qwen_router_argpartition_logits_topk_combine_exact` remains volatile and prefill-worse on average.

Decision update:
- GLM long-context confirmation (`runs=5`, `max_tokens=1024`) remained fidelity-safe and decode-positive (`+0.93 pp` vs control), but with prefill regression in that block (`+2.35 pp` vs control).
- GLM: keep `glm_combine_fp32_no_fma` as active default, with status `promote_candidate` (not fully promoted) pending tighter prefill variance.
- Qwen: no promotion; keep baseline control behavior as the benchmark truth anchor.
