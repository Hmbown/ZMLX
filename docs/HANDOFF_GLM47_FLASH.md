# GLM-4.7-Flash experiments handoff (branch `glm4.7flash`)

This document is a handoff for continuing GLM-4.7-Flash optimization work, plus
an idea backlog for DeepSeek-V3.2 / Kimi-K2.5 MoE fusions.

## What’s already implemented on `glm4.7flash`

### 1) Decode-only RoPE+concat fusion (opt-in, currently not a win)

- Kernel utilities: `src/zmlx/kernels/rope.py`
  - `_rope_cos_sin()` extracts **bit-identical** `(cos, sin)` tables from `mx.fast.rope`
    to avoid tiny float32 drift (important for GLM’s float32 attention).
  - `rope_concat_qk_decode_pos()` builds:
    - `queries = concat([q_nope, rope(q_pe)])`
    - `keys_step = concat([kv_latent, rope(k_pe)])`
    in a single Metal launch for decode (`T=1`).
- Patch pattern: `src/zmlx/patch/patterns/glm47_rope.py` (pattern name: `glm47_rope`)
  - Targets `mlx_lm.models.glm4_moe_lite.Glm4MoeLiteAttention`.
  - Decode-only (`L==1`) and falls back to upstream for prefill.
  - Uses power-of-two growth for RoPE table length to avoid materializing GLM’s full
    `max_position_embeddings` table eagerly.
- Test: `tests/test_glm47_rope_kernel.py` checks exact equality vs `mx.fast.rope`.

**Observed result (M4 Max 36GB, 3 runs × 200 tokens):**
- Baseline → `swiglu_mlp+moe_mlp` control: **1.067x** decode
- Baseline → `swiglu_mlp+moe_mlp+glm47_rope`: **1.053x** decode

Capsules:
- `benchmarks/repro_capsules/glm47_flash_control_m4max_20260205.json`
- `benchmarks/repro_capsules/glm47_flash_rope_m4max_20260205.json`

Conclusion: the RoPE fusion is **fidelity-safe** but **slightly worse than the
current best patch set**. Treat it as an experiment until it’s faster.

### 2) “Complex” idea: overlap `shared_experts(x)` with routed experts (regresses badly)

- Implementation: `src/zmlx/patch/patterns/moe_mlp.py`
- Gated behind env vars (OFF by default):
  - `ZMLX_MOE_STREAMS=2` (or more)
  - `ZMLX_MOE_SHARED_EXPERTS_OVERLAP=1`

**Observed result (M4 Max 36GB, 3 runs × 200 tokens):**
- `ZMLX_MOE_STREAMS=2 ZMLX_MOE_SHARED_EXPERTS_OVERLAP=1` → **0.705x** decode

Capsule:
- `benchmarks/repro_capsules/glm47_flash_shared_overlap_m4max_20260205.json`

Conclusion: keep this disabled; if revisiting, measure with Metal capture and
check for implicit syncs / resource contention (it likely just adds overhead).

### 3) “Matrix” logging for attempts + notes

- Module: `src/zmlx/matrix/`
- Ledger (JSONL): `benchmarks/matrix.jsonl`
- Notes are stored per entry (`notes` field) so you can record attempts/results.

Useful commands:

```bash
source .venv/bin/activate

# Append a new entry (writes to benchmarks/matrix.jsonl by default):
python -m zmlx.matrix run mlx-community/GLM-4.7-Flash-4bit \
  --patterns swiglu_mlp moe_mlp \
  --runs 3 --max-tokens 200 \
  --notes "control run"

# View history for GLM:
python -m zmlx.matrix history glm-4.7-flash --ledger benchmarks/matrix.jsonl
```

## How to reproduce the GLM experiments

Benchmark harness:
- `benchmarks/bench_glm47_flash_experiments.py`

Run variants one-at-a-time (running all sequentially can hit Metal OOM on some setups):

```bash
source .venv/bin/activate

# Control (current best):
python benchmarks/bench_glm47_flash_experiments.py \
  --variants control_swiglu_moe \
  --runs 3 --max-tokens 200 \
  --json-out benchmarks/repro_capsules/glm47_flash_control_<device>_<date>.json

# RoPE experiment:
python benchmarks/bench_glm47_flash_experiments.py \
  --variants glm47_rope \
  --runs 3 --max-tokens 200 \
  --json-out benchmarks/repro_capsules/glm47_flash_rope_<device>_<date>.json
```

Print a capsule report:

```bash
source .venv/bin/activate
python -m zmlx.bench.report benchmarks/repro_capsules/<capsule>.json
```

## GLM next steps (highest ROI)

1) **Understand why `glm47_rope` is slower**
   - Microbench (local) showed `rope_concat_qk_decode_pos` is ~1% slower than
     the baseline `mx.fast.rope + concat` chain for GLM decode shapes.
   - Likely reasons:
     - `mx.fast.rope` is already very optimized (hard to beat).
     - `metal_kernel(..., ensure_row_contiguous=True)` may be copying inputs.
   - Concrete follow-ups:
     - Inspect contiguity/copies around `q_pe` / `k_pe` (post-transpose/split).
     - Try a kernel variant that uses the full `(S, D/2)` table + scalar offset
       (avoid slicing `cos[pos]`, `sin[pos]` each step).
     - Consider fusing more than RoPE+concat (low-rank attention path may be
       better ROI: `q_a_proj → q_a_layernorm → q_b_proj`).

2) **Down-proj + combine fusion for GLM SwitchGLU**
   - `_try_fused_downproj_combine()` exists in `src/zmlx/patch/patterns/moe_mlp.py`
     but currently only fires for some families (not GLM’s SwitchGLU semantics).
   - Extending it for GLM may save an extra dispatch per MoE layer.
   - New experiment on this branch:
     - Env flag: `ZMLX_GLM_FUSED_DOWNPROJ_COMBINE=1`
     - Implementation: `_fused_switch_mlp_downproj_combine(...)` in
       `src/zmlx/patch/patterns/moe_mlp.py` (decode-only, no-FMA accumulation)
     - **Status:** currently marked **unsafe** — running it can crash with Metal
       OOM (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) due to the extra
       per-expert matmul loop. The bench harness skips it unless you pass
       `--allow-unsafe`.
     - Capsule (records the skip): `benchmarks/repro_capsules/glm47_flash_downproj_combine_m4max_20260205.json`

3) **KV-cache quantization performance**
   - The compatibility path for GLM quantized KV can run, but may regress for
     short generation lengths (overhead dominates). Re-benchmark at longer
     lengths (>= 1k tokens) and consider starting quantization after N steps.

## DeepSeek-V3.2 / Kimi-K2.5: MoE fusion idea backlog (token-identical goal)

Below is a blueprint (from a 2026-02-02 document) for adding DeepSeekMoE/Kimi
fusions. The key is preserving DeepSeek routing **exactly**:

### A) DeepSeek router fusion kernel

DeepSeek routing semantics to preserve:
- Affinity per expert: `s = sigmoid(logit)` (**bias is NOT applied here**)
- Top-K selection: choose by `rank = s + bias`
- Gating weights: `w = s_selected / sum(s_selected)`
- Deterministic tie-break: prefer lower expert index when ranks tie

Proposed kernel signature:
- Inputs:
  - `router_logits`: `(T, Nr)` (Nr=256 for DeepSeek-V3, Nr=384 for Kimi-K2)
  - `bias`: `(Nr,)` (or zeros)
- Outputs:
  - `topk_idx`: `(T, K)` (K=8)
  - `topk_w`: `(T, K)` (float32 preferred for fidelity)

Where to integrate in ZMLX:
- Add a new gating helper in `src/zmlx/kernels/moe.py` (or a submodule) alongside
  `topk_gating_softmax`.
- Extend `src/zmlx/patch/patterns/moe_mlp.py` `_gating(...)` to detect DeepSeek/Kimi
  router modules and route them to the sigmoid+topk+renorm fused kernel.

Validation checklist:
- Unit test: fused router output matches MLX reference chain (idx + weights).
- Model test: `python -m zmlx.validate <model> --max-tokens 200 --runs 3` must be
  token-identical.

### B) Combine fusion kernel: `shared + Σ(w_k * expert_out_k)` (+ optional residual)

DeepSeek/Kimi MoE typically has:
- K routed experts per token (K=8)
- plus a shared expert path (dense) added to the MoE output

Proposed kernel signature:
- Inputs:
  - `expert_out`: `(T, K, D)`
  - `w`: `(T, K)` (float32)
  - `shared_out`: `(T, D)` (optional)
  - `residual`: `(T, D)` (optional)
- Output:
  - `out`: `(T, D)`

Where to integrate:
- If the model materializes `(T, K, D)` expert outputs, a fused combine saves
  multiple small ops (broadcast/mul/reduce/add).
- For bigger wins, consider a phase-2 primitive that fuses expert down-proj and
  **streams** the weighted accumulation (avoid materializing `(T, K, D)` at all).

### C) Practical blockers (as of this branch)

- **DeepSeek-V3.2**: can likely load via `mlx_lm.models.deepseek_v32`, but full
  validation requires very high unified memory hardware (the code can be added
  and the matrix can record “SKIP(RAM)” on smaller machines).
- **Kimi-K2.5**: may be blocked on an `mlx-lm` model class/config mapping. Once
  it loads through `mlx_lm.load()`, ZMLX-side MoE work should apply.
