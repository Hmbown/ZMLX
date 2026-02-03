# ZMLX Roadmap

> **Updated:** February 3, 2026  
> **Status:** Moonshot feasibility captured; focus on shipping P0/P1 while prototyping experimental tracks.  
> **Current Version:** 0.7.13

This roadmap prioritizes features based on technical impact, user demand, and strategic alignment with MLX ecosystem growth.

---

## Priority Matrix

| Priority | Feature | Impact | Effort | Owner |
|:--------:|:--------|:------:|:------:|:------|
| **P0** | Per-Device Autotune Profiles | High | Medium | TBD |
| **P0** | Cross-Backend Correctness Harness | High | Medium | TBD |
| **P1** | Auto-Fusion Pattern Discovery | High | High | TBD |
| **P1** | Fused Dequantize + Compute | High | Medium | TBD |
| **P1** | TPU³ MVP (Anytime GEMV: int4 hi2/lo2, M=1 decode) | High | Medium | TBD |
| **P2** | Flash Attention (Tiled) | Medium | High | TBD |
| **P2** | CPU/GPU Stream Scheduling | Medium | Medium | TBD |
| **P3** | Paged KV Cache | Medium | High | TBD |
| **P3** | Device Scheduling Profiler | Low | Medium | TBD |
| **P3** | Transformer Block Fusion (MLX C++ deps) | High | Very High | TBD |
| **P3** | Model-Specific JIT Kernel Cache | Medium | High | TBD |

---

## Moonshot Alignment & Feasibility (Demis Hassabis Analysis)

This section captures which moonshot ideas can be folded into the current roadmap, what is feasible in ZMLX, and what requires upstream MLX C++ work or Apple hardware support.

### Incorporate Now (Fits ZMLX Scope)

1. **Fused dequant + compute** (already P1)
   - **Feasibility:** High. Fits custom kernel library and patch system.
   - **Next step:** Extend P1 task to include `dequant_rmsnorm` and `dequant_rope` after dequant+activation.
2. **Paged KV cache MVP** (already P3)
   - **Feasibility:** Medium. Pure Python + Metal kernels; UMA reduces complexity.
   - **Next step:** Add a constrained MVP (block size 256, max context 4096) and integrate with `paged_attention`.
3. **Thermal-aware autotuning**
   - **Feasibility:** Medium. Requires runtime telemetry and profile adaptation.
   - **Next step:** Add throttling detection and downshift threadgroup sizes when sustained clocks drop.

### Medium-Term (Needs MLX Coordination or Extra Infra)

4. **Graph-level fusion across layers**
   - **Feasibility:** Medium-low in ZMLX alone. Requires MLX graph/IR hooks or a C++ extension.
   - **Next step:** Prototype a "block fusion" path behind an experimental flag, document MLX prerequisites.
5. **Model-specific kernel cache (shape-specialized)**
   - **Feasibility:** Medium. Depends on a stable kernel cache key and shape tracing.
   - **Next step:** Extend autotune cache with `(arch, seq_len, batch)` keying and compile-once logic.
6. **Speculative decode kernel fusion**
   - **Feasibility:** Medium. Needs draft/verify APIs and sampling hooks.
   - **Next step:** Add a design doc with constraints, then prototype in experimental mode.

### Long-Term (Research / Hardware-Dependent)

7. **Neural Engine + GPU hybrid scheduling**
   - **Feasibility:** Low today. MLX lacks NE execution primitives.
   - **Next step:** Track Apple MLX NE roadmap; prepare interfaces but defer implementation.
8. **Ray-tracing core attention mapping**
   - **Feasibility:** Very low / research. Requires Metal ray-tracing pipeline integration.
   - **Next step:** Keep as research note only; do not schedule until a concrete path exists.

---

## TPU³: The "Anytime Inference" Technical Specification

TPU³ is not a hardware accelerator, but a **computational architecture** built on top of ZMLX. It treats transformer inference as a significance-weighted, error-bounded program. We are implementing this as a unified stack rather than a phased rollout.

### TPU³ MVP: Shippable Scaffolding + Working Kernel (Decode GEMV, M=1)

The fastest "real TPU³" we can ship on top of ZMLX starts with **decode** (M=1): a progressive int4 GEMV where **Pass 1** uses a 2-bit-ish approximation and **Pass 2** refines only selected output channels.

**Target primitive:** `y[N] = W[N,K] @ x[K]` with int4 weights packed as nibbles + per-group scales.

**Drop-in module layout (new):**

```
src/zmlx/tpu3/
  __init__.py
  pack.py              # CPU-side pack/unpack + per-group int4 quant (numpy)
  policy.py            # (eps, significance) -> refine mask
  kcache.py            # kernel cache keyed by (device_fingerprint, K, group, threadgroup)
  gemv_int4_anytime.py # Metal kernels (hi2 pass + masked lo2 refine)
tests/
  test_tpu3_pack.py
  test_tpu3_gemv_anytime.py
```

#### 1) Data Layout (MVP)

- `W4`: `uint32[N, K/8]` (8 int4 nibbles per word; two's complement in each nibble)
- `S`: per-group scales `float16[N, K/group]` (default `group=64`)

We **do not** store separate `W_hi2`/`W_lo2` buffers in the MVP; we extract them from nibbles in-kernel:

- `hi2 = nib & 0xC` (Pass 1)
- `lo2 = nib & 0x3` (Pass 2 residual)

#### 2) Core Kernel: Progressive int4 GEMV (hi2 + masked lo2)

**Pass 1 (always):**
- Computes `y0[n] = dot(x, sign_extend4(hi2)) * scale`
- Computes a conservative error bound `err_est[n]` using groupwise `sumabs(x)`:
  - `err_est += 3 * sumabs(x_group) * scale_group`

**Pass 2 (conditional):**
- Only for channels where `mask[n] == 1`, compute the residual dot using `lo2` and add into `y0[n]`.

Implementation detail: compile kernels once via `mx.fast.metal_kernel(...)` and invoke with explicit `grid/threadgroup/output_shapes/output_dtypes` (MLX custom Metal kernel workflow).

#### 3) Policy Layer: (epsilon, significance) -> mask

The MVP policy is intentionally simple and MoE-ready:

```python
# refine if the expected contribution of the error could exceed eps
refine = (err_est > eps) if significance is None else (significance * err_est > eps)
mask = refine.astype(mx.uint8)
```

Examples:
- MoE: `significance = p_e` (gate weight per expert contribution)
- Attention (FlashRefine): `significance = candidate_score` (only refine likely top-R keys)

#### 4) Kernel Cache (compile + dispatch stability)

Kernels are shape-specialized; cache them with a stable key:

`(device_fingerprint, K, group, threadgroup)`

Notes:
- `device_fingerprint` should match what we use for autotune profiles (e.g., chip family + GPU core count + OS/driver identifiers).
- Keep the MVP cache in-memory (`lru_cache`); add optional on-disk persistence later if compile latency matters in prod.

#### 5) Tests + Backend Gating (Linux CI-friendly)

- `tests/test_tpu3_pack.py`: pure numpy; runs everywhere (Linux/macOS).
- `tests/test_tpu3_gemv_anytime.py`: `@pytest.mark.metal`; skipped unless Metal is available.
- Linux CI target: run pure-Python tests under `mlx[cpu]` (or `mlx[cuda]` on GPU runners); Metal-marked tests remain skipped.

Acceptance criteria for MVP:
- `import zmlx.tpu3` is safe on CPU/CUDA (module must not hard-require Metal at import time).
- With `eps=0`, refinement runs everywhere and matches a dequantized int4 reference within expected int4 error bounds.
- With nonzero `eps`, refinement rate decreases (sanity check on mask effectiveness).

### Significance-Weighted MoE Superkernel

This kernel fuses routing, bit-budgeting, and expert compute into a single Metal dispatch.

```python
# Conceptual ZMLX API for TPU³ MoE
y = zmlx.tpu3.moe_layer(
    x, 
    gate_weights, 
    expert_weights_packed,
    tolerance=1e-3, 
    min_bits=2, 
    max_bits=4
)
```

**Inside the Superkernel:**
1.  **Route:** Top-K expert selection.
2.  **Budget:** For each expert $e$:
    *   $\text{Budget}_e = \text{softmax}(\text{gate}_e)$.
    *   If $\text{Budget}_e < \text{threshold}$, use **Pass 1 only** (2-bit).
    *   If $\text{Budget}_e \ge \text{threshold}$, enable **Pass 2 refinement** (4-bit).
3.  **Accumulate:** Fused sum of expert outputs directly into output buffer.

### Significance-Aware Attention (FlashRefine)

Instead of full precision QKᵀ, we use progressive refinement for attention scores:
1.  **Low-Bit QKᵀ:** Identify top-R keys per query using 2-bit weights.
2.  **Precise Refinement:** Re-compute only the top-R scores at full precision before softmax.
3.  **Result:** Sub-linear scaling in key-count for the precise portion of the attention matrix.

### The ZMLX TPU³ Engine (Policy + Compiler)

The engine acts as the orchestration layer:
*   **Kernel Generator:** Synthesizes Metal source specialized for (HiddenSize, ExpertCount, TileSize).
*   **Autotuner:** Finds the optimal `(Bq, Bk)` or `(BM, BN)` tiles for the specific M-series chip.
*   **Policy Manager:** Adjusts global `epsilon` (tolerance) dynamically based on thermal pressure or user-set latency budgets.

---
## TPU³ + Fused Quantized SwiGLU Test Matrix (Deep Validation)

This is the exhaustive validation plan for the TPU³ / progressive int4 path. Treat as a long-run checklist; prioritize later into a short list as needed.

### Status Snapshot (Feb 3, 2026)

**What has already been tried (do not repeat unless re‑validating):**
- [x] **Progressive TG eps sweep (TG=64) on LFM2.5‑1.2B and LFM2‑8B‑A1B** with eps in `0, 0.1, 1, 10`, `max_tokens=128`, `runs=5`. Best decode gain was on LFM2‑8B at eps=10 (~+8%), with small prompt regression (~‑1%). LFM2.5‑1.2B was mostly flat or small gains.
- [x] **Per‑group early‑exit (non‑TG)** on LFM2‑8B regressed heavily (‑35% to ‑38% decode). Not viable in current form.
- [x] **MoE Qwen3‑30B‑A3B benchmark** with local MLX (runs=5, max_tokens=256):  
  - gating+combine **‑4.5% decode** (prompt **‑1.6%**)  
  - fused SwiGLU **‑70% decode** (prompt **‑65%**)  
  JSON logged at `benchmarks/results/qwen3_30b_a3b_4bit_e2e_runs5.json` (MLX 0.30.4.dev). Earlier runs=3 snapshot at `benchmarks/results/qwen3_30b_a3b_e2e_local.json`.
- [x] **MLX pip venv check** still reports MLX 0.30.4 and `gather_qmm_swiglu` absent.
- [x] **Local MLX fork availability check**: `PYTHONPATH=mlx_local/python` exposes `mx.gather_qmm_swiglu` (MLX 0.30.4.dev). Pip MLX still lacks the op; MoE fused path should be re‑benchmarked under the local fork.
- [x] **TG skip‑refine reduction optimization** (avoid extra TG reductions when refinement is skipped). Microbench `benchmarks/bench_progressive_swiglu.py` with `B=1,K=256,N=512,bits=4,group=64,outer_runs=5` shows non‑TG eps=100 at **1.33x** vs baseline; TG eps=0 at **1.11x**, TG eps=10 slower. Logged in `benchmarks/results/slime_runs.json` (tag `tg_skip_refine_opt`, MLX 0.30.4).
- [x] **LFM TG=64 re‑run (eps=1,10, runs=5, max_tokens=128)** with TG skip‑refine (pre‑gating):  
  - LFM2.5‑1.2B: decode **‑0.5%** (eps=1) / **‑0.3%** (eps=10); prompt **‑1.3%** (eps=1) / **‑1.2%** (eps=10).  
  - LFM2‑8B‑A1B: decode **+4.8%** (eps=1) / **+8.0%** (eps=10); prompt **‑1.1%** (eps=1) / **‑0.1%** (eps=10).  
  Results in `benchmarks/results/lfm_baseline_results.json` (MLX 0.30.4, git SHA logged).
- [x] **Local MLX gather_qmm_swiglu microbench (outer_runs=3)**: decode‑like M=1 speedups **1.03x–1.27x**, prefill M=4 **1.23x**, but M=64 regresses to **0.46x**. Logged in `benchmarks/results/slime_runs.json` (tag `gather_qmm_local`, MLX 0.30.4.dev).
- [x] **SwiGLU TG allowlist + family gating**: TG progressive now requires tokens=1, eps>=10, KxN in `ZMLX_FUSED_QSWIGLU_TG_ALLOWLIST` (default `2048x7168`), and module family not in `ZMLX_FUSED_QSWIGLU_TG_DENY_FAMILY` (default `lfm`).
- [x] **MoE fused SwiGLU token gate tightened**: default max tokens for `gather_qmm_swiglu` path reduced to **1** (override via `moe_fused_swiglu_max_tokens` in `zmlx.patch.patch` or `bench_moe_e2e.py --fused-max-tokens`).
- [x] **Qwen perf denylist**: `moe_mlp` disabled by default for Qwen family due to recent decode regressions; override with `ZMLX_PATCH_ALLOW_PERF_RISK=1`.
- [x] **LFM TG=64 re‑run after allowlist gating (eps=10, runs=5)**:  
  - LFM2.5‑1.2B decode **+0.1%** (prompt +2.0%).  
  - LFM2‑8B‑A1B decode **+8.0%** (prompt ‑2.2%).  
  Results in `benchmarks/results/lfm_baseline_results.json` (MLX 0.30.4).
- [x] **Qwen3‑Coder‑Next 3‑bit MLX candidate found**: `NexVeridian/Qwen3-Coder-Next-3bit` (~34.9 GB). Not yet benchmarked; evaluate for decode speed and memory fit.

**Current knobs (documented to avoid rediscovery):**
- `ZMLX_FUSED_QSWIGLU_PROGRESSIVE=1` enables progressive decode path.
- `ZMLX_FUSED_QSWIGLU_EPS` controls refinement (eps=0 means always refine; higher eps skips more).
- `ZMLX_FUSED_QSWIGLU_TG` enables threadgroup kernel (best known TG so far: 64 for LFM2‑8B).
- `ZMLX_FUSED_QSWIGLU_PER_GROUP=1` enables per‑group refinement (currently regresses; avoid).
- `ZMLX_FUSED_QSWIGLU_MODE=auto|force|off` and `ZMLX_FUSED_QSWIGLU_MAX_TOKENS/MAX_OUT/MAX_IN` gate fusing by shape/token count.
- `ZMLX_FUSED_QSWIGLU_TG_MIN_EPS`, `ZMLX_FUSED_QSWIGLU_TG_ALLOWLIST`, and `ZMLX_FUSED_QSWIGLU_TG_DENY_FAMILY` gate TG progressive (default allowlist `2048x7168`, `eps>=10`, tokens=1; denylist defaults to `lfm`).
- `ZMLX_PATCH_ALLOW_PERF_RISK=1` re-enables perf‑risky patterns (e.g., `moe_mlp` on Qwen).

**Known model caveats:**
- Qwen models currently exclude `swiglu_mlp` due to fidelity issues in the patch set; override only with explicit evaluation.

### Run Templates (Copy/Paste)

Use `<REPO_ROOT>` for repo location; commands assume `PYTHONPATH=src`.

**Progressive TG eps sweep (decode focus):**
```
cd <REPO_ROOT>
PYTHONPATH=src python3 benchmarks/bench_lfm_baseline.py \
  --runs 5 --max-tokens 128 \
  --fused-qswiglu-mode auto \
  --fused-qswiglu-progressive \
  --fused-qswiglu-tg 64 \
  --fused-qswiglu-eps-sweep "0,0.1,1,10" \
  --fused-qswiglu-max-tokens 1 \
  --fused-qswiglu-max-out 16384 \
  --fused-qswiglu-max-in 8192
```

**Per‑group early‑exit (currently regresses; re‑test only after improvements):**
```
cd <REPO_ROOT>
PYTHONPATH=src python3 benchmarks/bench_lfm_baseline.py \
  --runs 3 --max-tokens 128 \
  --fused-qswiglu-mode auto \
  --fused-qswiglu-progressive \
  --fused-qswiglu-per-group \
  --fused-qswiglu-eps-sweep "0.1,10" \
  --fused-qswiglu-max-tokens 1 \
  --fused-qswiglu-max-out 16384 \
  --fused-qswiglu-max-in 8192
```

**MoE end‑to‑end (Qwen3‑30B‑A3B):**
```
cd <REPO_ROOT>
PYTHONPATH=src python3 benchmarks/bench_moe_e2e.py \
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --runs 3 --max-tokens 256 --fused-max-tokens 1
```

**Progressive microbench (kernel‑only):**
```
cd <REPO_ROOT>
PYTHONPATH=src python3 benchmarks/bench_progressive_swiglu.py
```

### Result Logging (Avoid Re‑runs)

Whenever you run a sweep, record these fields in a short note (or add to a future results table):
- model + bits + group_size
- eps + TG (or per‑group)
- prompt tok/s + decode tok/s (median)
- notes on regressions or wins
- MLX version + whether `mx.gather_qmm_swiglu` exists

### Interpretation Guidelines

- **Decode wins >3%** are meaningful; <2% is likely noise unless repeated.
- **Prompt regressions** are acceptable if decode gains are large (decode is the target).
- **Per‑group refinement** should be considered “failed” unless it beats TG=64.
- If **eps=0 and eps=10 behave similarly**, error‑bound checks are likely too conservative.

### Immediate Shortlist (If You Only Run 3 Things)

- [ ] LFM2‑8B‑A1B with TG=64 and eps in `1, 10` (confirm best decode gain).
- [ ] LFM2.5‑1.2B with TG=64 and eps in `0.1, 1` (find minimal overhead).
- [ ] Qwen3‑30B‑A3B MoE benchmark **only if** `mx.gather_qmm_swiglu` appears.

### Correctness (Must-Not-Break)
- [ ] Bit-level correctness vs `mx.quantized_matmul` for 4-bit and 8-bit (eps=0, TG on/off).
- [ ] Token-level parity on LFM2.5‑1.2B, LFM2‑8B‑A1B, Qwen3‑30B‑A3B (baseline vs patched).
- [ ] Error-bound tightness: compare estimated bounds vs actual error on random seeds.
- [ ] Long‑context stability: 1k+ tokens to ensure no drift or numeric blow‑ups.
- [ ] Regression guard: verify eps=0 always matches baseline within tolerance.

### Microbench Kernels (Shape Sweeps)
- [ ] Sweep K: 256 → 8192; N: 512 → 16384; B: 1 → 8.
- [ ] Sweep threadgroup sizes: 16 / 32 / 64 / 128.
- [ ] Sweep eps: 0 / 0.01 / 0.1 / 1 / 10 / 100.
- [ ] Bits: 4 vs 8; group_size: 32 / 64 / 128.
- [ ] Compare: baseline matmul vs fused vs progressive (no TG / TG / per-group).
- [ ] Track refine‑rate (%) vs eps to correlate with speedups.

### End‑to‑End Decode Benchmarks
- [ ] LFM2.5‑1.2B: max_tokens 128/256/512 with multiple prompts.
- [ ] LFM2‑8B‑A1B: max_tokens 128/256/512/1024 with multiple prompts.
- [ ] Qwen3‑30B‑A3B: MoE path with/without fused SwiGLU (requires MLX gather kernel).
- [ ] Additional MoE models (Mixtral/DeepSeek‑distill/Qwen variants).
- [ ] Cross‑prompt variance: at least 3 prompts per model.

### Prefill vs Decode Split
- [ ] Prefill-only (long prompt, short decode).
- [ ] Decode-only (short prompt, long decode).
- [ ] Mixed (realistic prompt + generation).
- [ ] TTFT sensitivity vs fused path (prefill impact).

### Kernel Variants (TPU³ Ideas)
- [ ] Progressive `sum_abs_x` precompute + group mask (reduce error‑bound cost).
- [ ] TG kernel that computes multiple output columns per group (tile reuse of x).
- [ ] Offline hi/lo weight packing to remove in‑kernel bit masking.
- [ ] Fast sigmoid approximation trade‑off (speed vs accuracy).
- [ ] Mixed‑precision accum (FP16 vs FP32) to test bandwidth/accuracy balance.

### Autotune & Heuristics
- [ ] Auto‑select TG size based on (K, N).
- [ ] Auto‑select eps based on per‑layer profiling.
- [ ] Gate progressive path only when it wins (avoid small‑model regressions).
- [ ] Add “fast path” denylist for known regressions (e.g., small K/N).

### Regression & Stability
- [ ] Cold‑start vs hot‑start compile overhead.
- [ ] Kernel cache growth vs performance (compile count + latency).
- [ ] Memory usage over time (no leaks) for long decode sessions.
- [ ] Multi‑batch decode behavior (B > 1).
- [ ] Power/thermal throttling impact on auto‑tuned kernels.

### Model Diversity & System Coverage
- [ ] Dense models: Llama/Mistral/Gemma/Phi.
- [ ] MoE models: Qwen‑A3B/DeepSeek‑distill/Mixtral.
- [ ] Small models (0.6B/1.7B) to quantify overhead vs benefit.
- [ ] Chip coverage: M1/M2/M3/M4, with notes on regressions.
- [ ] Custom MLX builds with `gather_qmm_swiglu` to enable full MoE path.

### System‑Level & Tooling
- [ ] GPU utilization + memory bandwidth tracking (if profiler access).
- [ ] Compare MLX builds (pip vs source) to unlock missing kernels.
- [ ] Record results under `benchmarks/results/` with metadata (eps, tg, bits, group_size).
- [ ] Tag runs with patch version + git SHA for reproducibility.

---
## SLIME Track (TPU⁰) — Full Vision + Execution Map

SLIME is the **runtime substrate** that generalizes TPU³ into a staged, policy‑driven system that prioritizes “uncertainty‑reduction per byte” under Apple UMA + Metal dispatch constraints. TPU³ is a special case (lockstep progressive bits). This section makes the full vision explicit and maps it to concrete files/tests so the next AI can continue without repeating work.

### Core Concepts (Operational)
- **Tile:** contiguous subregion sized for cache/register/threadgroup constraints.
- **Stage:** monotone refinement step (bits/dims/candidates/KV blocks).
- **Field:** runtime tensor metadata (confidence/importance/residency).
- **Potential:** “value per byte” scalar guiding work allocation.
- **Frontier:** active tile/stage set at runtime.
- **Plan:** compiled shape‑specialized kernels + legal schedules.
- **Policy:** runtime logic choosing stages/variants based on fields + budgets.

### Axioms / Invariants (What Must Hold)
- **Bytes‑first:** minimize bytes moved from UMA per unit certainty gained.
- **Launch scarcity:** minimize dispatches for M≈1 decode; prefer fused kernels.
- **Monotone refinement:** bounds can only shrink; stages only add information.
- **Locality is state:** cache/working‑set pressure is explicit in policy.
- **Potential flow:** diffusion‑like update over op/tile DAG drives scheduling.
- **Compiled‑vs‑runtime:** runtime chooses among precompiled variants only.
- **Kernel caching mandatory:** compile once; reuse via stable keys.

### SLIME → TPU³ Reduction (How It Emerges)
- If stages are only bitplanes and scheduling is lockstep, SLIME reduces to **progressive bits** (TPU³).
- If stages are single “full” and fusion is disabled, SLIME reduces to a **classic GPU pipeline**.
- If scheduling is fixed and tiling is static, SLIME reduces to a **systolic‑style** dataflow.

### Implementation Blueprint (Repo Map)

**Already prototyped (current codebase):**
- `src/zmlx/kernels/quant.py` — progressive int4 SwiGLU kernels (hi/lo refinement), TG variant, error bounds.
- `src/zmlx/patch/patterns/swiglu_mlp.py` — patch gating + env switches for progressive/TG/per‑group.
- `benchmarks/bench_lfm_baseline.py` — eps/TG sweep integration.
- `benchmarks/bench_progressive_swiglu.py` — kernel microbench.
- `benchmarks/bench_moe_e2e.py` — MoE end‑to‑end gating+combine benchmark (fused path awaits MLX gather kernel).
- `tests/test_quant_progressive_dequant.py`, `tests/test_quant_swiglu_progressive.py` — correctness tests.

**New SLIME layer (to be added):**
```
src/zmlx/slime/
  __init__.py
  config.py          # SlimeConfig, device fingerprint, defaults
  ir.py              # SlimeOp, StageSpec, TileTask
  plan.py            # compile_plan(), Plan object
  policy.py          # Policy interfaces + builtin policies
  fields.py          # GPU field allocation + update helpers
  ops/
    logits.py        # progressive lm_head / vocab projection
    moe.py           # staged MoE hooks (post‑MVP)
    attn.py          # FlashRefine skeleton (post‑MVP)
  kernels/
    qgemv_hi2.py      # Metal kernel builders
    qgemv_lo2.py
    confidence_mask.py
    flash_refine_attn.py
  autotune/
    search_spaces.py  # tile/threadgroup candidates
    runner.py         # uses zmlx.autotune utilities
  tests/
    test_bounds.py
    test_logits_progressive.py
    test_cache_keys.py
```

### File‑to‑Test Map (What to Run When You Touch It)

| Area | Primary Files | Tests / Benches |
|---|---|---|
| Progressive quantized SwiGLU | `src/zmlx/kernels/quant.py` | `tests/test_quant_swiglu_progressive.py`, `benchmarks/bench_progressive_swiglu.py` |
| Patch gating / env knobs | `src/zmlx/patch/patterns/swiglu_mlp.py` | `benchmarks/bench_lfm_baseline.py` |
| LFM end‑to‑end decode | `benchmarks/bench_lfm_baseline.py` | (script itself; write results to `benchmarks/results/`) |
| MoE end‑to‑end | `benchmarks/bench_moe_e2e.py` | (requires MLX gather kernel for fused path) |
| Progressive dequant utilities | `src/zmlx/kernels/quant.py` | `tests/test_quant_progressive_dequant.py` |
| SLIME core IR/policy | `src/zmlx/slime/ir.py`, `src/zmlx/slime/policy.py` | `tests/test_cache_keys.py`, `tests/test_bounds.py` |
| SLIME logits MVP | `src/zmlx/slime/ops/logits.py`, `src/zmlx/slime/kernels/qgemv_hi2.py` | `tests/test_logits_progressive.py`, `benchmarks/slime_compare.py` |

### SLIME MVP (Logits) — Concrete Deliverables

**Goal:** progressive lm_head with proof‑or‑fallback (token‑identical).
- **Packing:** split int4 planes + precompute per‑row low‑plane norms.
- **Stage 1:** hi2 kernel (estimate + radius).
- **Stage 2:** lo2 refinement only if proof fails.
- **Verification:** argmax proof using intervals; fallback to baseline if not proven.
- **Metrics:** fallback rate + decode tok/s + dispatch count proxy.

### SLIME Policy + Field System (Minimal Form)

**Fields:**
- `confidence` (interval radius or margin)
- `importance` (value of refining)
- `residency` (hotness / cache proxy)

**Policy (simple version):**
1. Run hi2 everywhere.
2. Compute bounds + mask.
3. Refine only if verification fails.

### Acceptance Metrics (SLIME MVP)
- **Correctness:** token‑identical greedy decode on at least one model for 1k tokens.
- **Performance:** ≥ +3% decode tok/s *or* ≥ 15% fewer bytes for lm_head step, no prefill regression.
- **Stability:** no kernel crashes, tests pass on at least two GPU archs.

### SLIME Moonshots (Longer Horizon)
- **Sparse logits refinement** via GPU compaction (avoid full fallback).
- **FlashRefine attention** with multi‑resolution KV cache and block bounds.
- **MoE effort allocation**: refine only top experts based on gate probabilities.
- **Persistent kernels / command‑buffer reuse** (requires MLX runtime hooks).
- **Texture‑backed weights** for cache/compression benefits (likely C++ work).

### SLIME “Infinite Time” Validation Matrix (Test Everything)

This is the maximal, exhaustive validation plan. Use it to avoid missing any failure mode or hidden regression. Every item should link to concrete files, tests, and benchmarks. Record results in `benchmarks/results/` with run metadata.

#### 1) Cost Model Calibration (Bytes, Launch, Occupancy)
- [ ] Measure launch overhead vs kernel duration for micro‑kernels (1–3 dispatches).
  - Files: `src/zmlx/slime/kernels/qgemv_hi2.py`, `src/zmlx/slime/kernels/qgemv_lo2.py`
  - New bench: `benchmarks/slime_launch_overhead.py`
- [ ] Validate UMA byte model against profiler counters (copy cost, cache hit proxy).
  - Files: `src/zmlx/slime/policy.py`, `src/zmlx/slime/config.py`
  - New bench: `benchmarks/slime_bytes_model.py`
- [ ] Evaluate effect of `ensure_row_contiguous` on copy overhead for all kernels.
  - Files: any kernel builder under `src/zmlx/slime/kernels/`
  - Tests: `tests/test_cache_keys.py` (new check for contiguity metadata)

#### 2) Kernel Correctness (Every Variant)
- [ ] Hi2 kernel bounds always contain the exact low‑bit result.
  - Files: `src/zmlx/slime/kernels/qgemv_hi2.py`
  - Tests: `tests/test_bounds.py`
- [ ] Hi2 + Lo2 reconstructs exact int4 dot for random seeds (all shapes).
  - Files: `src/zmlx/slime/kernels/qgemv_lo2.py`
  - Tests: `tests/test_logits_progressive.py`
- [ ] Progressive SwiGLU kernels remain exact at eps=0 across bits/group sizes.
  - Files: `src/zmlx/kernels/quant.py`
  - Tests: `tests/test_quant_swiglu_progressive.py`
- [ ] Error‑bound conservatism: bound ≥ actual error for all stages.
  - Files: `src/zmlx/slime/ops/logits.py`, `src/zmlx/slime/kernels/confidence_mask.py`
  - Tests: `tests/test_bounds.py`

#### 3) Microbench Sweep (All Shapes, All Knobs)
- [ ] Sweep K: 128 → 16384; N: 128 → 32768; B: 1 → 16.
  - Files: `benchmarks/bench_progressive_swiglu.py`
  - New bench: `benchmarks/slime_qgemv_sweep.py`
- [ ] Sweep threadgroup sizes: 8 / 16 / 32 / 64 / 128 / 256.
- [ ] Sweep eps: 0 / 0.01 / 0.1 / 1 / 10 / 100 / 1000.
- [ ] Sweep bits: 2 / 3 / 4 / 8 (if supported); group_size: 32 / 64 / 128 / 256.
- [ ] Measure refine‑rate (%) vs eps; correlate refine rate with speedup and accuracy.
- [ ] Validate cache/working‑set thresholds by increasing N until perf drops.

#### 4) Policy Evaluation (Runtime Scheduling)
- [ ] Baseline policy: lockstep stages (TPU³ emulation).
  - Files: `src/zmlx/slime/policy.py`
  - Tests: `tests/test_logits_progressive.py`
- [ ] Greedy frontier policy (priority = importance * radius / bytes).
  - Files: `src/zmlx/slime/plan.py`, `src/zmlx/slime/fields.py`
  - New bench: `benchmarks/slime_policy_compare.py`
- [ ] Diffusion‑based potential update (1–2 steps) vs no diffusion.
  - Files: `src/zmlx/slime/fields.py`
  - New bench: `benchmarks/slime_diffusion_ablation.py`
- [ ] Validate policy stability under thermal throttling (downshift TG).
  - Files: `src/zmlx/slime/policy.py`, `src/zmlx/slime/autotune/runner.py`

#### 5) End‑to‑End Decode (Token‑Level)
- [ ] LFM2.5‑1.2B: prompts×3, max_tokens 128/256/512.
  - Files: `benchmarks/bench_lfm_baseline.py`
- [ ] LFM2‑8B‑A1B: prompts×3, max_tokens 128/256/512/1024.
  - Files: `benchmarks/bench_lfm_baseline.py`
- [ ] Qwen3‑30B‑A3B: MoE path with fused SwiGLU (requires `mx.gather_qmm_swiglu`).
  - Files: `benchmarks/bench_moe_e2e.py`
- [ ] Mixed prefill/decode (long prompt, short decode; short prompt, long decode).
  - Files: `benchmarks/inference_benchmark.py`

#### 6) Prefill vs Decode Split (Detailed)
- [ ] Measure TTFT changes for SLIME vs baseline.
  - Files: `benchmarks/inference_benchmark.py`
- [ ] Verify no regression in pure prefill (very long prompts).
- [ ] Confirm decode‑only wins persist at batch sizes >1 (B=2,4,8).

#### 7) Cross‑Model Coverage
- [ ] Dense: Llama/Mistral/Gemma/Phi.
- [ ] MoE: Qwen‑A3B/DeepSeek‑distill/Mixtral.
- [ ] Small models (0.6B/1.7B) to ensure overhead doesn’t dominate.
- [ ] Mixed quantization families (4‑bit, 8‑bit, bf16 baselines).

#### 8) Cross‑Device Coverage
- [ ] M1/M2/M3/M4 (base/pro/max/ultra) with per‑chip notes.
- [ ] Kernel correctness A/B across device architectures (golden tests).
- [ ] Autotune profiles persist and re‑used across sessions.

#### 9) Runtime Robustness
- [ ] Long‑run decode (1000+ tokens) for leaks or slowdown.
- [ ] Cache pressure tests with multiple simultaneous models loaded.
- [ ] Kernel cache growth vs performance (compile count, memory use).
- [ ] Stress tests with random seeds for worst‑case numeric bounds.

#### 10) Tooling & Reporting
- [ ] Add a unified results schema: `benchmarks/results/slime_runs.json`.
  - Include: model, eps, TG, bits, group_size, refined_rate, MLX version, git SHA.
- [ ] Add a summarizer: `benchmarks/slime_report.py` → markdown report.
- [ ] Add a reproducibility stamp (commit + config) to every run.

#### 11) Optional Advanced Experiments
- [ ] Offline split hi/lo weight storage to avoid bit masking.
- [ ] Kernel that computes multiple output columns per TG (tile reuse).
- [ ] Fast sigmoid approximation kernel (bounded error).
- [ ] Mixed‑precision accumulation (FP16/FP32) to balance bandwidth.
- [ ] Persistent kernel experiments if MLX runtime hooks become available.

#### 12) Documentation + Guardrails
- [ ] Add a `docs/SLIME.md` with architecture + FAQs.
- [ ] Add a “when to enable SLIME” guide for end users.
- [ ] Create a denylist for known regressions (model or shape‑based).

---
## Apple M3/M4 (Apple9) Opportunity Map

This section captures M3/M4‑specific capabilities (Apple GPU family 9) and how to exploit them in ZMLX/SLIME without repeating research. It is a direct execution map: **capability → file → test**.

### Quick Facts (Do Not Re‑Research)
- M3 and M4 are **Apple GPU family 9 (Apple9)**; M1 is Apple7, M2 is Apple8.
- Apple9 guarantees **full 64‑bit atomics**, which enables new reduction paths.
- Apple9 supports **Metal 4** features that reduce command‑encoding overhead.
- Threadgroup limits remain **1024 threads / 32 KB shared memory**.

### Capability → ZMLX Integration Map

**Apple9 detection + feature flags**
- Files: `src/zmlx/device.py`, `src/zmlx/device_profile.py`
- Add `supports_apple9` flag and gate decode‑optimized kernels.
- Tests: `tests/test_device_profile.py` (add Apple9 branch)

**64‑bit atomic reductions**
- Files: `src/zmlx/kernels/reductions.py`, `src/zmlx/kernels/softmax.py`
- Add Apple9‑only 64‑bit atomic reduction path; keep fallback for Apple7/8.
- Tests: `tests/test_reductions.py` + microbench

**Metal 4 command‑encoding overhead reduction**
- Files: `src/zmlx/metal.py`, `src/zmlx/slime/*` (future)
- Use command allocators / decoupled command buffers / pipeline dataset serialization when available.
- Tests: add `benchmarks/slime_launch_overhead.py` (dispatch timing)

**Threadgroup memory pressure**
- Files: `src/zmlx/kernels/attention.py`, `src/zmlx/kernels/softmax.py`
- Add smaller TG variants (32–128) for M≈1 decode.
- Tests: `benchmarks/bench_progressive_swiglu.py` and attention microbench

**Decode‑focused kernel specialization**
- Files: `src/zmlx/kernels/linear.py`, `src/zmlx/kernels/transformer.py`
- Use function constants / specialization for M=1, small D, small N.
- Tests: `benchmarks/bench_lfm_baseline.py`, `benchmarks/inference_benchmark.py`

### Validation Plan (M3/M4‑Specific)
- [ ] **Dispatch overhead**: loop a minimal decode kernel 10k× and measure per‑dispatch CPU time.
  - New bench: `benchmarks/slime_launch_overhead.py`
- [ ] **Threadgroup sweep**: TG 32/64/128/256 on M3/M4 for M=1 kernels.
  - Use: `benchmarks/bench_progressive_swiglu.py`
- [ ] **64‑bit atomic reduction**: correctness + perf vs 32‑bit path.
  - New bench: `benchmarks/bench_atomic_reduce.py`
- [ ] **Cold vs warm compile timing**: quantify pipeline cache improvements.
  - New bench: `benchmarks/bench_compile_latency.py`
- [ ] **KV‑cache quantization**: A/B decode speed for memory‑bound models.
  - Use: `benchmarks/bench_lfm_baseline.py`

---
## P0: Critical Path

### 1. Per-Device Autotune Profiles

**Problem:** Current autotuner uses flat threadgroup candidates regardless of hardware. M1/M2/M3/M4 have different GPU core counts, memory bandwidth, and microarchitectural features.

**Goal:** Reduce search time and improve defaults when autotuning is disabled.

**Action Items:**
- [ ] Fix GPU core detection bug (currently returns CPU cores, not GPU)
- [ ] Create `DeviceTuningProfile` dataclass with per-chip defaults
- [ ] Build lookup table for all 16 chip variants (M1/M2/M3/M4 × base/Pro/Max/Ultra)
- [ ] Implement `@autotune()` decorator stub in `autotune.py`
- [ ] Upgrade autotune cache to v3 schema with device metadata

**Key Data Points:**
| Chip | GPU Cores | Bandwidth | Default TG |
|:-----|:---------:|:---------:|:----------:|
| M1 base | 8 | 68 GB/s | 128 |
| M1 Max | 32 | 400 GB/s | 256 |
| M3 Max | 40 | 400 GB/s | 256 |
| M4 Max | 40 | 546 GB/s | 256 |

**Files:** `src/zmlx/device.py`, `src/zmlx/autotune.py`

---

### 2. Cross-Backend Correctness Harness

**Problem:** ZMLX currently skips ALL tests on non-macOS-arm64 platforms. Pure-Python logic (IR, registry, config) is untested on Linux CI runners where MLX now supports CUDA and CPU backends.

**Goal:** Enable CI testing on Linux and catch GPU-generation-specific bugs.

**Action Items:**
- [ ] Add pytest markers (`@pytest.mark.metal`, `@pytest.mark.gpu`) for test classification
- [ ] Split modules into portable vs Metal-only in `__init__.py`
- [ ] Add `detect_backend()` function returning "metal"/"cuda"/"cpu"
- [ ] Create multi-backend CI workflow (ubuntu-latest, macos-14)
- [ ] Implement golden values cross-backend tests
- [ ] Add GPU-generation fingerprinting for M1-vs-M3 divergence detection

**Test Coverage Target:**
- 25+ pure-Python tests run on Linux CPU
- Full suite runs on macOS Metal
- Golden values stable across backends (atol=1e-4)

**Files:** `tests/conftest.py`, `src/zmlx/_compat.py`, `.github/workflows/ci.yml`

---

## P1: High Impact

### 3. Auto-Fusion Pattern Discovery

**Problem:** Adding new fusion patterns requires hand-writing `PatchPattern` classes. Every model architecture variant needs manual work.

**Goal:** Trace model forward pass, match against declarative fusion table, synthesize patterns at runtime.

**Action Items:**
- [ ] Implement `SubmoduleTracer` to record module call boundaries
- [ ] Create `FUSION_TABLE` declarative table for known fusible patterns
- [ ] Build `SynthesizedPattern` class implementing `PatchPattern` protocol dynamically
- [ ] Add `auto_patch()` and `discover_patterns()` public APIs
- [ ] Handle attribute name variants (gate_proj/up_proj/down_proj vs w1/w2/w3)

**Usage:**
```python
# Current approach
from zmlx.patch import patch
patch(model)  # uses hand-written patterns

# New approach
patterns = zmlx.patch.discover_patterns(model, sample)
model = zmlx.patch.auto_patch(model, sample)  # runtime pattern synthesis
```

**Files:** `src/zmlx/patch/_tracer.py`, `_fusion_table.py`, `_synthesize.py`, `_discovery.py`

---

### 4. Fused Dequantize + Compute

**Problem:** Current quant kernels dequantize to full-precision intermediate before consumer op, doubling memory traffic. LLM inference is memory-bandwidth-bound.

**Goal:** Fuse dequantization into consumer ops (activation, norm).

**Action Items:**
- [ ] Add MSL helpers for reading MLX packed uint32 format
- [ ] Implement `dequantize_mlx`, `dequantize_silu_mlx`, `dequantize_gelu_mlx`
- [ ] Create `quant_swiglu_mlp` patch pattern for quantized SwiGLU
- [ ] Add `elementwise_dequant_unary_source` codegen template
- [ ] Ensure bit-exact agreement with `mx.dequantize`

**Priority Order:**
1. dequant + activation (silu, gelu)
2. dequant + RMSNorm
3. dequant + RoPE
4. dequant + SwiGLU (gate+up fused)

**Files:** `src/zmlx/msl.py`, `src/zmlx/kernels/quant.py`, `src/zmlx/codegen.py`

---

## P2: Core Acceleration

### 5. Flash Attention (Tiled, Shared Memory)

**Problem:** Need memory-efficient fused attention with O(1) intermediate memory. Target use cases: custom masks, sliding window, paged KV integration.

**Goal:** Implement tiled Flash Attention kernel using Metal threadgroup memory.

**Action Items:**
- [ ] Implement online softmax algorithm for Q-row processing
- [ ] Support tile sizes: Bq=32, Bk=32 for D=64; Bq=16, Bk=32 for D=128
- [ ] Add causal mask fused into kernel
- [ ] Implement backward pass (two-kernel recomputation approach)
- [ ] Integrate with paged KV cache
- [ ] Autotune over (Bq, Bk) candidates

**Performance Expectations:**
- Standard shapes: 50-70% of `mx.fast.scaled_dot_product_attention`
- Custom masks/sliding window: 2-5x faster than naive implementation
- Paged prefill: Novel capability not in MLX built-in

**Files:** `src/zmlx/kernels/attention.py`

---

### 6. CPU/GPU Stream Scheduling

**Problem:** Current training loop runs everything synchronously on GPU, leaving CPU idle during forward/backward.

**Goal:** Overlap CPU batch preparation with GPU computation.

**Action Items:**
- [ ] Implement prefetch iterator wrapper with `mx.new_stream(mx.cpu)`
- [ ] Add `TrainConfig.prefetch_depth` field (default 0 = disabled)
- [ ] Add `_gating_cpu()` and `_topk_gating_cpu_fallback()` for MoE pattern
- [ ] Implement gradient correctness tests for CPU-gating backward pass

**Expected Improvement:** 0.5-4% throughput gain per step (batch prep is 0.5-2ms vs 50-200ms forward/backward)

**Files:** `src/zmlx/train/prefetch.py`, `src/zmlx/train/config.py`, `src/zmlx/patch/patterns/moe_mlp.py`

---

## P3: Infrastructure & Future

### 7. Paged KV Cache with UMA-Aware Scheduling

**Problem:** vLLM-style PagedAttention on Apple Silicon can leverage unified memory for zero-copy CPU/GPU access.

**Goal:** Implement full paged KV cache with O(1) alloc/free and LRU eviction.

**Action Items:**
- [ ] Implement `PagePool` with pre-allocated contiguous buffer
- [ ] Build `BlockAllocator` with doubly-linked free list
- [ ] Create `KVCacheManager` orchestrating sequence lifecycle
- [ ] Add LRU eviction and memory pressure detection
- [ ] Implement metadata-only defragmentation for UMA

**UMA Advantages vs CUDA:**
- Block table updates: Direct CPU writes, no `cudaMemcpyAsync`
- Page swap: No-op (same physical address)
- Eviction cost: Just update metadata (no copy)

**Files:** `src/zmlx/serving/page_pool.py`, `block_allocator.py`, `kv_cache_manager.py`

---

### 8. Micro-Benchmark Driven Device Scheduling

**Problem:** Small tensor operations (embedding lookups, MoE gating) can be faster on CPU due to GPU kernel launch overhead.

**Goal:** Profile each submodule on CPU vs GPU and route accordingly.

**Action Items:**
- [ ] Implement `DeviceProfiler` hooking into `nn.Module.__call__`
- [ ] Create `DevicePlacementPolicy` combining profiling + heuristics
- [ ] Add stream wrapping for CPU-routed submodules
- [ ] Integrate with `smart_patch()` as post-fusion optimization
- [ ] Add persistent placement cache

**Constraints:**
- Fused Metal kernels never route to CPU
- Linear/QuantizedLinear/Attention never route to CPU
- Only apply if end-to-end benchmark improves by >= 3%

**Files:** `src/zmlx/schedule/profiler.py`, `policy.py`, `apply.py`, `cache.py`

---

### 9. Transformer Block Fusion (Experimental, MLX C++ Dependency)

**Problem:** Kernel launch overhead dominates M=1 decode and small-tensor workloads.

**Goal:** Compile entire transformer blocks into a single Metal kernel with threadgroup-resident intermediates.

**Action Items:**
- [ ] Prototype block-level fusion in a feature-flagged experimental mode
- [ ] Identify MLX C++ primitives or hooks required for graph-level fusion
- [ ] Implement a minimal fused block (RMSNorm → QKV → RoPE → Attention → O proj → RMSNorm → MLP)
- [ ] Benchmark against layer-by-layer dispatch to quantify launch savings

**Dependency:** Requires MLX C++ support for multi-op fusion or custom graph compiler hooks.

**Files:** `docs/EXPERIMENTAL_MLX.md`, `src/zmlx/patch/experimental/`

---

### 10. Model-Specific JIT Kernel Cache

**Problem:** Static kernels are generic; model/sequence-specialized kernels can improve throughput.

**Goal:** Compile and cache kernels keyed by `(model_arch, seq_len, batch_size)` with trace-driven shape specialization.

**Action Items:**
- [ ] Extend autotune cache schema to capture model/shape keys
- [ ] Add a lightweight tracer for symbolic shapes at model load
- [ ] Implement compile-once + reuse flow for repeated decode shapes

**Files:** `src/zmlx/autotune.py`, `src/zmlx/patch/_tracer.py`

---

## Quick Wins (Can be done in parallel)

These are smaller tasks that don't require deep architectural changes:

1. **Documentation improvements**
   - [ ] Add troubleshooting guide for common patch failures
   - [ ] Create model compatibility matrix
   - [ ] Document kernel debugging with Metal Debugger

2. **Testing improvements**
   - [ ] Add property-based tests for kernel correctness
   - [ ] Create benchmark regression suite
   - [ ] Add memory leak detection for Metal kernels

3. **Developer experience**
   - [ ] Add `zmlx doctor` command for environment diagnostics
   - [ ] Improve error messages for unsupported shapes/dtypes
   - [ ] Add progress bars for long-running autotune

---

## Dependencies & Ordering

```
1. Per-Device Autotune Profiles ──────────────────────────────────────┐
                                                                      ├→ 8. Device Scheduling
2. Cross-Backend Correctness Harness (independent)                    │
                                                                      │
3. Auto-Fusion Pattern Discovery ─────→ (enhances patch system) ──────┤
                                                                      │
4. Flash Attention (32x32 tiles) ──────→ 7. Paged KV Cache ──────────┤
                                                                      │
5. CPU/GPU Stream Scheduling ─────────────────────────────────────────┤
                                                                      │
6. Fused Dequant + Compute ───────────────────────────────────────────┘
```

Items 1, 2, 3, 5, and 6 can be developed in parallel. Item 4 feeds into Item 7. Item 8 depends on Items 1, 3, and profiling infrastructure.

---

## Success Metrics

| Metric | Current | 3-Month Target | 6-Month Target |
|:-------|:-------:|:--------------:|:--------------:|
| Model coverage (tested) | 5 families | 8 families | 12 families |
| CI test pass rate | 0% (Linux) | 100% (Linux CPU) | 100% (Linux CPU/CUDA) |
| Autotune search time | ~10s | ~3s | ~1s (cached) |
| MoE speedup vs baseline | 1.0-1.1x | 1.3-1.6x | 1.5-2.0x |
| Dense model regression | 0% | 0% | 0% |
| Community contributors | 1 | 3 | 5 |

---

## Communication

- **Weekly updates:** Post progress to GitHub Discussions
- **Monthly reviews:** Update this roadmap based on learnings
- **Quarterly planning:** Re-prioritize based on MLX ecosystem changes

---

## Appendix: TPU³ Implementation Strategy

This appendix is the "engineering checklist" view of TPU³: what we ship first, and what comes next.

### 1) MVP Core Primitive: Decode GEMV (M=1)

Ship a working progressive int4 GEMV first (decode path is where ZMLX wins today):

- Pass 1: `hi2` approximation + `err_est` bound per output channel
- Pass 2: masked `lo2` residual refine

Reference API shape (MVP):

```python
# x: [K]
# w4: [N, K/8] uint32 packed nibbles
# scales: [N, K/group]
# significance: optional [N] (e.g., MoE gate weights)
y, err_est = zmlx.tpu3.gemv_anytime_int4(x, w4, scales, eps=1e-3, significance=None)
```

### 2) Policy: Spend Bits Where It Matters

Start with a conservative scalar rule and evolve toward fused GPU-side decision-making:

- MVP: `(significance * err_est) > eps` (or `err_est > eps` if `significance is None`)
- Next: generate mask on GPU (tiny kernel) to avoid CPU round-trips

### 3) Follow-on Upgrades (ZMLX-shaped)

- M<=32 batched decode (compute `[M, N]` in one dispatch to cut launch overhead)
- MoE integration: pass `significance=p_e` so low-prob experts stay hi2-only
- Attention FlashRefine: progressive score refinement for likely top-R keys

---

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX GitHub Issues](https://github.com/ml-explore/mlx/issues)
- [MLX PyPI](https://pypi.org/project/mlx/)
- [ZMLX Benchmarks](../benchmarks/results/TEST_SUMMARY.md)

---

*This roadmap is a living document. Priorities may shift based on user feedback, MLX updates, and hardware evolution.*
