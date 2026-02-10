# Next Steps: Unexplored Inference Optimizations

> **Date:** February 5, 2026
> **Scope:** Areas NOT already covered by `docs/ROADMAP.md`. Focused on inference speedups without training.
> **Models:** LFM2, GLM-4.7-Flash, Qwen3-30B-A3B (MoE); Llama-family (dense)

---

## Tier 1: Low-Hanging Fruit (days, high confidence)

### 1.1 mx.compile() — Zero Usage Today

**Finding:** `mx.compile` does not appear anywhere in `src/`. This is MLX's graph-level
compilation/tracing pass that fuses operations and eliminates intermediate buffers. It is
free to apply and can compound with ZMLX's kernel fusions.

**What it does:** Traces a function, builds a computation graph, merges common subexpressions,
and fuses elementwise ops into single dispatches. First call compiles; subsequent calls reuse.

**Why it matters for ZMLX:** Even with fused SwiGLU kernels, the surrounding operations
(residual adds, norms, projections) are dispatched individually. `mx.compile` can fuse the
non-ZMLX parts of each layer. This is orthogonal to ZMLX's Metal kernels — they compound.

**Action:**
- [ ] Wrap the model's `__call__` in `mx.compile()` after `zmlx.patch()` and benchmark
- [ ] Test that ZMLX's custom Metal kernels (via `mx.fast.metal_kernel`) survive compilation
      — they should, since MLX treats them as opaque primitives, but verify
- [ ] If it works, add a `compiled=True` option to `patch()` or a `zmlx.compile(model)` helper
- [ ] Validate token fidelity: compiled output must match uncompiled exactly

**Risk:** Low. `mx.compile` is stable in MLX 0.30+. Custom Metal kernels should pass through
as opaque ops. Worst case: 0% gain, no regression.

**Expected gain:** 1-5% on decode (fewer dispatch round-trips for non-fused ops).

---

### 1.2 Re-Benchmark Against mlx-lm v0.30.4+ Compiled SwiGLU

**Finding:** mlx-lm v0.30.4 (January 2026) added compiled SwiGLU operations. This means
the baseline we benchmarked against has gotten faster. Our `swiglu_mlp` pattern may now
provide less benefit or even regress on some models.

**Action:**
- [ ] Re-run `python -m zmlx.validate` on LFM2, GLM, Qwen3 with mlx-lm >= 0.30.4
- [ ] Compare decode tok/s for patched vs unpatched with the new baseline
- [ ] If `swiglu_mlp` regresses on any family, add to `_PERF_EXCLUDES`
- [ ] Update repro capsules with new baseline numbers

**Risk:** We may find `swiglu_mlp` is no longer beneficial for dense layers. MoE fusions
(which are more complex than what mlx-lm compiles) should still hold.

---

### 1.3 Fix KV Cache Quantization Defaults

**Finding:** ZMLX's `kv_cache.py` defaults `ZMLX_QUANTIZED_KV_START` to `0` (quantize from
the first token). mlx-lm defaults to `5000` — preserving full-precision KV for the first
5000 tokens, which protects quality on critical prefix context.

**Action:**
- [ ] Change default to `5000` to match mlx-lm and preserve quality
- [ ] Document the tradeoff: lower start = more memory savings, higher start = better quality

---

### 1.4 Memory Wiring Helper (wired_limit)

**Finding:** mlx-lm v0.29.0 introduced `wired_limit` — a context manager (macOS 15+) that
wires model + cache memory to prevent the OS from paging it out during inference. This fixed
a 10x slowdown in batch generation. ZMLX doesn't surface this.

**Action:**
- [ ] Add `zmlx.wired_limit()` helper that wraps `mx.metal.wired_limit()` with safe defaults
- [ ] Auto-calculate the right wiring budget based on model size + expected KV cache
- [ ] Document as a best practice for serving and long-context inference

---

## Tier 2: Medium Effort, High Impact (weeks)

### 2.1 Speculative Decoding with ReDrafter

**Finding:** Apple's [ReDrafter](https://github.com/apple/ml-recurrent-drafter) has an MLX
implementation showing **1.37x on M1 Max, 1.52x on M2 Ultra** speedups. It uses a lightweight
RNN draft model conditioned on the main LLM's hidden states.

**Why it matters for MoE:** Speculative decoding creates natural small batches (draft+verify
multiple tokens at once). For MoE models, this amortizes the expert routing overhead across
multiple tokens per forward pass. This should **compound** with ZMLX's fused expert dispatch.

**Action:**
- [ ] Validate that ZMLX-patched models produce correct output when used as the verifier
      in ReDrafter's MLX pipeline
- [ ] Benchmark the compound effect: ReDrafter + ZMLX patches vs ReDrafter alone
- [ ] If compound gain > 5%, create an integration guide / helper

**Expected gain:** 1.3-1.5x from speculative decoding, compounding with ZMLX's 5-12% MoE
gains, for a potential total of 1.4-1.7x vs unpatched non-speculative baseline.

---

### 2.2 vllm-mlx Integration

**Finding:** [vllm-mlx](https://github.com/waybarrios/vllm-mlx) provides continuous batching,
paged KV cache, and prefix caching for MLX models. It achieves **4.3x aggregate throughput**
at 16 concurrent requests. It uses mlx-lm models under the hood.

**Why it matters:** ZMLX's fused MoE kernels activate when M <= 32 (decode). In continuous
batching, each sequence still generates one token at a time, so the per-sequence M is 1 —
exactly where ZMLX shines. The two systems are naturally complementary.

**Critical question:** Does the M <= 32 threshold in `moe_mlp.py` check per-sequence or
per-batch? If the scheduler batches 64 requests and the fused kernel sees M=64, it falls back
to the unfused path. This needs investigation.

**Action:**
- [ ] Read vllm-mlx's model loading to understand where `zmlx.patch()` could hook in
- [ ] Verify M-threshold behavior when batch_size > 1 (does the MoE forward pass see M=1
      per expert, or M=batch_size?)
- [ ] If compatible, create a vllm-mlx plugin or monkey-patch integration
- [ ] Benchmark: vllm-mlx + ZMLX vs vllm-mlx alone at 1/4/16 concurrent requests

**Expected gain:** ZMLX's 5-12% decode speedup applied to every token in every concurrent
request, on top of vllm-mlx's batching gains.

---

### 2.3 Fused Routing + Dispatch + Combine Megakernel

**Finding:** Research (Alpha-MoE, SonicMoE) shows that fusing the entire MoE pipeline —
routing (top-k gating) + expert dispatch + weighted combine — into a single kernel can
eliminate 3-5 kernel launches per MoE layer.

**Current state:** ZMLX already fuses:
- Gate+up projection + SwiGLU via `gather_qmm_swiglu` (one launch instead of two)
- Expert output combine via `moe_combine` kernels (one launch)

But routing (softmax + top-k selection) and the dispatch logic are still separate launches.

**Action:**
- [ ] Profile the current MoE forward pass with Metal System Trace to quantify dispatch
      overhead per component (routing vs gate/up vs down vs combine)
- [ ] If routing dispatch is significant (> 5% of layer time), prototype a fused
      routing+dispatch kernel that outputs expert assignments + starts computation
- [ ] Consider fusing the down-projection + combine step (currently separate)

**Expected gain:** 2-5% per MoE layer from eliminated dispatch overhead. More impactful on
smaller models (LFM2-A1B) where dispatch overhead is a larger fraction of compute.

---

### 2.4 Expert Prediction and Prefetching

**Finding:** Recent research shows expert routing decisions can be predicted before the
attention block with 93-97% accuracy using cross-layer gate prediction. Since Apple Silicon
has unified memory (no PCIe transfer), the benefit isn't about prefetching from CPU→GPU, but
about **pre-computing routing decisions** to enable better kernel scheduling.

**How it helps on Apple Silicon:**
- Pre-compute expert assignments during the attention phase (which is running anyway)
- Use the pre-computed assignments to skip the full gating computation
- Enable speculative expert execution that overlaps with attention

**Action:**
- [ ] Measure correlation between adjacent layers' expert assignments for our MoE models
      (GLM, Qwen3, LFM2): if expert overlap > 80%, prediction is viable
- [ ] Prototype a simple "copy previous layer's routing" predictor and measure accuracy
- [ ] If accuracy > 90%, implement speculative expert dispatch with rollback on misprediction

**Expected gain:** 3-8% if gating overhead is significant and prediction accuracy is high.

---

### 2.5 Prefix Caching Validation

**Finding:** mlx-lm supports `cache_prompt` for serializing KV cache to disk. vllm-mlx uses
SHA-256 prefix matching for 5.8x TTFT speedup. Neither has been validated with ZMLX-patched
models.

**Key insight:** ZMLX's fused kernels only activate during decode (M <= 32). During prefill
(M >> 32), the standard MLX code path runs. This means the KV cache produced during prefill
should be identical between patched and unpatched models. But we haven't verified this.

**Action:**
- [ ] Run `cache_prompt` on a ZMLX-patched model, then load the cache and verify outputs
      match the unpatched model's cached prefill
- [ ] If identical, document as a supported workflow
- [ ] If not identical, investigate which pattern is active during prefill and fix

---

## Tier 3: Significant Effort, Speculative (months)

### 3.1 Mixed-Precision Expert Quantization

**Finding:** DynaExq and MxMoE research shows that not all experts need the same precision.
"Hot" experts (frequently activated) benefit from higher precision, while "cold" experts
can be aggressively quantized (2-bit) with minimal quality loss.

**Why it matters:** MoE models like GLM-4.7 have 64 experts but only activate 4 per token.
The inactive 60 experts' weights still consume memory bandwidth when loaded. If cold experts
were 2-bit instead of 4-bit, we'd halve their memory footprint.

**Action:**
- [ ] Profile expert activation frequency across a representative corpus for GLM/Qwen3/LFM2
- [ ] Identify hot (top 20%) vs cold (bottom 50%) experts per model
- [ ] Prototype a mixed-precision model where hot experts stay 4-bit, cold experts go 2-bit
- [ ] Validate quality: compare perplexity and token agreement vs uniform 4-bit
- [ ] If quality holds, add a `mixed_quant_moe()` utility to ZMLX

**Expected gain:** 15-25% reduction in MoE layer memory bandwidth (smaller weights = faster
loads), translating to 3-8% decode speedup on bandwidth-bound workloads.

---

### 3.2 Metal 4 ML Features (macOS 26+)

**Finding:** WWDC 2025 introduced Metal 4 with significant ML features:

- **MTLTensor:** Native multi-dimensional tensor type with baked-in strides
- **ML Encoder (`MTL4MachineLearningCommandEncoder`):** Run entire neural networks on GPU
  timeline alongside compute/render, using CoreML-compiled `.mtlpackage` models
- **Shader ML:** Embed ML operations directly in compute shaders via `#include <metal_tensor>`,
  eliminating device memory round-trips between steps
- **Metal Performance Primitives (MPP):** In-shader `matmul2d` and `convolution` primitives

**Relevance to ZMLX:**
- **Shader ML + MPP** could allow ZMLX to embed small neural network operations (e.g., expert
  gating MLP, draft model) directly inside a larger compute kernel. Instead of dispatching
  separate kernels for gating → expert → combine, the entire MoE block could run in a single
  dispatch using MPP's `matmul2d` for expert projections.
- **ML Encoder** could run a speculative draft model on the GPU timeline in parallel with the
  main model's decode, without CPU coordination overhead.
- **MTLTensor** provides native support for the multi-dimensional indexing that MoE experts
  require (expert index × hidden dim × sequence).

**Constraints:**
- Requires macOS 26 (Tahoe), not yet released
- Requires Metal 4 GPU (M4 family minimum)
- MLX would need to adopt Metal 4 APIs — this is upstream work
- Shader ML execution models (`execution_thread`, `execution_simdgroup`) have uniform control
  flow requirements that may conflict with MoE's inherent divergence

**Action:**
- [ ] Track MLX team's adoption of Metal 4 features
- [ ] When macOS 26 ships, prototype a Shader ML kernel that runs a small MLP (the gating
      network) inside a compute kernel, measuring dispatch savings
- [ ] Evaluate MPP `matmul2d` performance vs MLX's existing `mx.fast.metal_kernel` matmul
      for the expert projection sizes we care about (small M, medium N/K)

---

### 3.3 Metal Function Stitching for Dynamic Kernel Composition

**Finding:** Metal's `[[stitchable]]` function qualifier and `MTLFunctionStitchingGraph` API
allow composing pre-compiled kernel fragments into new kernels at runtime without
recompilation. This could replace ZMLX's current approach of generating MSL source strings
and compiling them.

**Current ZMLX approach:** Generate MSL source → compile → cache by source hash. Works well
but the compile step can be slow for complex kernels.

**Stitching approach:** Pre-compile a library of kernel fragments (activation functions,
reductions, dequantization) as `[[stitchable]]` functions, then compose them at runtime via
the stitching graph API. New combinations (e.g., dequant+silu+swiglu) would be instant.

**Constraints:**
- `[[stitchable]]` functions can't be entry points — need a wrapper `[[kernel]]`
- No direct support for scalar parameters (need workarounds)
- Re-encoding required if parameters change
- Unclear if `mx.fast.metal_kernel` can work with stitched functions

**Action:**
- [ ] Prototype a stitchable activation library (silu, gelu, tanh, swiglu) outside of MLX
      to measure composition overhead vs recompilation
- [ ] If overhead is < 1ms, propose an `mx.fast.stitched_kernel` API to MLX team

---

### 3.4 Grouped GEMM for Expert Batching

**Finding:** PyTorch's Triton grouped GEMM kernel achieves **2.62x speedup** over manual
per-expert loops for MoE by packing expert weights into contiguous vectors and dispatching
a single batched matmul.

**Current ZMLX state:** `gather_qmm_swiglu` already implements a form of this for the
gate+up projections. But the down projection is still handled per-expert or via `gather_qmm`.

**Action:**
- [ ] Profile whether the down projection dispatch is a bottleneck (it's a separate
      `gather_qmm` call in the current flow)
- [ ] If yes, prototype a fused `gather_qmm_down_combine` that batches the down projection
      across all active experts and fuses the weighted combine
- [ ] This would reduce the MoE forward pass to just 2 dispatches:
      1. `gather_qmm_swiglu` (gate+up+activation)
      2. `gather_qmm_down_combine` (down+weighted sum)

**Expected gain:** 3-7% on models where down-projection dispatch is significant.

---

### 3.5 Tile-Aware GEMM Scheduling

**Finding:** Research on CUDA MoE kernels shows that column-major GEMM scheduling (processing
output tiles in column order rather than row order) improves L2 cache hit rates by 2-4x for
the shapes common in MoE expert projections.

**Why it matters for Apple Silicon:** Apple GPUs have a tile-based architecture with on-chip
tile memory. The order in which threadgroups process output tiles affects how often weight
data is reused from the GPU's caches. For MoE's narrow M (M=1 decode), the weight matrix
is the dominant memory read — column-major scheduling maximizes reuse across threadgroups.

**Action:**
- [ ] Analyze the threadgroup dispatch order in `gather_qmm_swiglu` and `gather_qmm`
- [ ] Prototype column-major dispatch ordering for the expert projection shapes
- [ ] Benchmark on M4 Max with Metal System Trace to verify cache hit improvement

---

### 3.6 Expert Pruning at Inference Time

**Finding:** MoBiLE (Big/Little Experts) and REAP research shows that replacing rarely-used
experts with smaller "little" versions (or skipping them entirely) can reduce compute by
30-50% with < 1% quality loss.

**How it could work in ZMLX:**
- Track expert activation frequency during a warmup phase
- Replace bottom-N% experts with identity functions or smaller approximations
- This reduces the number of expert projections per token

**Constraint:** Modifying model weights is invasive and model-specific. Better suited as a
separate tool (`zmlx prune-experts`) than as a runtime patch.

**Action:**
- [ ] Implement expert frequency profiling in `moe_mlp.py` (count activations per expert
      during first K tokens)
- [ ] Expose the frequency data via `zmlx.patch.expert_stats(model)`
- [ ] Defer actual pruning to a future release

---

### 3.7 DeepSeek V3/V3.2 MLA (Multi-head Latent Attention)

**Finding:** mlx-lm v0.30.6 added DeepSeek V3.2 MLA support. MLA uses compressed latent
representations for KV cache (4-8x smaller KV cache than standard MHA). This is a new
architecture pattern distinct from GQA.

**Why it matters:** DeepSeek V3 and Kimi-K2.5 (both MoE, both in ZMLX's model matrix) use
MLA. ZMLX's current MoE fusion targets the expert MLP, which is orthogonal to attention.
But MLA's compressed KV cache means these models have different memory profiles — the
bottleneck shifts more toward MoE expert compute vs KV cache bandwidth.

**Action:**
- [ ] Profile DeepSeek V3.2 (when a 4-bit quant fits in RAM) to understand where time is
      spent: MoE experts vs MLA attention vs routing
- [ ] Verify ZMLX's `moe_mlp` pattern works with DeepSeek's MoE architecture
- [ ] Consider MLA-specific fusions if attention becomes the bottleneck (fused latent
      projection + attention)

---

## Summary: Priority-Ordered Action Items

| # | Item | Effort | Expected Gain | Confidence |
|:-:|:-----|:------:|:-------------:|:----------:|
| 1 | mx.compile() integration | 1-2 days | 1-5% | High |
| 2 | Re-benchmark vs mlx-lm v0.30.4+ | 1 day | Defensive | High |
| 3 | Fix KV quantization defaults | 1 hour | Quality | High |
| 4 | wired_limit helper | 1 day | Stability | High |
| 5 | Speculative decoding (ReDrafter) | 1-2 weeks | 30-50% | Medium |
| 6 | vllm-mlx integration | 2-3 weeks | Throughput 2-4x | Medium |
| 7 | Fused routing+dispatch+combine | 2-3 weeks | 2-5% | Medium |
| 8 | Expert prediction/prefetching | 2-3 weeks | 3-8% | Medium-Low |
| 9 | Prefix caching validation | 2-3 days | Enables feature | High |
| 10 | Mixed-precision experts | 1-2 months | 3-8% | Medium |
| 11 | Metal 4 ML features | Blocked on macOS 26 | Unknown | Low |
| 12 | Function stitching | 2-4 weeks | Compile time | Low |
| 13 | Grouped GEMM (down+combine) | 2-3 weeks | 3-7% | Medium |
| 14 | Tile-aware GEMM scheduling | 1-2 weeks | 2-4% | Medium-Low |
| 15 | Expert pruning | 1-2 months | 5-15% | Medium |
| 16 | DeepSeek V3 MLA fusion | 1-2 months | Unknown | Low |

**Recommended immediate sprint (items 1-4, 9):** ~1 week of work, defensive + enabling.
These are all low-risk, high-confidence items that either unlock gains or prevent regressions.

**Recommended next sprint (items 5-7):** ~3-4 weeks. Speculative decoding and vllm-mlx are
the two highest-leverage items because they provide multiplicative (not additive) gains on
top of existing ZMLX fusions.

---

## Research Sources

- MLX compile docs: `ml-explore.github.io/mlx/build/html/usage/compile.html`
- mlx-lm releases v0.29.0-v0.30.6: compiled SwiGLU, wired_limit, batch KV cache
- Apple ReDrafter: `github.com/apple/ml-recurrent-drafter` (MLX implementation)
- vllm-mlx: `github.com/waybarrios/vllm-mlx` (arxiv:2601.19139)
- Metal 4 ML: WWDC 2025 Session 262 (MTLTensor, ML Encoder, Shader ML, MPP)
- Metal function stitching: `MTLFunctionStitchingGraph`, `[[stitchable]]` qualifier
- ANEMLL: `github.com/Anemll/Anemll` (ANE LLM inference — limited to small models)
- Alpha-MoE / SonicMoE: fused routing+dispatch+combine megakernels
- DynaExq / MxMoE: dynamic mixed-precision expert quantization
- MoBiLE: Big/Little expert pruning
- PyTorch grouped GEMM: 2.62x speedup via persistent cache-aware Triton kernel
- Metal Performance Optimization: WWDC 2020 session on Apple Silicon GPU tuning
- Multi-node expert parallelism on Mac Studio: arxiv:2506.23635
- Metal FlashAttention v2.5: Draw Things, Neural Accelerator integration
