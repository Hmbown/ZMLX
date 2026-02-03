# ZMLX Roadmap

> **Updated:** January 30, 2025  
> **Status:** Following contact from Awni Hannun (Apple MLX team)  
> **Current Version:** 0.6.3

This roadmap prioritizes features based on technical impact, user demand, and strategic alignment with MLX ecosystem growth.

---

## Priority Matrix

| Priority | Feature | Impact | Effort | Owner |
|:--------:|:--------|:------:|:------:|:------|
| **P0** | Per-Device Autotune Profiles | High | Medium | TBD |
| **P0** | Cross-Backend Correctness Harness | High | Medium | TBD |
| **P1** | Auto-Fusion Pattern Discovery | High | High | TBD |
| **P1** | Fused Dequantize + Compute | High | Medium | TBD |
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

## P0: Critical Path (Next 4-6 Weeks)

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

## P1: High Impact (Next 2-3 Months)

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
1. dequant + activation (silu, gelu) - **Phase 1**
2. dequant + RMSNorm - **Phase 2**
3. dequant + RoPE - **Phase 3**
4. dequant + SwiGLU (gate+up fused) - **Phase 4**

**Files:** `src/zmlx/msl.py`, `src/zmlx/kernels/quant.py`, `src/zmlx/codegen.py`

---

## P2: Medium Term (Next 3-6 Months)

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

## P3: Future Work (6+ Months)

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

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX Custom Metal Kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX GitHub Issues](https://github.com/ml-explore/mlx/issues)
- [ZMLX Benchmarks](../benchmarks/results/TEST_SUMMARY.md)

---

*This roadmap is a living document. Priorities may shift based on user feedback, MLX updates, and hardware evolution.*
