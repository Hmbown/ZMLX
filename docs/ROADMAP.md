# ZMLX Roadmap

> Implementation roadmap for ZMLX's next-generation features: hardware-aware autotuning, cross-backend testing, auto-fusion discovery, flash attention, CPU/GPU scheduling, fused dequant, paged KV cache, and device-aware scheduling.
>
> Generated: January 2026

## Dependency Graph

```
1. Per-Device Autotune Profiles ──────────────────────────────────────┐
                                                                      ├─→ 8. Device Scheduling
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

Items 1, 2, 3, 5, and 6 are independent and can be developed in parallel. Item 4 feeds into Item 7. Item 8 depends on Items 1, 3, and profiling infrastructure.

---

## 1. Per-Device Autotune Profiles

### Motivation

Every Apple Silicon chip (M1 through M4, in base/Pro/Max/Ultra variants) has different GPU core counts, memory bandwidth, and microarchitectural features (e.g., M3+ dynamic caching). ZMLX's autotuner currently searches a flat list of threadgroup candidates `(32, 64, 128, 256, 512, 1024)` regardless of hardware. By baking in per-device starting points, we can reduce search time and improve defaults when autotuning is disabled.

### Current State

- `src/zmlx/device.py` has `DeviceProfile` with `family`, `variant`, `gpu_cores`, `simd_width`, and `default_threadgroup_candidates`. Detection via `_sysctl("machdep.cpu.brand_string")`.
- **Bug**: `detect_device()` uses `hw.perflevel0.logicalcpu` for GPU core count, which actually returns CPU performance cores, not GPU cores.
- `src/zmlx/autotune.py` has `AutotuneKey`, `GLOBAL_AUTOTUNE_CACHE`, `_FAST_CACHE` (512-entry LRU), and an `@autotune()` decorator that is currently a **stub** (line 241).

### Implementation Plan

#### Phase 1: Device Profile Enrichment

| Deliverable | File | Description |
|---|---|---|
| Fix GPU core detection | `src/zmlx/device.py` | Replace `hw.perflevel0.logicalcpu` with `system_profiler SPDisplaysDataType` + fallback lookup table |
| Add `memory_bandwidth_gbs` | `src/zmlx/device.py` | New field on `DeviceProfile`, populated from known-values lookup |
| Add `has_dynamic_caching` | `src/zmlx/device.py` | True for M3/M4 (Apple family 9+) |
| Add `gpu_family_version` | `src/zmlx/device.py` | 7 for M1, 8 for M2, 9 for M3/M4 |
| Add `tier` property | `src/zmlx/device.py` | Returns `"base"`, `"pro"`, `"max"`, or `"ultra"` |
| Add `DeviceTuningProfile` | `src/zmlx/device.py` | Frozen dataclass with `threadgroup_candidates`, `default_threadgroup`, `tile_m`, `tile_n`, `prefer_simd_reduction` |
| Add `_BUILTIN_PROFILES` | `src/zmlx/device.py` | Dict of `(family, tier) -> DeviceTuningProfile` for all 16 chip variants |
| Add `get_tuning_profile()` | `src/zmlx/device.py` | Lookup with conservative fallback |

**Key data points** for baked-in profiles:

| Chip | GPU Cores | Bandwidth (GB/s) | Default TG | Prefer SIMD Reduction |
|------|-----------|-------------------|------------|----------------------|
| M1 base | 8 | 68 | 128 | No |
| M1 Pro | 16 | 200 | 256 | No |
| M1 Max | 32 | 400 | 256 | No |
| M2 base | 10 | 100 | 256 | No |
| M2 Pro | 19 | 200 | 256 | No |
| M3 base | 10 | 100 | 256 | Yes (improved SIMD shuffle) |
| M3 Max | 40 | 400 | 256 | Yes |
| M4 base | 10 | 120 | 256 | Yes |
| M4 Pro | 20 | 273 | 256 | Yes |
| M4 Max | 40 | 546 | 256 | Yes |

GPU core detection fix -- use `system_profiler SPDisplaysDataType` to read "Total Number of Cores", with a fallback lookup table keyed by chip name string.

#### Phase 2: Autotune Integration

| Deliverable | File | Description |
|---|---|---|
| Profile-guided candidates | `src/zmlx/autotune.py` | `get_autotuned_config()` uses `get_tuning_profile().threadgroup_candidates` as default |
| Early-exit heuristic | `src/zmlx/autotune.py` | Stop search when improvement margin < 2% (candidates ordered best-first) |
| Implement `@autotune()` | `src/zmlx/autotune.py` | Replace stub with working decorator: intercept first call, extract kernel+inputs, run search, cache result per shape key |
| Consolidate `_modules.py` | `src/zmlx/patch/_modules.py` | Use `get_tuning_profile()` for candidate ordering |

The `@autotune()` decorator design:

```python
@autotune(warmup=2, iters=8)
def my_kernel_launcher(kernel, x, w, *, threadgroup=(256, 1, 1), **kw):
    return kernel(x, w, threadgroup=threadgroup, **kw)
```

On first call per input-shape signature, runs the search. All subsequent calls use the cached result.

#### Phase 3: Persistent Cache v3

Upgrade `~/.cache/zmlx/autotune_v2.json` to v3 schema with device profile metadata, timestamps, and measured timings. Migration: detect version on load, read both v2 and v3.

### Acceptance Criteria

- `detect_device()` returns correct `gpu_cores` and `memory_bandwidth_gbs` on M1/M2/M3/M4 hardware
- `get_tuning_profile()` returns a valid profile for every known chip, with conservative fallback for unknown devices
- `@autotune()` correctly discovers best threadgroup on first call and caches it
- Baked-in `default_threadgroup` is within 10% of measured optimum for the top 10 most-used kernels

### References

- [Apple GPU microarchitecture benchmarks (philipturner/metal-benchmarks)](https://github.com/philipturner/metal-benchmarks)
- [Apple Developer: Calculating threadgroup and grid sizes](https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes)
- [Apple Developer: GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375/)
- [Optimizing Parallel Reduction in Metal for Apple M1](https://kieber-emmons.medium.com/optimizing-parallel-reduction-in-metal-for-apple-m1-8e8677b49b01)

---

## 2. Cross-Backend Correctness Harness

### Motivation

ZMLX currently skips **all** tests on any platform that is not macOS arm64 with Metal. This means:

- No tests run on Linux CI runners (where MLX now supports CUDA and CPU backends)
- Pure-Python logic (IR serialization, registry, config parsing, autotune cache persistence) is untested outside macOS
- No mechanism to catch GPU-generation-specific bugs like [MLX issue #2205](https://github.com/ml-explore/mlx/issues/2205) (same kernel, different results on M1 Max vs M3 Max)

### Current State

`tests/conftest.py` applies a blanket skip via `pytest_collection_modifyitems` -- a single boolean gate where either everything runs or nothing does.

**Test file audit** -- two distinct categories exist:

| Category | Files | Requires GPU |
|----------|-------|-------------|
| Metal kernel tests | `test_kernels_catalog.py`, `test_autograd.py`, `test_elementwise.py`, `test_new_features.py`, `test_vlsp_kernels.py`, `test_patch.py` | Yes |
| Pure-Python logic | `test_ir.py`, `test_registry_enhanced.py`, `test_train_config.py`, `test_autotune_persistent.py` | No |

The four pure-Python test files (~30 test functions) are currently skipped on Linux for no reason.

### Implementation Plan

#### Phase 1: Marker-Based Test Classification

Replace the blanket skip with pytest markers:

```python
# tests/conftest.py
def pytest_collection_modifyitems(config, items):
    backend = _detect_backend()  # "metal", "cuda", or "cpu"
    for item in items:
        markers = {m.name for m in item.iter_markers()}
        if "metal" in markers and backend != "metal":
            item.add_marker(pytest.mark.skip(reason=f"Requires Metal (detected: {backend})"))
        elif "gpu" in markers and backend not in ("metal", "cuda"):
            item.add_marker(pytest.mark.skip(reason=f"Requires GPU (detected: {backend})"))
        # Unmarked tests run everywhere
```

Add `pytestmark = pytest.mark.metal` to each GPU test file. Leave pure-Python tests unmarked.

#### Phase 2: Relaxed Import Guard

`src/zmlx/__init__.py` currently raises `RuntimeError` on any attribute access from non-macOS-arm64. Split modules into portable (`testing`, `registry`, `codegen`, `ir`, `autotune`) and Metal-only (`autograd`, `elementwise`, `kernels`, `metal`, `rowwise`, `msl`, `patch`).

Add `detect_backend()` to `src/zmlx/_compat.py` returning `"metal"`, `"cuda"`, or `"cpu"`.

#### Phase 3: Multi-Backend CI Workflow

```yaml
jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - run: pip install "mlx[cpu]" && pip install .[dev]
      - run: pytest -q -m "not metal and not gpu"

  test-metal:
    runs-on: macos-14
    steps:
      - run: pip install .[dev]
      - run: pytest -q
```

#### Phase 4: Golden Values Cross-Backend Tests

`tests/test_cross_backend.py` computes reference MLX operations (softmax, RMSNorm) and compares against committed golden values in `tests/golden_values.json`. This catches silent numeric changes across backends or MLX versions.

#### Phase 5: GPU-Generation Fingerprinting

`tests/test_gpu_generation.py` runs ZMLX kernels, records output statistics tagged by device family, and saves to `tests/fingerprints/kernel_fingerprints.json`. On subsequent runs, compares against saved values to detect M1-vs-M3-style divergences. JUnit XML properties enable CI dashboard comparison.

### Acceptance Criteria

- `pytest -q -m "not metal and not gpu"` passes on `ubuntu-latest` with `mlx[cpu]` (25+ test functions)
- `pytest -q` passes on `macos-14` with full suite
- `from zmlx.codegen import elementwise_unary_source` succeeds on Linux CPU
- Golden values stable across Metal and CPU backends within `atol=1e-4`

### References

- [MLX Build and Install Documentation](https://ml-explore.github.io/mlx/build/html/install.html)
- [MLX GitHub Issue #2205](https://github.com/ml-explore/mlx/issues/2205) -- cross-generation divergence
- [MLX Custom Metal Kernels Documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)

---

## 3. Auto-Fusion Pattern Discovery

### Motivation

ZMLX's patch system currently has 7 hand-written patterns (swiglu_mlp, geglu_mlp, moe_mlp, rmsnorm, layernorm, softmax, residual_norm). Adding a new pattern requires writing a `PatchPattern` class with `matches()` and `apply()` methods. This approach doesn't scale: every new model architecture variant (different attribute names, different gating structures) needs a new hand-written pattern.

Auto-fusion discovery traces a model's forward pass, matches submodule call boundaries against a declarative fusion table, and synthesizes `PatchPattern` instances at runtime.

### Architecture

Four new modules under `src/zmlx/patch/`:

| Module | Purpose | Lines (est.) |
|--------|---------|-------------|
| `_tracer.py` | Record module call boundaries and op sequences | ~200 |
| `_fusion_table.py` | Declarative table of all known fusible patterns | ~150 |
| `_synthesize.py` | Generate `PatchPattern` instances from table entries | ~120 |
| `_discovery.py` | Tie tracer + table + synthesizer together | ~100 |

### Component 1: SubmoduleTracer

Wraps `nn.Module.__call__` to record which child modules are called, their input/output shapes, and execution order during a single forward pass:

```python
@dataclass
class ModuleTrace:
    module_type: str           # e.g., "LlamaMLP"
    child_calls: list[str]     # e.g., ["gate_proj", "up_proj", "down_proj"]
    input_shapes: tuple        # shapes of inputs to __call__
    output_shapes: tuple       # shapes of outputs
    attributes: frozenset[str] # set of nn.Module child attribute names
    ops: list[OpRecord]        # Phase 2: mx.* function calls within this module
```

### Component 2: Fusion Table

A declarative table encoding all known fusible patterns:

```python
FUSION_TABLE: list[FusionEntry] = [
    FusionEntry(
        name="swiglu_mlp",
        category=FusionCategory.GATED_MLP,
        signature=FusionSignature(
            required_attrs=(frozenset({"gate_proj", "up_proj", "down_proj"}),),
            forbidden_parent_attrs=frozenset({"router", "gate"}),
            activation_type="silu",
        ),
        existing_pattern="swiglu_mlp",
        priority=10,
    ),
    # ... entries for geglu_mlp, moe_mlp, rmsnorm, layernorm, etc.
    FusionEntry(
        name="swiglu_mlp_alt_names",  # Mistral w1/w2/w3 naming
        category=FusionCategory.GATED_MLP,
        signature=FusionSignature(
            required_attrs=(frozenset({"w1", "w2", "w3"}),),
            activation_type="silu",
        ),
        existing_pattern=None,  # auto-generated
        kernel_module="zmlx.kernels.transformer",
        kernel_function="swiglu2",
    ),
]
```

Entries with `existing_pattern != None` delegate to hand-written patterns. Entries with `kernel_module` + `kernel_function` get auto-synthesized.

### Component 3: Synthesized Patterns

`SynthesizedPattern` implements the `PatchPattern` protocol dynamically, generating `__call__` wrappers from table entries. It detects attribute name variants (gate_proj/up_proj/down_proj vs w1/w2/w3) and maps them to the appropriate kernel function.

Key design: synthesized patterns use `__call__` wrapping (not module replacement), ensuring they are revertible and compatible with `smart_patch`'s benchmark-and-revert loop.

### Component 4: Public API

```python
# Top-level: no pattern names needed
model = zmlx.patch.auto_patch(model, sample, verbose=True)

# Diagnostic: see what's fusible without patching
patterns = zmlx.patch.discover_patterns(model, sample, verbose=True)
```

### Phase 2 Extension: Op-Level Tracing

The initial implementation uses attribute-based matching. Phase 2 extends the tracer to capture actual `mx.*` function call sequences (by wrapping `mx.silu`, `mx.exp`, etc.), enabling matching against inline fusible patterns that don't correspond to named attributes.

### Acceptance Criteria

- `discover_patterns(llama_model, sample)` returns patterns equivalent to `["swiglu_mlp"]`
- `discover_patterns(qwen3_moe_model, sample)` returns `["swiglu_mlp", "moe_mlp"]`
- A model using `w1`/`w2`/`w3` naming gets auto-patched without a hand-written pattern
- Tracing adds < 100ms overhead for a single forward pass on a 30B parameter model
- `auto_patch()` produces identical results to `smart_patch()` on tested model families

---

## 4. Flash Attention (Tiled, Shared Memory)

### Motivation

Implement a Flash Attention kernel: tiled, memory-efficient fused Q*K^T softmax V using Metal threadgroup (shared) memory. The existing `attention_tile_proto` is a 16x16 dot-product-only prototype; `paged_attention` is a full decode kernel. This adds a prefill-capable Flash Attention with O(1) intermediate memory.

**Baseline to beat**: `mx.fast.scaled_dot_product_attention` (Apple's STEEL-based kernel). Our kernel targets use cases where Apple's kernel is suboptimal: custom masks, sliding window, paged KV integration, and non-standard head dimensions.

### Metal Hardware Constraints

| Constraint | Value |
|:---|:---|
| Max threadgroup memory | 32,768 bytes (32 KB) |
| Max threads per threadgroup | 1,024 |
| SIMD group width | 32 threads |
| `simdgroup_async_copy` | Available on M1+ |

### Tile Size Selection

All tiles (Q, K, V, scores, softmax state) must fit in 32 KB:

**Recommended default: Bq=32, Bk=32, D=64**

```
Q tile:   32 *  64 * 4 =  8,192 bytes  ( 8 KB)
K tile:   32 *  64 * 4 =  8,192 bytes  ( 8 KB)
Score:    32 *  32 * 4 =  4,096 bytes  ( 4 KB)
Softmax:  32 *   2 * 4 =    256 bytes
V tile:   32 *  64 * 4 =  8,192 bytes  ( 8 KB, aliases K tile after scores computed)
                   Total = ~29 KB  (fits)
```

**Fallback for D=128: Bq=16, Bk=32** with D tiled into 64-element chunks.

### Online Softmax Algorithm

The Flash Attention core: for each Q row, maintain running `m_i` (max) and `l_i` (exp sum) across K-block iterations. When a new K block produces scores, rescale the running V accumulator:

1. Find local max of new scores
2. Compute new global max
3. Rescale previous accumulator by `exp(m_prev - m_new)`
4. Accumulate new block's `exp(score - m_new)` contributions
5. Update running sum
6. Accumulate `P @ V` for this block

After all K blocks: `output[d] = acc[d] / l_final`.

### Implementation Plan

#### Phase 1: Forward-Only Flash Attention (MVP)

- `flash_attention(q, k, v, scale=, causal=)` in `src/zmlx/kernels/attention.py`
- Tile size selection for D=64, D=128 (constexpr-compiled)
- Causal mask fused into the kernel
- LSE output stored for backward pass
- GQA support via `kv_head_idx = head_idx / G`

**Acceptance**: `allclose(flash_attention(q, k, v), mx.fast.scaled_dot_product_attention(q, k, v), atol=1e-5)` for float32. Tested with B=1,4; H=8,32; N=128,512,2048; D=64,128.

#### Phase 2: Backward Pass

Two-kernel recomputation approach (avoids FP32 atomics, which Metal lacks natively):

- **Kernel 1**: Parallel over Q blocks, computes dQ
- **Kernel 2**: Parallel over KV blocks, computes dK and dV

This trades 2x compute for avoiding atomic conflicts. Total backward cost ~5x forward.

**Acceptance**: `mx.grad(lambda q: flash_attention(q, k, v).sum())(q)` matches finite-difference gradient (rtol=1e-3).

#### Phase 3: Paged Flash Attention

Replace contiguous K/V loads with block-table lookups, reusing the existing `paged_attention` block-indirect layout. Provides tiled Flash Attention for prefill with paged KV cache.

#### Phase 4: Autotune and Polish

Autotune over (Bq, Bk) candidates. Custom mask support. Benchmark script comparing vs `mx.fast.scaled_dot_product_attention`.

### Performance Expectations

| Scenario | Expected Result |
|:---|:---|
| Standard shapes (D=64/128), causal-only | 50-70% of `mx.fast.scaled_dot_product_attention` (honest target for `metal_kernel`-based impl) |
| Custom masks, sliding window | 2-5x faster than naive `softmax(Q @ K.T) @ V` |
| Paged prefill | Novel capability not available in MLX's built-in SDPA |

Note: `mx.fast.metal_kernel` introduces Python dispatch overhead and cannot use `simdgroup_async_copy`. The value is flexibility, not raw throughput on standard workloads.

### References

- [FlashAttention: Fast and Memory-Efficient Exact Attention (Dao et al.)](https://arxiv.org/abs/2205.14135)
- [Metal FlashAttention (philipturner)](https://github.com/philipturner/metal-flash-attention)
- [MLX scaled_dot_product_attention source](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scaled_dot_product_attention.metal)

---

## 5. CPU/GPU Stream Scheduling

### Motivation

MLX's unified memory architecture means arrays are accessible to both CPU and GPU with zero-copy semantics. MLX provides `mx.stream()`, `mx.new_stream()`, and `mx.async_eval()` for explicit device placement. Currently, ZMLX's training loop runs everything synchronously on the default GPU stream, leaving the CPU idle during forward/backward.

### Sub-project A: Data Loading Overlap

The `mlx_lm` trainer loop blocks on `mx.eval()` before fetching the next batch. Batch preparation (tokenization, padding, array construction) is CPU work that can overlap with GPU forward/backward.

**Design**: A prefetch wrapper around `iterate_batches` that pre-computes the next `prefetch_depth` batches on a dedicated CPU stream:

```python
def make_prefetch_iterator(base_iterate_batches, prefetch_depth=2):
    def prefetch_iterate_batches(dataset, tokenizer, batch_size, **kwargs):
        cpu_stream = mx.new_stream(mx.cpu)
        buffer = deque()
        base_iter = base_iterate_batches(dataset, tokenizer, batch_size, **kwargs)
        # Fill buffer with async_eval'd batches on cpu_stream
        # Yield from buffer, refilling one-ahead
        ...
    return prefetch_iterate_batches
```

MLX automatically inserts cross-stream barriers when GPU ops consume CPU-produced arrays.

**New config field**: `TrainConfig.prefetch_depth` (default 0 = disabled).

**Expected improvement**: 0.5-4% throughput gain per step (batch prep is 0.5-2ms vs 50-200ms forward/backward). Scales up for larger batch sizes or on-the-fly tokenization.

### Sub-project B: MoE Routing on CPU Stream

The MoE pattern runs gating + expert computation sequentially on GPU. The gate linear layer is a small projection `(B*T, D) -> (B*T, num_experts)` followed by top-k + softmax. This can run on CPU while the GPU runs shared experts in parallel.

**Key constraint**: Custom Metal kernels (`topk_gating_softmax`) cannot run on `mx.cpu`. The CPU-stream path uses standard MLX operations (`mx.softmax`, `mx.argpartition`, `mx.take_along_axis`).

**Overlap opportunity** (for models with shared experts like GLM-4, DeepSeek-V3):

```
CPU Stream              GPU Stream
----------              ----------
gate_fn(x)              shared_experts(x)  [concurrent]
topk_gating             (waiting for indices)
  |                     switch_mlp(x, indices)
  v                     moe_combine + shared_out
```

**New config flag**: `PatchConfig.moe_cpu_gating` (default `False`).

### Deliverables

1. `src/zmlx/train/prefetch.py` -- Prefetch iterator wrapper
2. `TrainConfig.prefetch_depth` field
3. `_gating_cpu()` and `_topk_gating_cpu_fallback()` in MoE pattern
4. Gradient correctness tests for CPU-gating backward pass
5. Benchmarks on Qwen3-30B-A3B (shared experts)

### Acceptance Criteria

- Prefetch yields identical batches to base iterator (bit-exact)
- `prefetch_depth=0` disables prefetching (passthrough)
- CPU-gating produces identical routing decisions within float32 tolerance
- No regression on models without shared experts

### References

- [MLX Using Streams](https://ml-explore.github.io/mlx/build/html/usage/using_streams.html)
- [MLX Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [Writing Fast MLX (Awni Hannun)](https://gist.github.com/awni/4beb1f7dfefc6f9426f3a7deee74af50)
- [WWDC 2025 -- Get started with MLX](https://developer.apple.com/videos/play/wwdc2025/315/)

---

## 6. Fused Dequantize + Compute

### Motivation

LLM inference on Apple silicon is memory-bandwidth-bound. The current ZMLX quant kernels dequantize to a full-precision intermediate tensor before the consumer op reads it -- doubling memory traffic. Fusing dequantization into consumer ops eliminates this.

### Current State

ZMLX's existing dequant kernels (`src/zmlx/kernels/quant.py`) use an ad-hoc format (raw int8/uint8 with scalar or blockwise scale). They do **not** read MLX's native `mx.quantize` format, which is what every `mlx-lm` model actually uses.

**MLX native quantization format**:

```
weight  = uint32[(out_dim, in_dim * bits / 32)]   # packed nibbles/bytes
scales  = float32[(out_dim, in_dim / group_size)]
biases  = float32[(out_dim, in_dim / group_size)]
Formula: w_i = scale * q_i + bias  (per group)
```

### Priority Analysis

| Fusion | Impact | Recommendation |
|--------|--------|---------------|
| **dequant + activation** (silu, gelu) | High -- eliminates fp16 materialization between QuantizedLinear and SiLU/GeLU | **Phase 1** |
| **dequant + RMSNorm** | Medium -- saves one read/write after attention output | **Phase 2** |
| **dequant + matmul** | Low -- `mx.quantized_matmul` already fuses this | **Do not pursue** |
| **dequant + SwiGLU (gate+up fused)** | High for dense MLP | **Phase 1b** |

### Implementation Plan

#### Phase 1: MLX-Format Dequant + Activation Kernels

New MSL helpers in `src/zmlx/msl.py` for reading MLX packed uint32 format:

```metal
inline float kk_dequant4(
    const device uint32_t* packed,
    const device float* scales,
    const device float* biases,
    uint row, uint elem, uint in_dim, uint group_size
) {
    uint word_idx = elem / 8;
    uint sub_idx  = elem % 8;
    uint32_t word = packed[row * (in_dim / 8) + word_idx];
    uint8_t q = (word >> (sub_idx * 4)) & 0xF;
    uint group_idx = elem / group_size;
    float s = scales[row * (in_dim / group_size) + group_idx];
    float b = biases[row * (in_dim / group_size) + group_idx];
    return s * (float)q + b;
}
```

New kernels: `dequantize_mlx`, `dequantize_silu_mlx`, `dequantize_gelu_mlx`.

#### Phase 1b: Patch Pattern for Quantized SwiGLU MLP

New `quant_swiglu_mlp` pattern detects `QuantizedLinear` gate/up projections and fuses the activation step. The `mx.quantized_matmul` call is preserved (already fast); the win is avoiding separate materialization of gate and up outputs before the `silu(gate) * up` fusion.

#### Phase 2: Codegen Template

`elementwise_dequant_unary_source` in `src/zmlx/codegen.py` -- generates dequant-fused kernel bodies from C expressions for easy creation: `dequant_unary("kk_silu(x)", bits=4, group_size=64)`.

#### Phase 3: Fused Dequant + Norm

`dequant_rmsnorm_mlx` -- reads quantized projection output, applies RMSNorm in one kernel.

### What NOT to Build

- **Custom quantized GEMM** -- `mx.quantized_matmul` is MPS-accelerated and unbeatable from `mx.fast.metal_kernel`
- **Custom backward for quantized ops** -- quantized weights are frozen in inference
- **New quant formats** -- target MLX-native format only

### Acceptance Criteria

- Bitwise agreement with `mx.dequantize` for plain dequant
- Agreement within `atol=1e-5, rtol=1e-3` for fused activation variants
- Tests cover bits=4 and bits=8, group_size=32/64/128
- `smart_patch` shows neutral-to-positive speedup on Qwen3-30B-A3B and Llama-3.1-8B (quantized)

---

## 7. Paged KV Cache with UMA-Aware Scheduling

### Motivation

vLLM's PagedAttention eliminates KV cache fragmentation by storing key/value tensors in fixed-size blocks with block-table indirection. On Apple Silicon's unified memory, this pattern gains unique advantages: zero-copy CPU/GPU access to page tables, no DMA transfers for eviction/swap, and concurrent metadata updates during GPU attention.

### Architecture

Three components under `src/zmlx/serving/`:

#### Component 1: PagePool

Pre-allocates a contiguous buffer for all KV cache blocks:

```python
class PagePool:
    def __init__(self, num_blocks, block_size, n_kv_heads, head_dim, dtype):
        self.k_pool = mx.zeros((num_blocks, block_size, n_kv_heads, head_dim), dtype=dtype)
        self.v_pool = mx.zeros((num_blocks, block_size, n_kv_heads, head_dim), dtype=dtype)
```

Memory budget formula: `2 * num_blocks * block_size * n_kv_heads * head_dim * dtype_bytes`. Example: 4096 blocks * 16 tokens/block * 8 heads * 128 dim * 2 bytes (fp16) * 2 (K+V) = 2 GB.

#### Component 2: BlockAllocator

O(1) allocate/free via doubly-linked free list with LRU eviction. Pre-allocates all `PhysicalBlock` metadata objects at init to avoid GC pressure during decode.

#### Component 3: KVCacheManager

Top-level orchestrator managing sequence lifecycle:

```python
mgr = KVCacheManager(num_blocks=4096, block_size=16, n_kv_heads=8, head_dim=128)
seq_a = mgr.register_sequence(max_context=2048)
mgr.allocate_slot(seq_a)
mgr.append_token(seq_a)
k_cache, v_cache, block_table, ctx_lens = mgr.build_kernel_args([seq_a])
```

Integrates with existing `paged_attention` and `paged_rope_and_cache_update` kernels without modifying their Metal source.

### UMA-Specific Advantages

| Aspect | CUDA (vLLM) | Apple Silicon (ZMLX) |
|--------|-------------|---------------------|
| Block table updates | CPU writes + `cudaMemcpyAsync` to GPU | CPU writes directly; GPU reads same array |
| Page swap (CPU<->GPU) | Explicit copy over PCIe (12-32 GB/s) | No-op; same physical address |
| Eviction cost | Copy to CPU DRAM before freeing GPU block | Just update metadata |
| Defragmentation | GPU-to-GPU copy or CPU staging | In-place swap (same bus) |

### Implementation Plan

| Phase | Deliverable | Acceptance |
|-------|------------|------------|
| 1 | Core page management (PagePool, BlockAllocator, KVCacheManager) | O(1) alloc/free; lifecycle test with 100 sequences; output matches non-paged reference within atol=1e-3 |
| 2 | Eviction + memory pressure | LRU eviction of finished sequences; preemption when pool full; pressure detection via Metal budget heuristic |
| 3 | Defragmentation + UMA optimizations | Metadata-only compaction; dedicated Metal swap kernel (< 10ms for 1000 blocks); CPU-stream RoPE precomputation |
| 4 | Dynamic max_context | Remove hardcoded max_context=4096; tiled fallback for > 16K context |
| 5 | Documentation + benchmarks | `docs/SERVING.md`; paged throughput within 5% of contiguous for single-sequence; > 2x batch throughput at high utilization |

### References

- [PagedAttention paper (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
- [vLLM Paged Attention Design](https://docs.vllm.ai/en/stable/design/paged_attention/)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.7.0/design/automatic_prefix_caching.html)
- [Native LLM Inference at Scale on Apple Silicon](https://arxiv.org/html/2601.19139)
- [MLX Unified Memory Documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)

---

## 8. Micro-Benchmark Driven Device Scheduling

### Motivation

Not every operation benefits from GPU execution. On Apple Silicon, small tensor operations (embedding lookups, MoE gating, single-token norms) can be faster on CPU due to GPU kernel launch overhead. By profiling each submodule on CPU vs GPU and routing accordingly, we can improve end-to-end throughput.

### Architecture

New `src/zmlx/schedule/` package with four modules:

#### DeviceProfiler

Hooks into `nn.Module.__call__` to capture inputs, then times each submodule on both `mx.cpu` and `mx.gpu` streams:

```python
profiler = DeviceProfiler(warmup=5, iters=20)
result = profiler.profile(model, sample)
print(result.summary())
# model.embed_tokens        CPU  0.012ms  GPU  0.089ms  -> CPU 7.4x faster
# layers[0].self_attn.norm  CPU  0.008ms  GPU  0.041ms  -> CPU 5.1x faster
# layers[0].mlp.gate_proj   CPU  3.200ms  GPU  0.180ms  -> GPU 17.8x faster
```

#### DevicePlacementPolicy

Combines three signals to decide CPU vs GPU per submodule:

1. **Empirical profiling** -- CPU must be >= threshold% faster (default 10%)
2. **Op-type heuristics** -- Embedding, RoPE (decode), scalar ops are CPU-favorable; Linear, QuantizedLinear, Attention are GPU-only
3. **Shape-based thresholds** -- Elementwise: CPU wins below ~4K elements; reductions: below ~2K; gather: below ~8K. Scaled by GPU core count (more cores = lower per-element overhead)

```python
policy = DevicePlacementPolicy(result, threshold_pct=10.0)
plan = policy.decide(model)
print(plan.summary())
```

Rules enforce: fused Metal kernels (anything with `_zmlx_original_call`) never route to CPU. `Linear`, `QuantizedLinear`, and `Attention` modules never route to CPU.

#### Stream Wrapping

`apply_placement()` wraps CPU-routed submodule `__call__` methods with `mx.stream(mx.cpu)`. Reversible via `remove_placement()`.

#### Integration with smart_patch

Device scheduling runs as a post-fusion optimization pass inside `smart_patch()`:

```python
model = zmlx.patch.smart_patch(
    model, sample,
    device_schedule=True,       # Enable CPU/GPU profiling
    device_threshold_pct=10.0,  # CPU must be >= 10% faster
    verbose=True,
)
```

Profiling happens **after** kernel fusion patterns are applied (since fusion changes the computation graph). The end-to-end benchmark gate ensures placement is only kept if it improves wall-clock time.

### Persistent Cache

Placement decisions cached per `(device_family, mlx_version, model_id)` in `~/.cache/zmlx/device_schedule_v1.json`, following the same pattern as the autotune cache.

### Acceptance Criteria

- `apply_placement()` + `remove_placement()` round-trips cleanly (bitwise identical output)
- Profiler timings reproducible within 5% CoV across 3 runs
- `smart_patch(device_schedule=True)` never slower than `smart_patch(device_schedule=False)` (gated by end-to-end benchmark)
- Measurable improvement (>= 3%) on at least one of: Qwen3-30B-A3B decode (batch=1), Llama-3.2-1B decode (batch=1), or any model's single-token prefill
- Fused Metal kernels never routed to CPU
- Cache save/load round-trips correctly

### References

- [MLX Unified Memory Documentation](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- [mlx-benchmark: CPU vs GPU timing data](https://github.com/TristanBilot/mlx-benchmark)
- [Benchmarking On-Device ML on Apple Silicon (arXiv:2510.18921)](https://arxiv.org/abs/2510.18921)
- [MLX GPU kernel launch overhead issue #1828](https://github.com/ml-explore/mlx/issues/1828)

---

## Summary of New Files

| File | Roadmap Item | Purpose |
|------|-------------|---------|
| `src/zmlx/device.py` (modified) | 1 | DeviceTuningProfile, baked-in profiles, GPU core fix |
| `src/zmlx/autotune.py` (modified) | 1 | @autotune() decorator, profile-guided candidates, cache v3 |
| `tests/conftest.py` (modified) | 2 | Marker-based test classification |
| `src/zmlx/__init__.py` (modified) | 2 | Relaxed import guard for portable modules |
| `src/zmlx/_compat.py` (modified) | 2 | `detect_backend()` function |
| `.github/workflows/ci.yml` (modified) | 2 | Multi-backend CI matrix |
| `tests/test_cross_backend.py` | 2 | Golden values cross-backend tests |
| `tests/test_gpu_generation.py` | 2 | GPU-generation fingerprinting |
| `src/zmlx/patch/_tracer.py` | 3 | SubmoduleTracer |
| `src/zmlx/patch/_fusion_table.py` | 3 | Declarative fusion table |
| `src/zmlx/patch/_synthesize.py` | 3 | Auto-generated PatchPattern instances |
| `src/zmlx/patch/_discovery.py` | 3 | discover_patterns(), auto_patch() |
| `src/zmlx/kernels/attention.py` (modified) | 4 | flash_attention(), paged_flash_attention() |
| `src/zmlx/train/prefetch.py` | 5 | Prefetch iterator wrapper |
| `src/zmlx/train/config.py` (modified) | 5 | prefetch_depth field |
| `src/zmlx/patch/patterns/moe_mlp.py` (modified) | 5 | CPU-stream gating |
| `src/zmlx/msl.py` (modified) | 6 | DEQUANT_HEADER with kk_dequant4/8 |
| `src/zmlx/kernels/quant.py` (modified) | 6 | MLX-format dequant kernels |
| `src/zmlx/codegen.py` (modified) | 6 | elementwise_dequant_unary_source |
| `src/zmlx/patch/patterns/quant_swiglu_mlp.py` | 6 | Quantized SwiGLU pattern |
| `src/zmlx/serving/page_pool.py` | 7 | PagePool |
| `src/zmlx/serving/block_allocator.py` | 7 | BlockAllocator with free list + LRU |
| `src/zmlx/serving/kv_cache_manager.py` | 7 | KVCacheManager orchestrator |
| `src/zmlx/schedule/profiler.py` | 8 | DeviceProfiler |
| `src/zmlx/schedule/policy.py` | 8 | DevicePlacementPolicy |
| `src/zmlx/schedule/apply.py` | 8 | Stream wrapping, apply/remove placement |
| `src/zmlx/schedule/cache.py` | 8 | Persistent placement cache |
