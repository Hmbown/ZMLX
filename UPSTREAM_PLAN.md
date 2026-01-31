# Upstream Plan

What belongs in MLX, what stays in ZMLX, and how to get there.

## Scope

ZMLX sits on top of MLX's `mx.fast.metal_kernel` and `mx.custom_function` APIs. Most of ZMLX (kernel authoring, autograd wrappers, the catalog) is a toolkit that doesn't need to be upstream. The fused C++ Metal primitives in `mlx_local/` are the upstream candidates — they need access to MLX internals that `metal_kernel` can't reach.

## What belongs upstream

| Primitive | Status | Why it needs to be in MLX |
|:--|:--|:--|
| `gather_qmm_swiglu` | Working (local fork) | Fused gather + quantized matmul + SwiGLU for MoE experts. Reads `x` once instead of twice, requires access to MLX's quantized matmul internals. |
| `add_rms_norm` | Planned | Fused residual add + RMSNorm. Needs to write the residual and the normalized output in a single pass (2x memory bandwidth reduction). |
| `gather_qmm_combine` | Planned | Fused down-projection + weighted expert sum. Eliminates the intermediate expert output buffer. |

### Smallest safe PR: `gather_qmm_swiglu`

**What it does:** Given a quantized weight matrix W_gate and W_up (packed as a SwitchLinear), and a set of expert indices per token, computes `SwiGLU(gather(W_gate, idx) @ x, gather(W_up, idx) @ x)` in a single Metal kernel dispatch.

**Why it helps:** In MoE decode (M=1), the standard path dispatches 3+ kernels (gather, two QMMs, SwiGLU). The fused version does one dispatch, reducing Metal command buffer overhead and reading the input once.

**Validation:**
- Token-identical output vs unfused path (greedy decode, 500 tokens)
- Tested on LFM2-8B-A1B-4bit and 8bit (E=32, K=4)
- Guarded: only activates for M <= 32 (decode). Prefill falls through to the standard path.
- Measured: +4% per MoE layer at M=1 on M1 Pro, +9-12% end-to-end decode on M4 Max

**PR structure:**
1. C++ primitive + Metal kernel in `mlx/backend/metal/kernels/`
2. Python binding in `mlx/fast.py`
3. Tests: correctness vs reference implementation, shape/dtype coverage
4. Benchmark: single-layer timing at M=1,4,16,64

## What stays in ZMLX

- **Kernel authoring API** (`elementwise()`, `reduce()`, `map_reduce()`) — developer tooling, not a runtime primitive.
- **Autograd wrappers** — convenience layer over `mx.custom_function`.
- **Kernel catalog** (70+ kernels) — reference implementations and benchmarks. Some may eventually inform upstream optimizations, but they're useful as-is for prototyping.
- **Model patching** (`patch()`, `smart_patch()`) — ZMLX-specific UX for applying fused kernels to existing models.
- **Autotune** — threadgroup search that works with `metal_kernel`. Could become upstream if MLX adds autotuning.

## Validation approach

Before any upstream PR:

1. **Correctness**: `python -m zmlx.validate <model> --max-tokens 500 --runs 5` must report `PASS` (token-identical greedy decode).
2. **Performance**: Single-layer benchmarks (`benchmarks/bench_moe_layer.py`) must show measurable improvement at decode sequence lengths (M=1 to M=32).
3. **Regression guard**: Prefill (M > 32) must not regress. The fused path is guarded with a sequence length threshold.
4. **Repro capsule**: Raw per-run data saved to `benchmarks/repro_capsules/` with hardware, OS, and version metadata.

## Timeline

| Step | Target |
|:--|:--|
| `gather_qmm_swiglu` local validation complete | Done |
| Open MLX GitHub Discussion with motivation + benchmarks | Next |
| Submit PR with primitive + tests + benchmark | After discussion feedback |
| `add_rms_norm` local prototype | After `gather_qmm_swiglu` lands |
