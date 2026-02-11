# Upstream Plan

What belongs in MLX, what stays in ZMLX, and how to get there.

## Scope

ZMLX sits on top of MLX's `mx.fast.metal_kernel` and `mx.custom_function` APIs. Most of ZMLX (kernel authoring, autograd wrappers, the catalog) is a toolkit that doesn't need to be upstream. The fused C++ Metal primitives in `mlx_local/` are the upstream candidates — they need access to MLX internals that `metal_kernel` can't reach.

## Principles

- Prefer general-purpose primitives that benefit many model families.
- Keep MoE‑specific fusions in ZMLX unless MLX maintainers request them.
- Default user path must remain token‑identical (performance without silent drift).

## Upstream candidates (general‑purpose first)

| Candidate | Status | Why it belongs in MLX |
|:--|:--|:--|
| `mx.fast.swiglu` | Proposed | Common transformer activation (Llama/Qwen/Mistral/Gemma/Phi). Small, clean primitive that mirrors `mx.fast.*` patterns. |
| `add_rms_norm` | Benchmark‑gated | Only useful if MLX doesn’t already fuse `x + residual` into `rms_norm`. |
| `gather_qmm_swiglu` | Local fork (RFC first) | MoE‑specific, uses MLX quantized matmul internals. Needs maintainers’ buy‑in before a PR. |

### Minimal PR: `mx.fast.swiglu`

**What it does:** Computes `silu(gate) * up` in a single pass with a small elementwise Metal kernel.

**Why it helps:** Every modern transformer uses SwiGLU. A fused op reduces dispatch overhead and intermediate buffers and fits MLX’s existing `mx.fast` API pattern.

**PR structure:**
1. Declare `swiglu()` in `mlx/fast.h`
2. Add primitive + VJP (patterned on RMSNorm)
3. Implement a tiny Metal kernel (single‑pass elementwise)
4. Add Python binding
5. Tests: forward correctness + VJP + CPU/Metal parity

### Benchmark gate: `add_rms_norm`

Before proposing a primitive, benchmark whether MLX already fuses:

```
mx.fast.rms_norm(x + residual, w, eps)
```

If MLX already fuses this into a single dispatch, an `add_rms_norm` primitive adds complexity without benefit. If it does not, a fused add+norm can halve memory bandwidth at every layer.

### MoE‑specific fusions

- **`gather_qmm_swiglu`**: keep in `mlx_local/` until an MLX Discussion confirms interest. Share LFM2 and Qwen3 repro capsules to justify it.
- **`gather_qmm_combine`**: stays in ZMLX (achievable via `metal_kernel` and too specialized for core MLX).

## Release policy (ZMLX)

- **Stable (default)**: stock MLX, only token‑identical patches enabled.
- **Fast (opt‑in)**: dev MLX allowed; experimental kernels are opt‑in and must be validated.
- **Edge (opt‑in)**: nightly/dev MLX + experimental kernels for local testing only.

ZMLX should auto‑detect dev MLX features (e.g. `mx.gather_qmm_swiglu`) and only enable them when present.

## Roadmap (sequenced)

1. **Dev‑MLX delivery**: publish a reproducible dev‑MLX build path (wheel or source) and a runtime capability check.
2. **Qwen3‑30B‑A3B (base + instruct)**: run token‑parity + perf on stock MLX vs dev MLX and record results in capsules.
3. **Capsules + docs**: update README/CLAUDE with release policy and dev‑MLX behavior.
4. **Upstream `mx.fast.swiglu`**: submit the small, general PR once tests are ready.
5. **`add_rms_norm`**: submit only if the benchmark shows no existing fusion.
6. **MoE RFC**: open an MLX Discussion for `gather_qmm_swiglu` before any PR.
