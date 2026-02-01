# Experimental MLX Fork (Optional)

This document describes ZMLX’s **optional** custom‑MLX work. You do **not** need any of this for the stable, stock‑MLX results in the README. It exists for research and upstream prototyping only.

## What this is

ZMLX includes a local MLX fork in `mlx_local/` for experiments that require MLX internals (quantized matmul kernels, fused projections). These are intended for eventual upstream contribution.

## When to use it

Only if you are experimenting with C++ Metal primitives or trying to fuse operations that MLX doesn’t expose in Python. The stable results for LFM2, Qwen3, and GPT‑OSS do **not** rely on this.

## Key primitive: `gather_qmm_swiglu`

This is a C++ Metal primitive that fuses **gate projection + up projection + SwiGLU** for quantized MoE experts into a single kernel launch. It can save additional dispatches, but its precision currently differs from stock MLX for some models (notably Qwen3). Until that is fixed, it is not enabled by default.

## Known issue (Qwen3)

Qwen3 diverges when `gather_qmm_swiglu` is forced. The likely cause is **precision mismatch** (float16 accumulation) in the custom kernel vs MLX’s default path. The fix would be float32 accumulation inside the kernel. Until then, this path remains experimental.

## Experimental benchmarks (historical)

Qwen3‑30B‑A3B (max_tokens=500, runs=3):

| Config | Decode speedup | Fidelity | Notes |
|:--|:--|:--|:--|
| Dev MLX + `moe_mlp` forced | +6.9% (base) / +8.9% (instruct) | FAIL | Diverges at token 6 (base) / token 146 (instruct) |

## Prototype primitives

| Primitive | Status | Description |
|:--|:--|:--|
| `gather_qmm_swiglu` | Working | Fused gate+up+SwiGLU for quantized MoE experts |
| `gather_qmm_combine` | Working | Fused down projection + weighted expert sum |
| `add_rms_norm` | Planned | Fused residual add + RMSNorm |

## Upstream plan

See [`UPSTREAM_PLAN.md`](../UPSTREAM_PLAN.md) for what is intended to be upstreamed to MLX.
