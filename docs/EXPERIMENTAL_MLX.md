# Experimental MLX Fork (Optional)

This document describes ZMLX’s **optional** custom‑MLX work. Most ZMLX patterns run on released MLX, but some MoE decode wins (notably GLM/Qwen3) currently depend on the fused primitive `mx.gather_qmm_swiglu`, which may not be exposed in your installed MLX build.

## What this is

ZMLX includes a local MLX fork in `mlx_local/` for experiments that require MLX internals (quantized matmul kernels, fused projections). These are intended for eventual upstream contribution.

## When to use it

- If you are experimenting with C++ Metal primitives or trying to fuse operations that MLX doesn’t expose in Python.
- If you want MoE fused decode for models where ZMLX relies on `mx.gather_qmm_swiglu` but your installed MLX build does not expose it (as of MLX 0.30.4/0.30.5 releases).

## Key primitive: `gather_qmm_swiglu`

This is a C++ Metal primitive that fuses **gate projection + up projection + SwiGLU** for quantized MoE experts into a single kernel launch. Some MLX dev builds expose this as `mx.gather_qmm_swiglu`; the `mlx_local/` fork is useful when you want to prototype changes to the primitive itself or add new primitives.

## Known issue (Qwen3)

Historical note: early experiments with custom fused MoE implementations diverged on Qwen3 due to precision differences vs MLX’s reference path. ZMLX now prefers `mx.gather_qmm_swiglu` when available and keeps MoE fused paths guarded behind small-token heuristics. Always validate token fidelity on your hardware.

## Experimental benchmarks (historical)

Qwen3‑30B‑A3B (max_tokens=500, runs=3):

| Config | Decode speedup | Fidelity | Notes |
|:--|:--|:--|:--|
| Early dev MLX + `moe_mlp` forced | +6.9% (base) / +8.9% (instruct) | FAIL | Diverged due to precision mismatch (since fixed) |

## Prototype primitives

| Primitive | Status | Description |
|:--|:--|:--|
| `gather_qmm_swiglu` | Working | Fused gate+up+SwiGLU for quantized MoE experts |
| `gather_qmm_combine` | Working | Fused down projection + weighted expert sum |
| `add_rms_norm` | Planned | Fused residual add + RMSNorm |

## Upstream plan

See [`UPSTREAM_PLAN.md`](../UPSTREAM_PLAN.md) for what is intended to be upstreamed to MLX.
