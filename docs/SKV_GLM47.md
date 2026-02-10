# SKV + GLM-4.7-Flash (MLA) Integration

This document tracks the **SKV-labeled** integration path in ZMLX.

## New SKV kernel APIs

All new kernels are in `zmlx.kernels.skv`:

- `skv_fused_project_quantize`
- `skv_fused_dequantize_unproject`
- `skv_compressed_attention`

## MLA helpers

`zmlx.kvtc.skv_mla` provides GLM-specific helpers:

- `skv_split_glm_keys` (`[kv_latent | k_rope]` split)
- `skv_compute_basis`
- `skv_compress_glm_latent`
- `skv_decompress_glm_latent`
- `skv_reconstruct_glm_keys`
- `skv_project_glm_queries_to_rank`
- `skv_glm_compressed_attention_scores`

RoPE component (`k_rope`) is intentionally uncompressed.

## Patch pattern (opt-in)

Pattern name: `skv_mla` in `zmlx.patch`.

Enable + run:

```bash
export ZMLX_SKV_MLA_ENABLE=1
export ZMLX_SKV_MLA_RANK=128
export ZMLX_SKV_MLA_BITS=4
export ZMLX_SKV_MLA_GROUP_SIZE=32
export ZMLX_SKV_MLA_WARMUP_TOKENS=128
export ZMLX_SKV_MLA_STRATEGY=B

python -m zmlx.validate mlx-community/GLM-4.7-Flash-4bit \
  --patterns skv_mla \
  --runs 1 \
  --max-tokens 64
```

Notes:
- This path is currently experimental and disabled by default.
- RoPE stays uncompressed by design.
- Strategy `A`: materialize dense K/V from compressed latent and run SDPA.
- Strategy `B` (default): decode uses compressed-space no-PE scores plus RoPE
  score add, then applies attention to dense latent values.
- Cache storage is persistent compressed latent + raw RoPE (dense latent is
  dropped after warmup basis calibration).

## Current Status

- Fidelity: passing short and long prompt checks (`16/16` and `32/32` tokens
  identical in `zmlx.validate` smoke runs).
- Cache payload reduction (prefill 1024 tokens, rank=96 bits=4): about `3.47x`
  versus dense MLA key cache storage.
- Decode throughput is still below baseline for long prompts in both strategies
  (`A` currently faster than `B` in this prototype). Further kernel fusion is
  still needed on the value path.
