# Development Guide

## Project Status (v0.7.12)

- 70+ Metal kernel catalog (activations, norms, RoPE, attention, MoE, quantization, loss, scan)
- Model patching: `patch(model)` with model-aware defaults, validated on LFM2, Qwen3, GPT-OSS
- Fused MoE inference: +5-12% decode on LFM2-8B-A1B, +7% on Qwen3-30B-A3B, +1% on GPT-OSS-20B (token-identical, prefill neutral)
- Benchmark infrastructure: repro capsules, `bench.report` CLI, `validate` CLI
- Experimental MLX fork lives in `docs/EXPERIMENTAL_MLX.md` (optional, not required for stable results)

## Development Areas

### Kernel authoring
- Expand `codegen.py` patterns (broadcasting, 2D launches, multiple outputs)
- Improve error messages for shape/dtype mismatches and non-contiguous inputs

### Autograd
- `jvp` support for elementwise ops
- Higher-order derivatives (if MLX supports nested `custom_function`)

### Kernel library
- Flash Attention with threadgroup shared memory for small tiles
- Fused dequant+compute (int4 dequant fused with matmul or activation)

### Performance
- Per-device autotune profiles (M1/M2/M3/M4 families)
- Regression tracking with JSON repro capsules
- Memory bandwidth analysis
- Experimental MLX work is documented in `docs/EXPERIMENTAL_MLX.md`.

### Release policy
- **Stable (default)**: stock MLX, only token‑identical patches enabled.
- **Experimental (opt‑in)**: custom MLX allowed; kernels are opt‑in and must be validated.

### Upstream contributions
- `mx.fast.swiglu` fused primitive (small, general‑purpose)
- `add_rms_norm` fused primitive (benchmark‑gated)
- `gather_qmm_swiglu` (experimental; RFC before any PR)

## Backlog

### High Priority
- [ ] Fused dequant+compute (int4 dequant fused into activation)
- [ ] Per-device autotune profiles
- [ ] Upstream `mx.fast.swiglu` to MLX

### Medium Priority
- [ ] `jvp` support for elementwise ops
- [ ] More architecture support (Gemma, Phi, Mistral MoE)
- [ ] `add_rms_norm` fused primitive (if MLX doesn't already fuse add+norm)

### Low Priority
- [ ] Flash Attention with shared memory tiles
- [ ] Higher-order derivatives
- [ ] Speculative decoding acceleration
