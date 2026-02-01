# Development Guide

## Project Status (v0.7.11)

- 70+ Metal kernel catalog (activations, norms, RoPE, attention, MoE, quantization, loss, scan)
- Model patching: `patch(model)` with model-aware defaults, validated on LFM2
- Fused MoE inference: +5-12% decode on LFM2-8B-A1B (token-identical, prefill neutral)
- Optimization lab: `gather_qmm_swiglu` C++ Metal primitive (local MLX fork)
- Benchmark infrastructure: repro capsules, `bench.report` CLI, `validate` CLI
- Dev MLX workflow: use the local `mlx_local` fork when validating MoE gains (Qwen3 needs `mx.gather_qmm_swiglu` for speedups).

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
- Dev MLX env: `.venv-mlx-dev` with `pip install -e mlx_local` to expose `mx.gather_qmm_swiglu`
- Verify dev MLX: `python -c "import mlx.core as mx; print(hasattr(mx, 'gather_qmm_swiglu'))"`

### Release policy
- **Stable (default)**: stock MLX, only token‑identical patches enabled.
- **Fast (opt‑in)**: dev MLX allowed; experimental kernels are opt‑in and must be validated.
- **Edge (opt‑in)**: nightly/dev MLX + experimental kernels for local testing only.

### Upstream contributions
- `mx.fast.swiglu` fused primitive (small, general‑purpose)
- `add_rms_norm` fused primitive (benchmark‑gated)
- `gather_qmm_swiglu` (local fork; RFC before any PR)

## Backlog

### High Priority
- [ ] Fused dequant+compute (int4 dequant fused into activation)
- [ ] Per-device autotune profiles
- [ ] Upstream `mx.fast.swiglu` to MLX

### Medium Priority
- [ ] `jvp` support for elementwise ops
- [ ] Fix token fidelity on Qwen3-MoE (`moe_mlp` diverges at token 0)
- [ ] More architecture support (Gemma, Phi, Mistral MoE)
- [ ] `add_rms_norm` fused primitive (if MLX doesn't already fuse add+norm)

### Low Priority
- [ ] Flash Attention with shared memory tiles
- [ ] Higher-order derivatives
- [ ] Speculative decoding acceleration
