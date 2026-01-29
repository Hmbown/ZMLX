# Upstreaming to MLX (practical notes)

MLX already supports:
- custom Metal kernels via `mx.fast.metal_kernel()`
- custom gradients via `mx.custom_function`

This repo focuses on **ergonomics** and **cookbook patterns**.

## Best upstream targets (high-likelihood)
1. **Docs / cookbook additions**
   - Add a section showing: kernel builder pattern + safe threadgroup defaults.
   - Add an example demonstrating `mx.custom_function` + a backward Metal kernel.

2. **A tiny helper**
   - A small helper in `mlx.core.fast` (or docs-only) that chooses threadgroup sizes.
   - Optional: a convenience wrapper that defaults `grid=(out.size,1,1)`.

3. **Tests for metal_kernel examples**
   - A minimal correctness test around a simple kernel (exp) and strided indexing.

## How to propose
1) Start with a GitHub Discussion describing the pain-point and the proposal.
2) Link to a small working implementation in this repo.
3) Offer an incremental PR:
   - small diff
   - adds tests
   - updates docs

## MLX contribution workflow (typical)
- Fork `ml-explore/mlx`
- Create a branch
- Make changes + add tests
- Run the MLX test suite
- Open a PR and describe motivation + benchmarks

See the upstream repo’s `CONTRIBUTING.md` for the authoritative checklist.

## A note on scope
If the change is “developer experience” (helper utilities, demos, tutorials), it’s often easiest to upstream docs + a tiny helper first, and keep the larger toolkit here as a community project.
