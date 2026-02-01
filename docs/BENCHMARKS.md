# Benchmark Methodology

## Standard benchmark protocol

All benchmarks use `python -m zmlx.validate` with the following defaults:

- **Decoding:** greedy (`temp=0`), fixed prompt, median of N runs
- **Baseline:** unpatched `mlx_lm` model
- **Patched:** `patch(model)` with default settings (stable mode)
- **Fidelity:** every generated token ID compared between patched and unpatched

### Repro capsules

Each benchmark result is backed by a repro capsule in `benchmarks/repro_capsules/` containing:

- Hardware (chip, memory, GPU cores)
- OS version and MLX version
- ZMLX version and git commit
- Raw per-run timing data (all runs, not just median)
- Model ID and quantization config

Print a formatted report from any capsule:

```bash
python -m zmlx.bench.report benchmarks/repro_capsules/<capsule>.json
```

### Running your own benchmarks

```bash
# Standard validation (5 runs, 500 tokens)
python -m zmlx.validate <model> --max-tokens 500 --runs 5

# Quick check (3 runs, 200 tokens)
python -m zmlx.validate <model> --max-tokens 200 --runs 3
```

## Why prefill is neutral

All fused kernels are guarded with a sequence length check (`M <= 32`). During prefill, M equals the prompt length (typically hundreds or thousands of tokens). At this scale, the compute-to-dispatch ratio is high and the standard MLX path is already efficient. The guards ensure ZMLX never regresses prefill performance.

## Op-level microbenchmarks

B=16, S=1024, D=1024, float16, M4 Max. Run with `python benchmarks/microbench.py`.

These measure individual kernel performance in isolation, not end-to-end model inference. End-to-end gains are smaller because kernels compete with MLX's lazy evaluation and command buffer batching.
