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

### Quantized KV cache (opt-in)

Use quantized KV cache to reduce decode bandwidth on dense models. This trades a
small amount of accuracy for speed and memory savings, so keep it opt-in.

```bash
ZMLX_KV_BITS=8 ZMLX_KV_GROUP_SIZE=64 python -m zmlx.validate <model> --max-tokens 200 --runs 3
```

Equivalent CLI flags:

```bash
python -m zmlx.validate <model> --max-tokens 200 --runs 3 --kv-bits 8 --kv-group-size 64 --quantized-kv-start 0
```

### Stream pool experiments (list-of-experts MoE)

Use `benchmarks/bench_moe_streams.py` to compare baseline vs patched across stream counts.
This is intended for list-of-experts models (Mixtral/DeepSeek). SwitchGLU models ignore the
stream pool.

```bash
python benchmarks/bench_moe_streams.py \
  --model mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit \
  --streams 1,2,4,8 \
  --runs 5 \
  --max-tokens 500 \
  --json-out benchmarks/repro_capsules/mixtral_streams.json
```

Optional: experiment with alternative stream reductions (non-default, may change rounding):

```bash
ZMLX_MOE_STREAMS_REDUCE=tree python benchmarks/bench_moe_streams.py \
  --model mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit \
  --streams 2,4,8 \
  --runs 3 \
  --max-tokens 200
```

## Why prefill is neutral

All fused kernels are guarded with a sequence length check (`M <= 32`). During prefill, M equals the prompt length (typically hundreds or thousands of tokens). At this scale, the compute-to-dispatch ratio is high and the standard MLX path is already efficient. The guards ensure ZMLX never regresses prefill performance.

## Op-level microbenchmarks

B=16, S=1024, D=1024, float16, M4 Max. Run with `python benchmarks/microbench.py`.

These measure individual kernel performance in isolation, not end-to-end model inference. End-to-end gains are smaller because kernels compete with MLX's lazy evaluation and command buffer batching.

---

## Development run log (2026-02-02)

### Environment

- Hardware: Apple M4 Max (36 GB)
- OS: macOS 26.1
- Python: 3.14.2
- ZMLX commit: `11b9310dd1c79bff52253caec8faf1035c937548`
- MLX: 0.30.4 (stock), and custom MLX build from `mlx_local` (commit `2f324cc3b200700b422db4811ae3ff8bd5bf48b4`, local modifications present)

### Commands (placeholders)

```bash
export PYTHONPATH=<REPO_ROOT>/src:$PYTHONPATH
python3 -m zmlx.validate <model> --max-tokens 1000 --runs 15
```

```bash
export PYTHONPATH=<REPO_ROOT>/mlx_local/python:<REPO_ROOT>/src:$PYTHONPATH
python3 -m zmlx.validate <model> --max-tokens 1000 --runs 15
```

### Results (1000 tokens, 15 runs, greedy)

**Stock MLX (ZMLX only)**

| Model | Base prompt | Patched prompt | Base decode | Patched decode | Decode speedup | Fidelity |
|:--|--:|--:|--:|--:|--:|:--|
| LFM2-8B-A1B-4bit | 742.8 tok/s | 756.7 tok/s | 223.5 tok/s | 249.4 tok/s | 1.116x | PASS |
| LFM2-8B-A1B-8bit-MLX | 555.3 tok/s | 552.1 tok/s | 151.8 tok/s | 162.5 tok/s | 1.071x | PASS |
| GPT-OSS-20B-MXFP4-Q4 | 320.1 tok/s | 317.9 tok/s | 121.8 tok/s | 122.9 tok/s | 1.008x | PASS |

**Custom MLX kernel (optional)**

| Model | Base prompt | Patched prompt | Base decode | Patched decode | Decode speedup | Fidelity |
|:--|--:|--:|--:|--:|--:|:--|
| Qwen3-30B-A3B-Instruct-2507-4bit | 331.0 tok/s | 332.3 tok/s | 108.0 tok/s | 116.5 tok/s | 1.078x | PASS |

### Notes

- Quick sanity after adding fused down‑projection + combine (200 tokens, 3 runs, custom MLX): Qwen3‑30B‑A3B‑Instruct‑2507‑4bit **1.042x**, PASS. This is not directly comparable to the 1000‑token/15‑run baseline and will be re‑run.
- Qwen3 minimal profile (`--patch-profile qwen3`, 1000 tokens, 15 runs, custom MLX): base 331.7 / 109.4 → patched 333.4 / 116.3, **1.063x**, PASS. This is below the 1.078x baseline and needs further tuning before adoption.
- Qwen3 default profile (custom MLX, 1000 tokens, 15 runs): base 333.2 / 111.2 → patched 333.3 / 116.8, **1.050x**, PASS. This is below the earlier 1.078x result and may indicate regression or run‑to‑run variance.
- LFM2‑8B‑A1B‑4bit (stock MLX, 1000 tokens, 15 runs) after fused down‑proj+combine: base 738.5 / 221.1 → patched 739.3 / 245.6, **1.111x**, PASS (no regression vs prior).
- GPT‑OSS‑20B‑MXFP4‑Q4 (stock MLX, 1000 tokens, 15 runs) after fused down‑proj+combine: base 319.9 / 121.1 → patched 319.1 / 122.1, **1.008x**, PASS (no regression vs prior).
- Multi‑queue overlap microbench (custom MLX) with `MLX_MAX_OPS_PER_BUFFER=1` shows **~1.01–1.02x**, indicating minimal overlap under current MLX eval/commit policy.
