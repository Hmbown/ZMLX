# Tour

A quick walkthrough of what ZMLX does and how to verify it.

## 1. Install

```bash
pip install zmlx
```

Requires macOS on Apple Silicon, Python >= 3.10, MLX >= 0.30.0.

## 2. Patch a model

```python
import mlx_lm
from zmlx.patch import patch

model, tokenizer = mlx_lm.load("mlx-community/LFM2-8B-A1B-4bit")
patch(model)
```

`patch()` walks the model tree, detects the architecture, and replaces matching layers with fused Metal kernel equivalents. No weights are modified.

```
[zmlx.patch] Applying 3 patterns: ['swiglu_mlp', 'geglu_mlp', 'moe_mlp']
Patched 24 modules:
  moe_mlp: 22
  swiglu_mlp: 2
```

## 3. Validate correctness

```bash
python -m zmlx.validate mlx-community/LFM2-8B-A1B-4bit --max-tokens 500 --runs 5
```

This loads the model twice (baseline then patched), generates tokens with greedy decode (`temp=0`), and compares output token-for-token. A `PASS` verdict means every token is identical.

## 4. Read a benchmark report

```bash
python -m zmlx.bench.report benchmarks/repro_capsules/lfm2_m1pro_20260131.json
```

Prints the full results from a saved repro capsule, including environment, methodology, and per-model metrics. The JSON capsule contains raw per-run data for independent verification.

## 5. Where the fused primitive lives

The MoE speedup comes from two things:

**Fused gating** (`src/zmlx/kernels/moe.py`): The standard path computes top-k softmax with multiple MLX ops (softmax, argpartition, gather, normalize). The fused kernel does this in a single Metal dispatch for small sequence lengths.

**Fused expert combine** (`src/zmlx/kernels/moe.py`): After experts compute their outputs, the standard path does element-wise multiply + sum across experts. The fused kernel does this in one pass.

**Sequence length guard** (`src/zmlx/patch/patterns/moe_mlp.py`): The fused kernels only activate when the sequence length M <= 32 (decode). At larger M (prefill), the standard MLX path is used. This is why prefill throughput is neutral — it's the same code path.

```python
_FUSED_SWIGLU_MAX_TOKENS = 32  # fused path for decode only
```

## 6. What "token-identical" means

ZMLX's correctness guarantee: given the same model weights, the same prompt, and greedy decoding (`temp=0`), the patched model produces the exact same token sequence as the unpatched model. This is verified by `python -m zmlx.validate`, which compares token IDs one-by-one.

When a pattern breaks fidelity on a model family (e.g. `moe_mlp` on Qwen3-MoE), `patch()` auto-excludes it. You can override with `patterns=[...]`, but validate first.
