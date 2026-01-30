# ZMLX Benchmark Summary for Apple MLX Team

**Date:** January 30, 2025  
**Contact:** Hunter Bown  
**Prepared for:** Awni Hannun / Apple MLX Team  

---

## Executive Summary

Comprehensive benchmarks were run on **Qwen3-30B-A3B-Instruct** (4-bit quantized, MoE model) comparing ZMLX-patched vs unpatched MLX inference. The results show **neutral performance** with current MLX 0.30.0 — a significant change from earlier benchmarks showing 1.6x speedups.

### Key Findings

| Metric | Baseline MLX | ZMLX Patched | Speedup |
|:-------|:------------:|:------------:|:-------:|
| **Prompt TPS** | 1,090 tok/s | 1,096 tok/s | **1.01x** (+1%) |
| **Generation TPS** | 115 tok/s | 114 tok/s | **0.99x** (-1%) |
| **Peak Memory** | 17.55 GB | 17.55 GB | **No change** |
| **TTFT** | 0.323s | 0.322s | **-0.4ms** |

**Verdict:** Patches are **safe** (no regressions) but provide **neutral performance** on this configuration.

---

## Hardware & Software Configuration

| Component | Specification |
|:----------|:--------------|
| **Device** | Apple M4 Max |
| **Memory** | 36 GB unified |
| **OS** | macOS 15.1 (Darwin 25.1.0) |
| **MLX** | 0.30.0 |
| **mlx_lm** | 0.30.0 |
| **ZMLX** | 0.6.3 |
| **Python** | 3.14.2 |

---

## Detailed Results

### Test Configuration
- **Model:** mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
- **Model Type:** Mixture of Experts (MoE)
- **Specs:** 30B total params, 3B active, top-2 routing
- **Quantization:** 4-bit
- **Prompt Length:** ~306 tokens (technical prompt)
- **Generation:** 150 tokens
- **Runs:** 3 iterations (median reported)

### Raw Results

#### Baseline (unpatched mlx_lm)
```json
{
  "median_prompt_tps": 1090.44,
  "median_gen_tps": 115.15,
  "median_ttft_sec": 0.323,
  "peak_memory_gb": 17.55,
  "runs": [
    {"prompt_tps": 1090.44, "gen_tps": 115.19},
    {"prompt_tps": 1098.23, "gen_tps": 113.66},
    {"prompt_tps": 1076.27, "gen_tps": 115.15}
  ]
}
```

#### ZMLX Patched
```json
{
  "median_prompt_tps": 1096.15,
  "median_gen_tps": 113.89,
  "median_ttft_sec": 0.322,
  "peak_memory_gb": 17.55,
  "runs": [
    {"prompt_tps": 1096.54, "gen_tps": 113.57},
    {"prompt_tps": 1096.15, "gen_tps": 113.89},
    {"prompt_tps": 1094.41, "gen_tps": 114.21}
  ]
}
```

---

## Historical Comparison

### Previous Test (Same Day, Different Configuration)
- **Qwen3-30B-A3B** (non-Instruct)
- Short prompt (8 tokens)
- Results: +9.5% prompt, +1.1% generation

### README Claims (Earlier MLX Versions)
- **Qwen3-30B-A3B-Instruct:** +60% prompt, +31% decode
- **Qwen3-30B-A3B:** +61% prompt, +37% decode

### Analysis of Discrepancy

The gap between README claims (1.6x) and current results (1.0x) is likely due to:

1. **MLX Version Differences:** Current test uses MLX 0.30.0, which may have incorporated similar optimizations
2. **Prompt Length Sensitivity:** The 306-token prompt may mask MoE routing benefits vs very short prompts
3. **Model Variations:** Instruct vs base model may have different gating mechanisms
4. **macOS/Metal Improvements:** System-level optimizations may have reduced the advantage of custom kernels

---

## Microbenchmark Results (Kernel Level)

ZMLX still shows significant speedups for **fused operations** that MLX doesn't provide as single ops:

| Operation | MLX | ZMLX | Speedup |
|:----------|:--:|:--:|:-------:|
| **SwiGLU** | 0.86 ms | 0.41 ms | **2.1x** |
| **Dropout** | 2.86 ms | 0.39 ms | **7.3x** |
| **Top-K** | 1.63 ms | 0.47 ms | **3.4x** |
| **Gather-Add** | 0.52 ms | 0.40 ms | **1.3x** |
| RMSNorm | 0.42 ms | 0.42 ms | 1.0x |
| Softmax | 0.52 ms | 0.71 ms | 0.72x |
| Sum | 0.20 ms | 0.36 ms | 0.56x |

**Key Insight:** ZMLX wins on fused operations; MLX built-ins remain faster for basic primitives.

---

## Recommendations for Upstreaming to MLX

### 1. High-Value Integration Targets

Based on the microbenchmarks, these ZMLX patterns would benefit MLX core:

| Pattern | Speedup | Priority | Notes |
|:--------|:-------:|:--------:|:------|
| **Fused SwiGLU/GeGLU** | 2.1x | **High** | Common in modern transformers |
| **Fused Dropout** | 7.3x | **High** | Training workloads |
| **Top-K Gating** | 3.4x | **Medium** | MoE models |
| **Gather-Add Fusion** | 1.3x | **Low** | Specialized use cases |

### 2. Implementation Notes

```python
# Current ZMLX approach (fused kernel)
from zmlx.patch import patch
patch(model)  # Replaces MoE layers with fused versions

# What MLX could provide (built-in optimization)
# - Native fused_swiglu() op
# - Native fused_dropout() op  
# - Optimized topk_gating path in MoE layers
```

### 3. Technical Considerations

**Why MLX built-ins are already fast:**
- `mx.fast.rms_norm`: Apple-optimized Metal kernel
- `mx.softmax`: Optimized parallel reduction
- `mx.fast.scaled_dot_product_attention`: Hardware-accelerated

**Where ZMLX adds value:**
- Operations MLX doesn't have as single ops
- Custom kernel authoring for research/experimentation
- Rapid prototyping of new fused patterns

### 4. Suggested MLX Core Improvements

1. **Native Fused GLU Variants:**
   ```python
   mx.nn.swiglu(x, w1, w2, w3)  # Single op vs 3 matmuls + 1 mul
   ```

2. **Fused Dropout for Training:**
   ```python
   mx.random.dropout_fused(x, p=0.1)  # RNG + mask + mul in one kernel
   ```

3. **Optimized MoE Gating:**
   ```python
   mx.fast.moe_topk_gating(logits, k=2)  # Fused softmax + topk
   ```

---

## Conclusion

### For MLX Team

1. **ZMLX patches are safe** — no regressions observed
2. **Current MLX 0.30.0 is already well-optimized** for standard inference
3. **Consider upstreaming:** Fused SwiGLU, dropout, and MoE gating patterns
4. **ZMLX remains valuable** for custom kernel research and operations MLX doesn't provide

### For ZMLX Users

1. **Use `patch(model)` freely** — it's safe with no regressions
2. **Don't expect large speedups** on current MLX versions for standard models
3. **Benefits may vary** by prompt length, batch size, and specific model architecture
4. **Consider `smart_patch()`** to benchmark and keep only beneficial patterns

---

## Deliverables

1. ✅ **Benchmark results saved:** `benchmarks/results/qwen3_30b_a3b_instruct_comprehensive.json`
2. ✅ **README updated:** Performance claims adjusted to reflect current results
3. ✅ **Summary report:** This document for Awni Hannun / Apple MLX team
4. ✅ **Git repository:** Changes staged for commit

---

## Appendix: Test Commands

```bash
# Reproduce these results
cd ~/clawd/ZMLX
python3 benchmarks/inference_benchmark.py --models qwen3-30b-a3b-instruct --runs 3 --max-tokens 150

# View microbenchmarks
python3 benchmarks/microbench.py
```

---

*Report generated by Clawdbot subagent for Hunter Bown / ZMLX project*
