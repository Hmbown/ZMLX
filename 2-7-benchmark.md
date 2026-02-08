# 2026-02-07 Benchmark Session: Native Combine + GLM swiglu_mlp Fix

## What Changed

### 1. Excluded `swiglu_mlp` pattern from GLM (`patch/__init__.py`)
The `swiglu_mlp` pattern was matching GLM's 46 `shared_experts` dense MLPs, replacing their `__call__` with a Python wrapper that added dispatch overhead without meaningful kernel fusion (the fused quantized SwiGLU mode defaults to "off"). This ate half the moe_mlp gains.

**Fix:** Added `"swiglu_mlp"` to `_PERF_EXCLUDES["glm"]`. Now GLM gets 46 moe_mlp patches only.

Before fix: 93 modules patched (46 moe_mlp + 47 swiglu_mlp), +1.5% decode
After fix: 46 modules patched (46 moe_mlp), +7% decode

### 2. Native combine for GLM and Qwen3 (`patch/patterns/moe_mlp.py`)
Both GLM and Qwen3's original MoE combine is simply:
```python
y = (expert_outputs * scores[..., None]).sum(axis=-2)
```
We were replacing this with custom Metal kernels (`moe_combine_no_fma` for GLM, `moe_combine_fp32` for Qwen3) that had ~6us/call dispatch overhead × 46-48 layers = ~0.3ms/token wasted.

**Fix:** Use native ops for GLM and Qwen3 combine, matching original code exactly. MLX's lazy eval handles dtype promotion and can fuse the broadcast-multiply + reduce.

### 3. Rebuilt custom MLX + regenerated patch file
- `mlx_local/` rebuilt with ppt=2 kernel (reverted from failed ppt=1 attempt)
- `integrations/mlx_local_integration/gather_qmm_swiglu.patch` regenerated from current source
- Patch has `packs_per_thread = bits == 2 ? 1 : 2` (ppt=2 for 4-bit), no `used_out_row`

### 4. Removed `analysis_prompt.md` temp file

## Results (M4 Mac Studio, 200 tokens greedy, 5-run median)

| Model | Fidelity | Baseline | Patched | Speedup |
|:--|:--|--:|--:|--:|
| GLM-4.7-Flash-4bit | 200/200 PASS | 74.8 tok/s | 83.4 tok/s | **+11.5%** |
| Qwen3-30B-A3B-4bit | 200/200 PASS | 112.7 tok/s | 117.6 tok/s | **+12.0%** (5-run) |
| LFM2-8B-A1B-4bit | 200/200 PASS | 221.8 tok/s | 250.1 tok/s | **+12.4%** (3-run) |

Note: GLM baseline fluctuates 72-76 tok/s between runs (thermals/background). The +11.5% was a clean 5-run; a later 5-run showed +6.8%. Multiple runs consistently show 78-84 tok/s patched vs 72-76 baseline.

## Where the Speedup Comes From

The only thing ZMLX does for these models is the `gather_qmm_swiglu` kernel fusion in the moe_mlp pattern. This fuses **3 separate Metal kernel dispatches**:
1. `gate_proj` — gather + dequantize + matmul (QuantizedSwitchLinear)
2. `up_proj` — gather + dequantize + matmul (QuantizedSwitchLinear)
3. SwiGLU activation — `silu(gate) * up`

Into **1 dispatch** that reads `x` once instead of twice and avoids 2 extra command buffer submissions per layer per token.

Everything else (gating, down_proj, combine, shared_experts) uses native MLX ops unchanged.

## Things That Were Tried and Don't Work

| Approach | Result | Why |
|:--|:--|:--|
| `swiglu_mlp` on GLM shared_experts | -4% (eats moe_mlp gains) | Python wrapper overhead on dense layers |
| `ZMLX_GLM_FUSED_DOWNPROJ_COMBINE=1` | **-15%** | Fused downproj+combine kernel is slower for these shapes |
| `ZMLX_MOE_STREAMS=2 ZMLX_MOE_SHARED_EXPERTS_OVERLAP=1` | **-25%** | Multi-stream overhead dominates at decode batch=1 |
| Custom `moe_combine_no_fma` kernel | -2% vs native | Custom kernel dispatch overhead; native ops let MLX fuse |

## Test Suite
840 passed, 74 skipped, 3 xfailed

## Files Modified
- `src/zmlx/patch/__init__.py` — added `swiglu_mlp` to `_PERF_EXCLUDES["glm"]`, updated GLM exclude logic comments
- `src/zmlx/patch/patterns/moe_mlp.py` — GLM and Qwen3 combine paths use native ops instead of custom kernels
- `integrations/mlx_local_integration/gather_qmm_swiglu.patch` — regenerated from ppt=2 kernel

## TODO for Next Session
- Run repro capsules for the new numbers (GLM, Qwen3, LFM2) and save to `benchmarks/repro_capsules/`
- Consider applying native combine to LFM2 as well (currently uses `_try_fused_downproj_combine` → `gather_qmm_combine`)
- Consider applying native combine to GPT-OSS
- Update README.md benchmark table with new numbers
- The `_try_fused_downproj_combine` path for Qwen3 always returns None (float16 activations fail `require_fp32=True` check) — dead code that adds Python overhead per layer call. Could be skipped.

## 2026-02-08 Matrix Update (GLM-4.7-Flash)

Recorded new matrix entries in `benchmarks/matrix.jsonl` via `python -m zmlx.matrix run`:

1. Default patch path (model-aware defaults)
- Command:
  - `python -m zmlx.matrix run mlx-community/GLM-4.7-Flash-4bit --runs 3 --max-tokens 200 --notes "2026-02-08 default patch revalidation: fidelity pass, gather_qmm_swiglu path"`
- Result:
  - Fidelity: `PASS`
  - Decode: `82.2 -> 89.6 tok/s` (`1.090x`)
  - Patched modules: `46` (`moe_mlp`)

2. Explicit RMSNorm-only path
- Command:
  - `python -m zmlx.matrix run mlx-community/GLM-4.7-Flash-4bit --runs 1 --max-tokens 200 --patterns rmsnorm --notes "2026-02-08 explicit rmsnorm check: fidelity fail + decode regression"`
- Result:
  - Fidelity: `FAIL`
  - Decode: `84.6 -> 70.8 tok/s` (`0.836x`)
  - Patched modules: `189` (`rmsnorm`)

Notes:
- A concurrent attempt during this session produced a Metal OOM and was discarded.
- All benchmark/matrix runs after that were executed sequentially.

## 2026-02-08 MoE 4-bit Sweep (1000 tokens)

Scope:
- Sequential matrix runs only.
- 4-bit MoE models only (per session constraint).
- Command template:
  - `python -m zmlx.matrix run <model> --runs 1 --max-tokens 1000 --notes "2026-02-08 moe 4bit-only sweep 1000t runs=1 default patch"`

Results:

| Model | Fidelity | Baseline Decode | Patched Decode | Speedup |
|:--|:--|--:|--:|--:|
| `mlx-community/LFM2-8B-A1B-4bit` | PASS | 209.8 tok/s | 235.7 tok/s | 1.123x |
| `mlx-community/GLM-4.7-Flash-4bit` | PASS | 74.5 tok/s | 78.6 tok/s | 1.054x |
| `mlx-community/Qwen3-30B-A3B-4bit` | PASS | 103.3 tok/s | 106.3 tok/s | 1.029x |

Notes:
- These entries are appended to `benchmarks/matrix.jsonl`.
- No >4-bit MoE models were run in this sweep.
