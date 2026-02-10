# KVTC Real-Model Testing Plan

Status: Synthetic tests pass (24/24). No real model runs yet.

## Prerequisites

```bash
cd /Volumes/VIXinSSD/ZMLX
source .venv/bin/activate
pip install -e ".[kvtc]"        # numpy
pip install mlx-lm              # for dump step
```

Models used below (already cached via HF_HOME or will download on first use):
- `mlx-community/LFM2-8B-A1B-4bit`
- `mlx-community/Qwen3-30B-A3B-4bit`
- `mlx-community/glm-4-9b-chat-4bit` (or whichever GLM-4.7-Flash variant is available — confirm exact HF path before running)

## Phase 1: Cache Dumping

Goal: Prove `python -m zmlx.kvtc dump` can extract real KV caches to NPZ.

### 1a. LFM2
```bash
python -m zmlx.kvtc dump \
  --model mlx-community/LFM2-8B-A1B-4bit \
  --prompt "The quick brown fox jumps over the lazy dog. Once upon a time in a land far away," \
  --max-tokens 512 \
  --preset lfm2 \
  --out /tmp/kvtc_lfm2_cache.npz
```

**Verify:**
- File created, non-empty
- `python -c "import numpy as np; d=np.load('/tmp/kvtc_lfm2_cache.npz'); print('k:', d['k'].shape, 'v:', d['v'].shape)"`
- Expected shape: `k: (24, 1, 8, seq, 64)`, `v: (24, 1, 8, seq, 64)` where seq ≈ prompt_tokens + 512

**Known risk:** The dump script accesses `model.cache` after `mlx_lm.generate()`. If mlx-lm clears the cache after generation, the script will fail. Fix: patch dump.py to hook into the generation loop via a custom sampler or step function that snapshots the cache before it's cleared. Alternatively, do a manual dump:

```python
import mlx_lm, numpy as np
model, tokenizer = mlx_lm.load("mlx-community/LFM2-8B-A1B-4bit")
# Run a forward pass to populate cache, then extract
# (details depend on mlx-lm version — may need to inspect model.layers[0].self_attn)
```

If `model.cache` doesn't work, the fallback is to manually walk `model.model.layers` and extract each layer's `self_attn.cache` or equivalent. Document what works.

### 1b. Qwen3
```bash
python -m zmlx.kvtc dump \
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --prompt "Explain quantum entanglement in simple terms." \
  --max-tokens 256 \
  --preset qwen3 \
  --out /tmp/kvtc_qwen3_cache.npz
```

**Verify:** `k: (48, 1, 4, seq, 128)`, `v: (48, 1, 4, seq, 128)`

### 1c. GLM-4.7-Flash
```bash
python -m zmlx.kvtc dump \
  --model <glm-model-path> \
  --prompt "What is the meaning of life?" \
  --max-tokens 256 \
  --preset glm \
  --out /tmp/kvtc_glm_cache.npz
```

**Verify:**
- `k: (47, 1, 1, seq, 576)` — the MLA latent+pe concatenation
- `v` should be empty or zero (head_dim=0) if the GLM adapter works correctly
- If v is NOT empty, investigate what GLM actually stores in the values slot and update `GLMMlaCacheAdapter` accordingly

**Known risk:** GLM MLA may not store cache the same way as standard models. The `keys` slot might be the compressed latent `c_kv` (512-dim) with `k_pe` (64-dim) appended, OR the model might store them separately. Inspect the actual cache object structure before trusting the adapter.

## Phase 2: Calibration

Goal: Run DP calibration on real cache dumps to produce artifacts.

### 2a. LFM2
```bash
python -m zmlx.kvtc calibrate \
  --npz /tmp/kvtc_lfm2_cache.npz \
  --preset lfm2 \
  --compression-ratio 4.0 \
  --out /tmp/kvtc_cal_lfm2/
```

**Verify:**
- Directory contains: `k_mu.npy`, `k_V.npy`, `k_plan.json`, `v_mu.npy`, `v_V.npy`, `v_plan.json`, `meta.json`
- `k_plan.json` has non-empty groups with mixed qtypes (not all "none")
- `v_plan.json` same
- `meta.json` records mode=dual_stream and rope config

**Timings:** DP calibration is O(r² × budget) — may be slow for large p. If it takes >5 min, try `--rank 32` to limit PCA rank.

### 2b. Qwen3
```bash
python -m zmlx.kvtc calibrate \
  --npz /tmp/kvtc_qwen3_cache.npz \
  --preset qwen3 \
  --compression-ratio 4.0 \
  --out /tmp/kvtc_cal_qwen3/
```

### 2c. GLM (single_stream)
```bash
python -m zmlx.kvtc calibrate \
  --npz /tmp/kvtc_glm_cache.npz \
  --preset glm \
  --compression-ratio 4.0 \
  --out /tmp/kvtc_cal_glm/
```

**Verify:**
- `v_plan.json` should have empty groups `{"groups": []}` (single_stream dummy)
- `v_mu.npy` and `v_V.npy` should be tiny dummy arrays
- `k_plan.json` should have real groups

## Phase 3: Compress / Decompress Roundtrip

Goal: Verify real caches survive compress→decompress with acceptable error.

```python
import numpy as np
from zmlx.kvtc import KVTCCacheCodec, CalibrationArtifacts, model_preset

# --- LFM2 ---
preset = model_preset("lfm2")
arts = CalibrationArtifacts.from_dir("/tmp/kvtc_cal_lfm2/")
codec = KVTCCacheCodec(arts, w=128, s=4, rope_cfg=preset.rope, mode=preset.mode)

data = np.load("/tmp/kvtc_lfm2_cache.npz")
k_orig = [data["k"][i] for i in range(data["k"].shape[0])]
v_orig = [data["v"][i] for i in range(data["v"].shape[0])]

blob = codec.compress(k_orig, v_orig)
k_hat, v_hat = codec.decompress(blob)

# Report
print(f"Blob size: {len(blob):,} bytes")
print(f"Original size: {sum(k.nbytes + v.nbytes for k, v in zip(k_orig, v_orig)):,} bytes")
print(f"Compression ratio: {sum(k.nbytes + v.nbytes for k, v in zip(k_orig, v_orig)) / len(blob):.1f}x")

# Check prefix/suffix exact match
for i in range(len(k_orig)):
    assert np.array_equal(k_hat[i][:, :, :4, :], k_orig[i][:, :, :4, :]), f"K prefix mismatch layer {i}"
    assert np.array_equal(k_hat[i][:, :, -128:, :], k_orig[i][:, :, -128:, :]), f"K suffix mismatch layer {i}"

# Check middle region MSE
for i in range(len(k_orig)):
    mid_orig = k_orig[i][:, :, 4:-128, :].astype(np.float32)
    mid_hat = k_hat[i][:, :, 4:-128, :].astype(np.float32)
    mse = np.mean((mid_orig - mid_hat) ** 2)
    print(f"  Layer {i} K mid MSE: {mse:.6f}")

print("LFM2 roundtrip OK")
```

Repeat similarly for Qwen3 (dual_stream) and GLM (single_stream, check v_hat is empty).

**Pass criteria:**
- Prefix/suffix exact match (these are stored raw)
- Mid-region per-layer MSE < 0.1 (this is a soft target — actual threshold depends on model sensitivity)
- Compression ratio ≥ 3x at CR=4.0 target (some overhead from headers/prefix/suffix)

## Phase 4: Generation Fidelity

Goal: Verify that using KVTC-compressed cache doesn't break model output quality. This is the hardest step and may require changes to how mlx-lm uses the cache.

### Approach A: Offline comparison (no live injection)
1. Run model normally, save full output text + logits at a few token positions
2. Run model, dump cache mid-generation, compress, decompress, compare logits
3. This doesn't require modifying mlx-lm's generation loop

```python
# Pseudocode — actual implementation depends on mlx-lm internals
# 1. Generate N tokens, snapshot cache
# 2. Compress + decompress cache
# 3. Feed decompressed cache back and generate 1 more token
# 4. Compare that token's logits to what the original cache would have produced
```

### Approach B: Perplexity benchmark
1. Pick a held-out text (e.g., first 2K tokens of a Wikipedia article)
2. Run model with original cache → measure perplexity
3. At a midpoint (e.g., after 512 tokens), compress+decompress the cache
4. Continue generation → measure perplexity on remaining tokens
5. Compare: perplexity delta < 0.5 is good, < 0.1 is excellent

### What to record
For each model, record in a results JSON:
- Model name, quant, hardware
- Sequence length at compression point
- Compression ratio achieved
- Prefix/suffix token counts (s, w)
- Per-layer MSE for K and V middle regions
- Generation quality metric (perplexity delta or logit cosine similarity)

## Phase 5: Edge Cases and Robustness

### 5a. Short sequences
Test with seq_len barely above s+w (e.g., s=4, w=128, seq=140 → only 8 middle tokens).
Should work but compression ratio will be poor.

### 5b. Very long sequences
Test with seq_len=4096+ to stress memory and verify chunked encoding works.

### 5c. Multiple calibration files
Calibrate from 2+ different prompts to check that calibration generalizes:
```bash
python -m zmlx.kvtc dump --model ... --prompt "prompt A" --out /tmp/cal_a.npz
python -m zmlx.kvtc dump --model ... --prompt "prompt B" --out /tmp/cal_b.npz
python -m zmlx.kvtc calibrate --npz /tmp/cal_a.npz /tmp/cal_b.npz --preset lfm2 --out /tmp/multi_cal/
```

### 5d. Compression ratio sweep
Test at CR=2, 4, 8, 16 for each model. Plot MSE vs CR. Find the sweet spot.

## Phase 6: Integration with Exo (stretch)

If Phases 1–4 pass, test KVTC within the exo distributed inference pipeline:
1. Add a hook in exo's generation loop that periodically compresses the KV cache
2. Verify that exo+KVTC produces same output as exo alone (token-identical for lossless prefix/suffix tokens)
3. Measure throughput impact (KVTC compress/decompress adds CPU overhead but reduces memory)

This is lower priority — get standalone working first.

## Likely Issues and Fixes

| Issue | Symptom | Fix |
|:------|:--------|:----|
| `model.cache` not accessible after generation | dump.py crashes with `None` | Patch dump.py to use a generation hook or manual forward pass |
| GLM cache structure different than expected | Shape mismatch in adapter | Inspect actual cache object, update `GLMMlaCacheAdapter` |
| Calibration too slow for large models | >10 min for Qwen3 (p=48×4×128=24576) | Use `--rank 64` or `--rank 32` to limit PCA rank |
| Calibration artifacts dim mismatch | ValueError on compress | Verify NPZ shapes match preset (L, H, D) exactly |
| High MSE on middle region | MSE > 1.0 | Increase bit budget (lower CR), increase PCA rank, or calibrate on more diverse prompts |
| RoPE config wrong for a model | Recovered values drift | Check model's actual RoPE config in its `config.json` and update preset |
| mlx-lm API changed | Import errors or signature mismatches | Check mlx-lm version, update dump.py accordingly |

## Checklist Summary

- [ ] Phase 1a: LFM2 cache dump works, shapes correct
- [ ] Phase 1b: Qwen3 cache dump works, shapes correct
- [ ] Phase 1c: GLM cache dump works, keys-only confirmed
- [ ] Phase 2a: LFM2 calibration produces valid artifacts
- [ ] Phase 2b: Qwen3 calibration produces valid artifacts
- [ ] Phase 2c: GLM calibration produces valid artifacts (single_stream)
- [ ] Phase 3: LFM2 compress/decompress, prefix exact, MSE < 0.1
- [ ] Phase 3: Qwen3 compress/decompress, prefix exact, MSE < 0.1
- [ ] Phase 3: GLM compress/decompress, keys only, MSE < 0.1
- [ ] Phase 4: At least one model tested for generation fidelity
- [ ] Phase 5a: Short sequence edge case
- [ ] Phase 5d: CR sweep for at least one model
- [ ] All results recorded in a JSON/markdown file under `benchmarks/`
