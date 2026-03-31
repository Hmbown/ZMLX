#!/usr/bin/env python3
"""Validate fused postconv DeltaNet on Qwen3.5 models.

Tests token-identical fidelity and measures decode speedup for:
  - Qwen3.5-4B-MLX-4bit
  - Qwen3.5-9B-MLX-4bit
  - Qwen3.5-27B-4bit (mlx-community)
  - Qwen3.5-35B-A3B-MLX-4bit

Usage:
    python scripts/validate_qwen35_fused_postconv.py [model_path ...]
    python scripts/validate_qwen35_fused_postconv.py  # runs all 4
"""

from __future__ import annotations

import gc
import sys
import time

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

DEFAULT_MODELS = [
    "mlx-community/Qwen3.5-4B-OptiQ-4bit",
    "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    "mlx-community/Qwen3.5-27B-4bit",
    "/Volumes/VIXinSSD/models/Qwen3.5-35B-A3B-MLX-4bit",
]

PROMPTS = [
    "What is 2+2? Answer in one word.",
    "Name the capital of Japan in one word.",
    "Write a haiku about the moon.",
]

MAX_TOKENS = 100
NUM_WARMUP = 1
NUM_TIMED = 3


def test_model(model_path: str) -> dict:
    short_name = model_path.split("/")[-1]
    print(f"\n{'='*60}")
    print(f"  Model: {short_name}")
    print(f"{'='*60}")

    # Load model
    print(f"  Loading...", end=" ", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    print(f"{time.perf_counter() - t0:.1f}s")

    sampler = make_sampler(temp=0.0)

    # Check if model has DeltaNet layers
    has_deltanet = False
    for name, mod in model.named_modules():
        if hasattr(mod, "A_log") and hasattr(mod, "in_proj_qkv"):
            has_deltanet = True
            break

    if not has_deltanet:
        print("  No DeltaNet layers found — skipping fused postconv test")
        print("  (This model may be dense or use a different attention type)")
        del model, tokenizer
        gc.collect()
        return {"model": short_name, "skip": "no DeltaNet layers"}

    # Build prompts
    chat_prompts = []
    for p in PROMPTS:
        chat_prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    # --- Baseline (no patches) ---
    print("  Baseline (unpatched):")
    baseline_outputs = []
    for prompt in chat_prompts:
        out = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler)
        baseline_outputs.append(out)

    # Warmup + timed runs
    for _ in range(NUM_WARMUP):
        generate(model, tokenizer, prompt=chat_prompts[0], max_tokens=MAX_TOKENS, sampler=sampler)

    t0 = time.perf_counter()
    for _ in range(NUM_TIMED):
        for prompt in chat_prompts:
            generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler)
    baseline_time = (time.perf_counter() - t0) / NUM_TIMED
    print(f"    Avg time: {baseline_time:.2f}s ({NUM_TIMED} runs x {len(PROMPTS)} prompts)")

    # --- Patched (fused postconv ON by default now) ---
    from zmlx.patch import patch

    patch(model, verbose=False)
    result = getattr(model, "_zmlx_patch_result", None)
    if result:
        print(f"  Patched: {result.patched_count} modules ({result.pattern_counts})")
    else:
        print("  Patched: (no result info)")

    # Fidelity check
    print("  Fidelity check:")
    patched_outputs = []
    all_identical = True
    for i, prompt in enumerate(chat_prompts):
        out = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler)
        patched_outputs.append(out)
        identical = baseline_outputs[i].strip() == out.strip()
        status = "PASS" if identical else "FAIL"
        if not identical:
            all_identical = False
        print(f"    Prompt {i+1}: {status}")
        if not identical:
            print(f"      Baseline: {baseline_outputs[i].strip()[:80]!r}")
            print(f"      Patched:  {out.strip()[:80]!r}")

    # Warmup + timed runs
    for _ in range(NUM_WARMUP):
        generate(model, tokenizer, prompt=chat_prompts[0], max_tokens=MAX_TOKENS, sampler=sampler)

    t0 = time.perf_counter()
    for _ in range(NUM_TIMED):
        for prompt in chat_prompts:
            generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler)
    patched_time = (time.perf_counter() - t0) / NUM_TIMED
    print(f"    Avg time: {patched_time:.2f}s ({NUM_TIMED} runs x {len(PROMPTS)} prompts)")

    speedup = baseline_time / patched_time if patched_time > 0 else 0
    pct = (speedup - 1) * 100

    print(f"\n  RESULT: {'PASS' if all_identical else 'FAIL'} fidelity, {speedup:.3f}x ({pct:+.1f}%) speedup")

    # Cleanup
    del model, tokenizer
    gc.collect()

    return {
        "model": short_name,
        "fidelity": "PASS" if all_identical else "FAIL",
        "baseline_time": baseline_time,
        "patched_time": patched_time,
        "speedup": speedup,
        "pct": pct,
        "patched_count": result.patched_count if result else 0,
        "pattern_counts": result.pattern_counts if result else {},
    }


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_MODELS

    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)

    results = []
    for model_path in models:
        try:
            r = test_model(model_path)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"model": model_path.split("/")[-1], "error": str(exc)})

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<35} {'Fidelity':<10} {'Speedup':<10} {'Modules'}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        if "skip" in r:
            print(f"  {r['model']:<35} {'SKIP':<10} {'n/a':<10} {r['skip']}")
        elif "error" in r:
            print(f"  {r['model']:<35} {'ERROR':<10} {'n/a':<10} {r['error'][:30]}")
        else:
            print(
                f"  {r['model']:<35} {r['fidelity']:<10} {r['pct']:+.1f}%{'':5} {r['patched_count']}"
            )


if __name__ == "__main__":
    main()
