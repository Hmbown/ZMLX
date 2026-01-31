#!/usr/bin/env python3
"""Quick benchmark for ZMLX testing - focused on key models."""

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx_lm

from zmlx.patch import patch

# Results storage
OUTPUT_DIR = Path("benchmarks/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class BenchmarkMetrics:
    model: str
    patched: bool
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tokens: int = 0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    runtime_sec: float = 0.0

def run_benchmark(model_id: str, patched: bool = False, max_tokens: int = 200) -> BenchmarkMetrics:
    """Run a single benchmark."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"Patched: {patched}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading {model_id}...")
    start = time.time()
    model, tokenizer = mlx_lm.load(model_id)
    load_time = time.time() - start
    print(f"  Loaded in {load_time:.1f}s")
    
    # Apply patches if requested
    if patched:
        print("Applying ZMLX patches...")
        patch(model, verbose=False)
    
    # Warmup
    print("Warming up...")
    _ = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=5)
    mx.eval(mx.zeros(1))
    
    # Test prompt
    prompt = "Explain the key differences between transformers and RNNs in 3 sentences."
    
    # Run generation
    print(f"Running generation (max_tokens={max_tokens})...")
    start = time.time()
    
    response = None
    for resp in mlx_lm.stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        response = resp
    
    runtime = time.time() - start
    
    metrics = BenchmarkMetrics(
        model=model_id,
        patched=patched,
        prompt_tokens=response.prompt_tokens if response else 0,
        prompt_tps=response.prompt_tps if response else 0.0,
        generation_tokens=response.generation_tokens if response else 0,
        generation_tps=response.generation_tps if response else 0.0,
        peak_memory_gb=response.peak_memory if response else 0.0,
        runtime_sec=runtime
    )
    
    print("\nResults:")
    print(f"  Prompt TPS:     {metrics.prompt_tps:.1f} tok/s")
    print(f"  Generation TPS: {metrics.generation_tps:.1f} tok/s")
    print(f"  Peak Memory:    {metrics.peak_memory_gb:.2f} GB")
    print(f"  Runtime:        {metrics.runtime_sec:.2f}s")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    if hasattr(mx, 'clear_memory_cache'):
        mx.clear_memory_cache()
    
    return metrics

def compare_results(baseline: BenchmarkMetrics, patched: BenchmarkMetrics) -> dict:
    """Compare baseline vs patched results."""
    prompt_speedup = patched.prompt_tps / baseline.prompt_tps if baseline.prompt_tps > 0 else 0
    gen_speedup = patched.generation_tps / baseline.generation_tps if baseline.generation_tps > 0 else 0
    
    return {
        "prompt_speedup_x": round(prompt_speedup, 3),
        "gen_speedup_x": round(gen_speedup, 3),
        "prompt_tps_baseline": round(baseline.prompt_tps, 1),
        "prompt_tps_patched": round(patched.prompt_tps, 1),
        "gen_tps_baseline": round(baseline.generation_tps, 1),
        "gen_tps_patched": round(patched.generation_tps, 1),
        "peak_mem_baseline_gb": round(baseline.peak_memory_gb, 2),
        "peak_mem_patched_gb": round(patched.peak_memory_gb, 2),
    }

def main():
    # Models to test (4-bit for smaller download size)
    models = {
        "lfm2-8b": "mlx-community/LFM2-8B-A1B-4bit",
        "qwen3-30b-a3b": "mlx-community/Qwen3-30B-A3B-4bit",
    }
    
    results = {}
    max_tokens = 300  # Shorter for faster testing
    
    for name, model_id in models.items():
        print(f"\n{'#'*60}")
        print(f"# Benchmarking: {name}")
        print(f"{'#'*60}")
        
        try:
            # Baseline
            baseline = run_benchmark(model_id, patched=False, max_tokens=max_tokens)
            
            # Patched
            patched = run_benchmark(model_id, patched=True, max_tokens=max_tokens)
            
            # Compare
            comparison = compare_results(baseline, patched)
            results[name] = {
                "model_id": model_id,
                "comparison": comparison,
                "baseline": asdict(baseline),
                "patched": asdict(patched),
            }
            
            print(f"\n{'='*60}")
            print(f"Summary for {name}:")
            print(f"  Prompt speedup:   {comparison['prompt_speedup_x']:.2f}x")
            print(f"  Generation speedup: {comparison['gen_speedup_x']:.2f}x")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"ERROR benchmarking {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}
    
    # Save results
    output_file = OUTPUT_DIR / "quick_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for name, res in results.items():
        if "error" in res:
            print(f"{name}: ERROR - {res['error']}")
        else:
            c = res["comparison"]
            print(f"{name}:")
            print(f"  Prompt:   {c['prompt_tps_baseline']:.1f} -> {c['prompt_tps_patched']:.1f} tok/s ({c['prompt_speedup_x']:.2f}x)")
            print(f"  Generate: {c['gen_tps_baseline']:.1f} -> {c['gen_tps_patched']:.1f} tok/s ({c['gen_speedup_x']:.2f}x)")

if __name__ == "__main__":
    main()
