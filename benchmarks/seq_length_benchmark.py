#!/usr/bin/env python3
"""Variable sequence length benchmark for Qwen3-30B-A3B-Instruct."""

import time
import json
import gc
from pathlib import Path
from dataclasses import dataclass, field, asdict
import statistics

import mlx.core as mx
import mlx_lm
from zmlx.patch import patch

# Config
MODEL_ID = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
MODEL_NAME = "Qwen3-30B-A3B-Instruct"
OUTPUT_DIR = Path("benchmarks/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test configurations
SEQ_LENGTHS = [128, 512, 1024, 2048]
NUM_RUNS = 5
MAX_TOKENS = 150

@dataclass 
class BenchmarkResult:
    seq_length: int
    baseline_prompt_tps: float = 0.0
    baseline_gen_tps: float = 0.0
    baseline_memory_gb: float = 0.0
    patched_prompt_tps: float = 0.0
    patched_gen_tps: float = 0.0
    patched_memory_gb: float = 0.0
    prompt_speedup: float = 0.0
    gen_speedup: float = 0.0

@dataclass
class RunMetrics:
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    memory_gb: float = 0.0

def generate_prompt(target_tokens: int, tokenizer) -> str:
    """Generate a prompt of approximately target tokens."""
    base = "The development of artificial intelligence has transformed numerous industries, from healthcare to finance. Machine learning models, particularly deep neural networks, have demonstrated remarkable capabilities in pattern recognition and prediction tasks. "
    
    # Estimate tokens
    tokens = tokenizer.encode(base)
    ratio = len(base) / len(tokens)
    chars_needed = int(target_tokens * ratio)
    
    prompt = (base * (chars_needed // len(base) + 1))[:chars_needed]
    actual_tokens = len(tokenizer.encode(prompt))
    
    # Adjust if needed
    if actual_tokens < target_tokens:
        prompt += " Additional context and detail are provided here to reach the target length. " * 10
        prompt = prompt[:int(target_tokens * ratio)]
    
    final_tokens = len(tokenizer.encode(prompt))
    print(f"    Target: {target_tokens}, Actual: {final_tokens} tokens")
    return prompt

def run_single_benchmark(model_id: str, patched: bool, prompt: str, max_tokens: int, num_runs: int) -> RunMetrics:
    """Run benchmark with fresh model load."""
    print(f"  Loading model (patched={patched})...")
    model, tokenizer = mlx_lm.load(model_id)
    
    if patched:
        print("  Applying patches...")
        patch(model, verbose=False)
    
    # Warmup
    print("  Warming up...")
    _ = mlx_lm.generate(model, tokenizer, prompt="Hi", max_tokens=5)
    mx.eval(mx.zeros(1))
    
    # Runs
    prompt_tps_list = []
    gen_tps_list = []
    memory_list = []
    
    for i in range(num_runs):
        response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        prompt_tps_list.append(response.prompt_tps)
        gen_tps_list.append(response.generation_tps)
        memory_list.append(response.peak_memory)
        print(f"    Run {i+1}/{num_runs}: prompt={response.prompt_tps:.1f} tok/s, gen={response.generation_tps:.1f} tok/s")
    
    # Cleanup
    del model, tokenizer
    gc.collect()
    if hasattr(mx, 'clear_memory_cache'):
        mx.clear_memory_cache()
    
    return RunMetrics(
        prompt_tps=statistics.median(prompt_tps_list),
        gen_tps=statistics.median(gen_tps_list),
        memory_gb=max(memory_list)
    )

def benchmark_sequence_length(seq_length: int) -> BenchmarkResult:
    """Benchmark one sequence length."""
    print(f"\n{'='*60}")
    print(f"SEQUENCE LENGTH: {seq_length} tokens")
    print(f"{'='*60}")
    
    # Load tokenizer for prompt generation
    _, tokenizer = mlx_lm.load(MODEL_ID)
    prompt = generate_prompt(seq_length, tokenizer)
    del tokenizer
    gc.collect()
    
    # Baseline
    print("\n--- BASELINE ---")
    baseline = run_single_benchmark(MODEL_ID, False, prompt, MAX_TOKENS, NUM_RUNS)
    
    # Patched  
    print("\n--- ZMLX PATCHED ---")
    patched = run_single_benchmark(MODEL_ID, True, prompt, MAX_TOKENS, NUM_RUNS)
    
    result = BenchmarkResult(
        seq_length=seq_length,
        baseline_prompt_tps=baseline.prompt_tps,
        baseline_gen_tps=baseline.gen_tps,
        baseline_memory_gb=baseline.memory_gb,
        patched_prompt_tps=patched.prompt_tps,
        patched_gen_tps=patched.gen_tps,
        patched_memory_gb=patched.memory_gb,
        prompt_speedup=patched.prompt_tps / baseline.prompt_tps if baseline.prompt_tps > 0 else 0,
        gen_speedup=patched.gen_tps / baseline.gen_tps if baseline.gen_tps > 0 else 0,
    )
    
    print(f"\n--- RESULTS ---")
    print(f"  Prompt: {baseline.prompt_tps:.1f} -> {patched.prompt_tps:.1f} tok/s ({result.prompt_speedup:.2f}x)")
    print(f"  Gen:    {baseline.gen_tps:.1f} -> {patched.gen_tps:.1f} tok/s ({result.gen_speedup:.2f}x)")
    print(f"  Memory: {baseline.memory_gb:.2f} -> {patched.memory_gb:.2f} GB")
    
    return result

def main():
    print(f"{'#'*60}")
    print(f"# ZMLX Variable Sequence Length Benchmark")
    print(f"# Model: {MODEL_NAME}")
    print(f"# Runs per config: {NUM_RUNS}")
    print(f"# Max tokens: {MAX_TOKENS}")
    print(f"{'#'*60}")
    
    results = []
    for seq_len in SEQ_LENGTHS:
        result = benchmark_sequence_length(seq_len)
        results.append(result)
    
    # Save results
    output = {
        "model": MODEL_ID,
        "model_name": MODEL_NAME,
        "hardware": "Apple M4 Max 36GB",
        "mlx_version": "0.30.0",
        "zmlx_version": "0.6.3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_runs": NUM_RUNS,
        "max_tokens": MAX_TOKENS,
        "results": [asdict(r) for r in results]
    }
    
    output_file = OUTPUT_DIR / f"{MODEL_NAME.lower().replace('-', '_')}_seq_length_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Seq Len':<10} {'Prompt TPS':<30} {'Gen TPS':<30}")
    print(f"{'':10} {'Baseline':>12} {'Patched':>10} {'Speedup':>6} {'Baseline':>12} {'Patched':>10} {'Speedup':>6}")
    print("-" * 70)
    
    for r in results:
        print(f"{r.seq_length:<10} {r.baseline_prompt_tps:>12.1f} {r.patched_prompt_tps:>10.1f} {r.prompt_speedup:>6.2f}x "
              f"{r.baseline_gen_tps:>12.1f} {r.patched_gen_tps:>10.1f} {r.gen_speedup:>6.2f}x")
    
    # Average
    avg_prompt_speedup = statistics.median([r.prompt_speedup for r in results])
    avg_gen_speedup = statistics.median([r.gen_speedup for r in results])
    print(f"\nMedian speedup: Prompt={avg_prompt_speedup:.2f}x, Gen={avg_gen_speedup:.2f}x")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
