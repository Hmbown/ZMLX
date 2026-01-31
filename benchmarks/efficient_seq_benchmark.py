#!/usr/bin/env python3
"""Efficient variable sequence length benchmark for Qwen3-30B-A3B-Instruct.

Loads model once per configuration (baseline/patched) and tests all sequence lengths.
"""

import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
class SeqResult:
    seq_length: int
    baseline_prompt_tps: float = 0.0
    baseline_gen_tps: float = 0.0
    baseline_memory_gb: float = 0.0
    patched_prompt_tps: float = 0.0
    patched_gen_tps: float = 0.0
    patched_memory_gb: float = 0.0
    prompt_speedup: float = 0.0
    gen_speedup: float = 0.0

def generate_prompts(tokenizer):
    """Generate prompts for all sequence lengths."""
    base_text = (
        "The development of artificial intelligence has transformed numerous industries, "
        "from healthcare to finance. Machine learning models, particularly deep neural networks, "
        "have demonstrated remarkable capabilities in pattern recognition, natural language "
        "processing, and complex decision making tasks. Modern transformer architectures "
        "have enabled unprecedented scale and performance. "
    )
    
    prompts = {}
    for target_len in SEQ_LENGTHS:
        # Estimate and adjust
        test_tokens = tokenizer.encode(base_text)
        ratio = len(base_text) / len(test_tokens)
        estimated_chars = int(target_len * ratio * 1.1)  # 10% buffer
        
        prompt = (base_text * (estimated_chars // len(base_text) + 1))[:estimated_chars]
        actual_len = len(tokenizer.encode(prompt))
        
        # Fine-tune
        while actual_len < target_len:
            prompt += " Additional technical detail about machine learning systems."
            actual_len = len(tokenizer.encode(prompt))
        while actual_len > target_len + 10:
            prompt = prompt[:-50]
            actual_len = len(tokenizer.encode(prompt))
        
        prompts[target_len] = prompt
        print(f"  Generated prompt: {actual_len} tokens (target: {target_len})")
    
    return prompts

def benchmark_config(model, tokenizer, prompts: dict, config_name: str) -> dict:
    """Benchmark all sequence lengths with one model load."""
    results = {}
    
    for seq_len in SEQ_LENGTHS:
        print(f"\n  Testing {seq_len} tokens...")
        prompt = prompts[seq_len]
        
        prompt_tps_list = []
        gen_tps_list = []
        memory_list = []
        
        for i in range(NUM_RUNS):
            response = mlx_lm.generate(
                model, tokenizer, prompt=prompt, 
                max_tokens=MAX_TOKENS, verbose=False
            )
            prompt_tps_list.append(response.prompt_tps)
            gen_tps_list.append(response.generation_tps)
            memory_list.append(response.peak_memory)
            print(f"    Run {i+1}: prompt={response.prompt_tps:.1f} tok/s, gen={response.generation_tps:.1f} tok/s")
        
        results[seq_len] = {
            'prompt_tps': statistics.median(prompt_tps_list),
            'gen_tps': statistics.median(gen_tps_list),
            'memory_gb': max(memory_list),
        }
        print(f"  Median: prompt={results[seq_len]['prompt_tps']:.1f} tok/s, gen={results[seq_len]['gen_tps']:.1f} tok/s")
    
    return results

def main():
    print(f"{'#'*60}")
    print("# ZMLX Variable Sequence Length Benchmark")
    print(f"# Model: {MODEL_NAME}")
    print("# Hardware: Apple M4 Max 36GB")
    print(f"# Runs per config: {NUM_RUNS}")
    print(f"# Max tokens: {MAX_TOKENS}")
    print(f"{'#'*60}\n")
    
    # Generate prompts first
    print("Generating prompts...")
    _, tokenizer = mlx_lm.load(MODEL_ID)
    prompts = generate_prompts(tokenizer)
    del tokenizer
    gc.collect()
    if hasattr(mx, 'clear_memory_cache'):
        mx.clear_memory_cache()
    
    # Baseline benchmark
    print(f"\n{'='*60}")
    print("BASELINE (unpatched)")
    print(f"{'='*60}")
    print("Loading model...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    
    print("Warming up...")
    _ = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=5, verbose=False)
    mx.eval(mx.zeros(1))
    
    baseline_results = benchmark_config(model, tokenizer, prompts, "baseline")
    
    del model, tokenizer
    gc.collect()
    if hasattr(mx, 'clear_memory_cache'):
        mx.clear_memory_cache()
    
    # Patched benchmark
    print(f"\n{'='*60}")
    print("ZMLX PATCHED")
    print(f"{'='*60}")
    print("Loading model...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    
    print("Applying ZMLX patches...")
    patch(model, verbose=False)
    
    print("Warming up...")
    _ = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=5, verbose=False)
    mx.eval(mx.zeros(1))
    
    patched_results = benchmark_config(model, tokenizer, prompts, "patched")
    
    del model, tokenizer
    gc.collect()
    if hasattr(mx, 'clear_memory_cache'):
        mx.clear_memory_cache()
    
    # Compile results
    final_results = []
    for seq_len in SEQ_LENGTHS:
        baseline = baseline_results[seq_len]
        patched = patched_results[seq_len]
        
        result = SeqResult(
            seq_length=seq_len,
            baseline_prompt_tps=baseline['prompt_tps'],
            baseline_gen_tps=baseline['gen_tps'],
            baseline_memory_gb=baseline['memory_gb'],
            patched_prompt_tps=patched['prompt_tps'],
            patched_gen_tps=patched['gen_tps'],
            patched_memory_gb=patched['memory_gb'],
            prompt_speedup=patched['prompt_tps'] / baseline['prompt_tps'],
            gen_speedup=patched['gen_tps'] / baseline['gen_tps'],
        )
        final_results.append(result)
    
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
        "results": [asdict(r) for r in final_results]
    }
    
    output_file = OUTPUT_DIR / f"{MODEL_NAME.lower().replace('-', '_')}_seq_benchmark.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Seq Len':<10} {'Prompt TPS':<30} {'Gen TPS':<30}")
    print(f"{'':10} {'Baseline':>12} {'Patched':>10} {'Speedup':>6} {'Baseline':>12} {'Patched':>10} {'Speedup':>6}")
    print("-" * 70)
    
    for r in final_results:
        print(f"{r.seq_length:<10} {r.baseline_prompt_tps:>12.1f} {r.patched_prompt_tps:>10.1f} {r.prompt_speedup:>6.2f}x "
              f"{r.baseline_gen_tps:>12.1f} {r.patched_gen_tps:>10.1f} {r.gen_speedup:>6.2f}x")
    
    # Average speedup
    avg_prompt_speedup = statistics.median([r.prompt_speedup for r in final_results])
    avg_gen_speedup = statistics.median([r.gen_speedup for r in final_results])
    
    print(f"\nMedian speedup: Prompt={avg_prompt_speedup:.2f}x, Gen={avg_gen_speedup:.2f}x")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
