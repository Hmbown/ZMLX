#!/usr/bin/env python3
"""Comprehensive benchmark for Qwen3-30B-A3B-Instruct with multiple sequence lengths.

Tests different prompt lengths (128, 512, 1024, 2048 tokens) to measure how
ZMLX-patched MLX performs across various sequence lengths.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx_lm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
MODEL_NAME = "Qwen3-30B-A3B-Instruct"
OUTPUT_DIR = Path("benchmarks/results")

# Sequence lengths to test
SEQUENCE_LENGTHS = [128, 512, 1024, 2048]
NUM_RUNS = 5  # Multiple runs for statistical significance
MAX_TOKENS = 200  # Tokens to generate

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    """Metrics from a single generation run."""
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tokens: int = 0
    generation_tps: float = 0.0
    ttft_sec: float = 0.0
    total_time_sec: float = 0.0
    peak_memory_gb: float = 0.0

@dataclass
class SequenceResult:
    """Results for a specific sequence length."""
    seq_length: int
    baseline_runs: list[RunMetrics] = field(default_factory=list)
    patched_runs: list[RunMetrics] = field(default_factory=list)
    
    # Aggregated metrics
    baseline_median_prompt_tps: float = 0.0
    baseline_median_gen_tps: float = 0.0
    baseline_median_ttft_sec: float = 0.0
    baseline_peak_memory_gb: float = 0.0
    
    patched_median_prompt_tps: float = 0.0
    patched_median_gen_tps: float = 0.0
    patched_median_ttft_sec: float = 0.0
    patched_peak_memory_gb: float = 0.0
    
    prompt_speedup_x: float = 0.0
    gen_speedup_x: float = 0.0
    ttft_delta_sec: float = 0.0
    memory_delta_gb: float = 0.0

# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------
def generate_prompt(target_tokens: int, tokenizer) -> str:
    """Generate a prompt of approximately target_tokens length."""
    # Use a repeating technical text to get consistent token counts
    base_text = (
        "Artificial intelligence and machine learning are transforming how we "
        "approach complex computational problems. Deep neural networks, particularly "
        "transformer architectures, have demonstrated remarkable capabilities in "
        "natural language processing, computer vision, and multimodal understanding. "
        "The key innovation of attention mechanisms allows models to focus on relevant "
        "parts of the input sequence, enabling better context understanding. "
    )
    
    # Encode to count tokens
    tokens = tokenizer.encode(base_text)
    
    # Calculate repetitions needed
    repeats = max(1, target_tokens // len(tokens) + 1)
    
    # Generate prompt
    prompt = (base_text * repeats)[:target_tokens * 6]  # Approximate chars per token
    
    # Verify token count
    final_tokens = tokenizer.encode(prompt)
    print(f"  Generated prompt: {len(final_tokens)} tokens (target: {target_tokens})")
    
    return prompt

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------
def load_model():
    """Load model + tokenizer."""
    print(f"Loading {MODEL_ID}...")
    model, tokenizer = mlx_lm.load(MODEL_ID)
    print("  Model loaded successfully")
    return model, tokenizer

def warmup(model, tokenizer):
    """Run warmup to compile shaders."""
    print("  Warming up (compiling Metal shaders)...")
    _ = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=5)
    mx.eval(mx.zeros(1))

def timed_generate(model, tokenizer, prompt: str, max_tokens: int) -> RunMetrics:
    """Run one generation and collect metrics."""
    wall_start = time.perf_counter()
    ttft = 0.0
    last_response = None

    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens
    ):
        if last_response is None or last_response.generation_tokens == 0:
            ttft = time.perf_counter() - wall_start
        last_response = response

    total_time = time.perf_counter() - wall_start

    if last_response is None:
        return RunMetrics()

    return RunMetrics(
        prompt_tokens=last_response.prompt_tokens,
        prompt_tps=last_response.prompt_tps,
        generation_tokens=last_response.generation_tokens,
        generation_tps=last_response.generation_tps,
        ttft_sec=ttft,
        total_time_sec=total_time,
        peak_memory_gb=last_response.peak_memory,
    )

def apply_patches(model):
    """Apply ZMLX patches."""
    from zmlx.patch import patch as zmlx_patch
    print("  Applying ZMLX patches (fused activations)...")
    zmlx_patch(model, verbose=False)

def cleanup():
    """Clean up memory."""
    gc.collect()
    if hasattr(mx, 'clear_memory_cache'):
        mx.clear_memory_cache()

# ---------------------------------------------------------------------------
# Run benchmark for one sequence length
# ---------------------------------------------------------------------------
def benchmark_sequence_length(seq_length: int, num_runs: int, max_tokens: int) -> SequenceResult:
    """Benchmark baseline vs patched for a specific sequence length."""
    print(f"\n{'='*70}")
    print(f"  SEQUENCE LENGTH: {seq_length} tokens")
    print(f"{'='*70}")
    
    result = SequenceResult(seq_length=seq_length)
    
    # Load fresh model for baseline
    print("\n--- BASELINE (unpatched) ---")
    model, tokenizer = load_model()
    prompt = generate_prompt(seq_length, tokenizer)
    warmup(model, tokenizer)
    
    baseline_runs = []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end=" ")
        metrics = timed_generate(model, tokenizer, prompt, max_tokens)
        baseline_runs.append(metrics)
        print(f"prompt: {metrics.prompt_tps:.1f} tok/s | gen: {metrics.generation_tps:.1f} tok/s | mem: {metrics.peak_memory_gb:.2f} GB")
    
    del model, tokenizer
    cleanup()
    
    # Load fresh model for patched
    print("\n--- ZMLX PATCHED ---")
    model, tokenizer = load_model()
    apply_patches(model)
    warmup(model, tokenizer)
    
    patched_runs = []
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end=" ")
        metrics = timed_generate(model, tokenizer, prompt, max_tokens)
        patched_runs.append(metrics)
        print(f"prompt: {metrics.prompt_tps:.1f} tok/s | gen: {metrics.generation_tps:.1f} tok/s | mem: {metrics.peak_memory_gb:.2f} GB")
    
    del model, tokenizer
    cleanup()
    
    # Aggregate results
    result.baseline_runs = baseline_runs
    result.patched_runs = patched_runs
    
    result.baseline_median_prompt_tps = statistics.median(r.prompt_tps for r in baseline_runs)
    result.baseline_median_gen_tps = statistics.median(r.generation_tps for r in baseline_runs)
    result.baseline_median_ttft_sec = statistics.median(r.ttft_sec for r in baseline_runs)
    result.baseline_peak_memory_gb = max(r.peak_memory_gb for r in baseline_runs)
    
    result.patched_median_prompt_tps = statistics.median(r.prompt_tps for r in patched_runs)
    result.patched_median_gen_tps = statistics.median(r.generation_tps for r in patched_runs)
    result.patched_median_ttft_sec = statistics.median(r.ttft_sec for r in patched_runs)
    result.patched_peak_memory_gb = max(r.peak_memory_gb for r in patched_runs)
    
    # Calculate speedups
    result.prompt_speedup_x = result.patched_median_prompt_tps / result.baseline_median_prompt_tps
    result.gen_speedup_x = result.patched_median_gen_tps / result.baseline_median_gen_tps
    result.ttft_delta_sec = result.patched_median_ttft_sec - result.baseline_median_ttft_sec
    result.memory_delta_gb = result.patched_peak_memory_gb - result.baseline_peak_memory_gb
    
    # Print summary
    print(f"\n--- SUMMARY for {seq_length} tokens ---")
    print(f"  Prompt TPS:   {result.baseline_median_prompt_tps:.1f} -> {result.patched_median_prompt_tps:.1f} tok/s ({result.prompt_speedup_x:.2f}x)")
    print(f"  Gen TPS:      {result.baseline_median_gen_tps:.1f} -> {result.patched_median_gen_tps:.1f} tok/s ({result.gen_speedup_x:.2f}x)")
    print(f"  TTFT:         {result.baseline_median_ttft_sec:.3f} -> {result.patched_median_ttft_sec:.3f}s ({result.ttft_delta_sec:+.3f}s)")
    print(f"  Peak Memory:  {result.baseline_peak_memory_gb:.2f} -> {result.patched_peak_memory_gb:.2f} GB ({result.memory_delta_gb:+.2f} GB)")
    
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Comprehensive Qwen3-30B-A3B-Instruct benchmark")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs per config")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Tokens to generate")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=SEQUENCE_LENGTHS,
                        help="Sequence lengths to test")
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print("# ZMLX Comprehensive Benchmark")
    print(f"# Model: {MODEL_NAME}")
    print(f"# Runs per config: {args.runs}")
    print(f"# Max tokens to generate: {args.max_tokens}")
    print(f"# Sequence lengths: {args.seq_lengths}")
    print(f"{'#'*70}\n")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks for each sequence length
    all_results = {}
    for seq_length in args.seq_lengths:
        result = benchmark_sequence_length(seq_length, args.runs, args.max_tokens)
        all_results[seq_length] = result
    
    # Save results
    output_data = {
        "model": MODEL_ID,
        "model_name": MODEL_NAME,
        "num_runs": args.runs,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {
            str(seq_len): {
                "seq_length": r.seq_length,
                "baseline": {
                    "median_prompt_tps": r.baseline_median_prompt_tps,
                    "median_gen_tps": r.baseline_median_gen_tps,
                    "median_ttft_sec": r.baseline_median_ttft_sec,
                    "peak_memory_gb": r.baseline_peak_memory_gb,
                    "runs": [asdict(run) for run in r.baseline_runs],
                },
                "patched": {
                    "median_prompt_tps": r.patched_median_prompt_tps,
                    "median_gen_tps": r.patched_median_gen_tps,
                    "median_ttft_sec": r.patched_median_ttft_sec,
                    "peak_memory_gb": r.patched_peak_memory_gb,
                    "runs": [asdict(run) for run in r.patched_runs],
                },
                "speedup": {
                    "prompt_speedup_x": r.prompt_speedup_x,
                    "gen_speedup_x": r.gen_speedup_x,
                    "ttft_delta_sec": r.ttft_delta_sec,
                    "memory_delta_gb": r.memory_delta_gb,
                }
            }
            for seq_len, r in all_results.items()
        }
    }
    
    output_file = OUTPUT_DIR / f"{MODEL_NAME.replace('-', '_').lower()}_comprehensive.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")
    
    # Final summary table
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Seq Length':<12} {'Prompt TPS':<25} {'Gen TPS':<25} {'Speedup':<15}")
    print(f"{'(tokens)':<12} {'Baseline':>12} {'Patched':>12} {'Baseline':>12} {'Patched':>12} {'Prompt':>7} {'Gen':>7}")
    print("-" * 70)
    
    for seq_len in sorted(all_results.keys()):
        r = all_results[seq_len]
        print(f"{seq_len:<12} {r.baseline_median_prompt_tps:>12.1f} {r.patched_median_prompt_tps:>12.1f} "
              f"{r.baseline_median_gen_tps:>12.1f} {r.patched_median_gen_tps:>12.1f} "
              f"{r.prompt_speedup_x:>7.2f}x {r.gen_speedup_x:>7.2f}x")
    
    # Average speedup
    avg_prompt_speedup = statistics.median(r.prompt_speedup_x for r in all_results.values())
    avg_gen_speedup = statistics.median(r.gen_speedup_x for r in all_results.values())
    
    print(f"\n{'Median Speedup Across All Lengths:':<40} {avg_prompt_speedup:>7.2f}x {avg_gen_speedup:>7.2f}x")

if __name__ == "__main__":
    main()
