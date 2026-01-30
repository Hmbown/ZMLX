#!/usr/bin/env python3
"""Benchmark: ZMLX-patched vs baseline inference on Apple Silicon.

Compares prompt evaluation (prefill) and token generation (decode) speed
with and without ZMLX kernel patches. Inference is memory-bandwidth-bound,
so fused kernels that eliminate memory round-trips should show measurable
speedups — especially for larger models and autoregressive decode.

Usage:
    python benchmarks/inference_benchmark.py                    # default: Qwen3-8B-4bit, FUSED_ACTIVATIONS
    python benchmarks/inference_benchmark.py --models all       # run all models
    python benchmarks/inference_benchmark.py --runs 5           # more runs for stability
    python benchmarks/inference_benchmark.py --all-patterns     # use ALL_PATTERNS (norms too, known regression)
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
MODELS = {
    "qwen3-8b": "mlx-community/Qwen3-8B-4bit",
    "qwen3-8b-bf16": "mlx-community/Qwen3-8B-bf16",
    "qwen3-30b-a3b": "mlx-community/Qwen3-30B-A3B-4bit",
    "qwen3-30b-a3b-instruct": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "qwen3-32b": "mlx-community/Qwen3-32B-4bit",
    "glm-4.7-flash": "mlx-community/GLM-4.7-Flash-4bit",
    "gemma3-27b": "mlx-community/gemma-3-27b-it-qat-4bit",
    "deepseek-r1-32b": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
}
DEFAULT_MODEL = "qwen3-8b"

MAX_TOKENS = 1500
NUM_RUNS = 3
OUTPUT_BASE = Path("benchmarks/results")

# Substantial prompt (~600 tokens) — forces real prefill work
PROMPT = (
    "You are an expert systems architect with deep knowledge of distributed "
    "systems, compiler design, and machine learning infrastructure. I need a "
    "comprehensive technical analysis covering the following topics in detail.\n\n"
    "## Part 1: Distributed Database Internals\n"
    "Explain how a modern distributed database handles data partitioning and "
    "sharding strategies (consistent hashing, range-based, directory-based), "
    "consensus protocols for strong consistency (Raft, Multi-Paxos, EPaxos), "
    "conflict resolution in multi-leader replication (CRDTs, last-writer-wins, "
    "application-level merge), read-path optimization with caching layers, and "
    "failure detection with automatic failover.\n\n"
    "## Part 2: Compiler Optimization Passes\n"
    "Describe the key optimization passes in a modern optimizing compiler: "
    "SSA construction, dead code elimination, constant propagation, loop "
    "invariant code motion, strength reduction, register allocation via graph "
    "coloring, and instruction scheduling. For each pass, explain the algorithm, "
    "its time complexity, and how it interacts with other passes in the pipeline.\n\n"
    "## Part 3: ML Training Infrastructure\n"
    "Explain how large-scale ML training systems handle distributed data "
    "parallelism, model parallelism (tensor, pipeline, expert), gradient "
    "accumulation and synchronization (all-reduce, ring-allreduce, parameter "
    "server), mixed-precision training (FP16/BF16 with loss scaling), and "
    "checkpointing strategies for fault tolerance.\n\n"
    "For each topic, describe the key algorithms, trade-offs, failure modes, "
    "and real-world systems that implement them. Be thorough, precise, and "
    "provide concrete examples with numbers where possible."
)


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
class BenchmarkResult:
    """Aggregated results across multiple runs."""

    label: str = ""
    model: str = ""
    patched: bool = False
    runs: list[RunMetrics] = field(default_factory=list)

    # Aggregated (filled in after runs complete)
    median_prompt_tps: float = 0.0
    median_gen_tps: float = 0.0
    median_ttft_sec: float = 0.0
    median_total_sec: float = 0.0
    peak_memory_gb: float = 0.0

    def aggregate(self):
        if not self.runs:
            return
        self.median_prompt_tps = statistics.median(r.prompt_tps for r in self.runs)
        self.median_gen_tps = statistics.median(r.generation_tps for r in self.runs)
        self.median_ttft_sec = statistics.median(r.ttft_sec for r in self.runs)
        self.median_total_sec = statistics.median(r.total_time_sec for r in self.runs)
        self.peak_memory_gb = max(r.peak_memory_gb for r in self.runs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(model_path: str):
    """Load model + tokenizer from HuggingFace/mlx-community."""
    print(f"  Loading {model_path} ...")
    model, tokenizer = mlx_lm.load(model_path)
    return model, tokenizer


def warmup(model, tokenizer):
    """Run a short generation to compile Metal shaders before timing."""
    print("  Warming up (compiling Metal shaders) ...")
    _ = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=5)
    mx.eval(mx.zeros(1))  # sync


def timed_generate(model, tokenizer, prompt: str, max_tokens: int) -> RunMetrics:
    """Run one generation pass and collect timing metrics."""
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


def apply_patches(model, selective: bool = True):
    """Apply ZMLX kernel patches.

    Args:
        selective: If True (default), only apply fused-activation patterns
            (SwiGLU/GeGLU/MoE) — matches ``patch(model)`` default.
            If False, apply ALL patterns (including norms, which cause
            3–5% decode regression on all tested models).
    """
    from zmlx.patch import ALL_PATTERNS
    from zmlx.patch import patch as zmlx_patch

    if selective:
        print("  Applying ZMLX patches (default: fused activations only) ...")
        zmlx_patch(model, verbose=True)  # default is FUSED_ACTIVATIONS
    else:
        print("  Applying ZMLX patches (all patterns) ...")
        zmlx_patch(model, patterns=ALL_PATTERNS, verbose=True)


# ---------------------------------------------------------------------------
# Run benchmark for one configuration
# ---------------------------------------------------------------------------
def bench_config(
    model_path: str,
    label: str,
    patched: bool,
    num_runs: int,
    max_tokens: int,
    selective: bool = False,
) -> BenchmarkResult:
    """Benchmark one model configuration (baseline or patched)."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  model={model_path}  patched={patched}  selective={selective}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_path)

    if patched:
        apply_patches(model, selective=selective)

    warmup(model, tokenizer)

    result = BenchmarkResult(label=label, model=model_path, patched=patched)

    for i in range(num_runs):
        print(f"\n  Run {i + 1}/{num_runs} ...")
        metrics = timed_generate(model, tokenizer, PROMPT, max_tokens)
        result.runs.append(metrics)
        print(
            f"    prompt: {metrics.prompt_tps:.1f} tok/s  |  "
            f"gen: {metrics.generation_tps:.1f} tok/s  |  "
            f"TTFT: {metrics.ttft_sec:.3f}s  |  "
            f"mem: {metrics.peak_memory_gb:.2f} GB"
        )

    result.aggregate()

    print(f"\n  Median ({num_runs} runs):")
    print(f"    Prompt:     {result.median_prompt_tps:.1f} tok/s")
    print(f"    Generation: {result.median_gen_tps:.1f} tok/s")
    print(f"    TTFT:       {result.median_ttft_sec:.3f}s")
    print(f"    Peak mem:   {result.peak_memory_gb:.2f} GB")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if hasattr(mx, "clear_memory_cache"):
        mx.clear_memory_cache()

    return result


# ---------------------------------------------------------------------------
# Compare and report
# ---------------------------------------------------------------------------
def compare(baseline: BenchmarkResult, patched: BenchmarkResult) -> dict:
    """Print comparison table and return summary dict."""
    prompt_speedup = (
        patched.median_prompt_tps / baseline.median_prompt_tps
        if baseline.median_prompt_tps > 0
        else 0
    )
    gen_speedup = (
        patched.median_gen_tps / baseline.median_gen_tps
        if baseline.median_gen_tps > 0
        else 0
    )
    ttft_delta = patched.median_ttft_sec - baseline.median_ttft_sec
    mem_delta = patched.peak_memory_gb - baseline.peak_memory_gb

    print(f"\n{'Metric':<30} {'Baseline':>12} {'ZMLX':>12} {'Delta':>12}")
    print("-" * 66)
    print(
        f"{'Prompt eval (tok/s)':<30} "
        f"{baseline.median_prompt_tps:>12.1f} "
        f"{patched.median_prompt_tps:>12.1f} "
        f"{prompt_speedup:>11.2f}x"
    )
    print(
        f"{'Token gen (tok/s)':<30} "
        f"{baseline.median_gen_tps:>12.1f} "
        f"{patched.median_gen_tps:>12.1f} "
        f"{gen_speedup:>11.2f}x"
    )
    print(
        f"{'TTFT (sec)':<30} "
        f"{baseline.median_ttft_sec:>12.3f} "
        f"{patched.median_ttft_sec:>12.3f} "
        f"{ttft_delta:>+12.3f}"
    )
    print(
        f"{'Peak memory (GB)':<30} "
        f"{baseline.peak_memory_gb:>12.2f} "
        f"{patched.peak_memory_gb:>12.2f} "
        f"{mem_delta:>+12.2f}"
    )

    return {
        "prompt_speedup_x": round(prompt_speedup, 3),
        "gen_speedup_x": round(gen_speedup, 3),
        "ttft_delta_sec": round(ttft_delta, 4),
        "memory_delta_gb": round(mem_delta, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ZMLX inference benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL],
        help=f"Model keys to test: {', '.join(MODELS)} or 'all'",
    )
    parser.add_argument(
        "--runs", type=int, default=NUM_RUNS, help="Number of runs per config"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS, help="Tokens to generate"
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="Custom prompt (overrides default)"
    )
    parser.add_argument(
        "--all-patterns",
        action="store_true",
        default=False,
        help="Apply ALL patterns (norms/softmax too). Default uses FUSED_ACTIVATIONS only.",
    )
    args = parser.parse_args()

    global PROMPT
    if args.prompt:
        PROMPT = args.prompt

    # Default is FUSED_ACTIVATIONS (matches patch(model) default).
    # --all-patterns opts into ALL_PATTERNS (norms/softmax too — known 3-5% regression).
    selective = not args.all_patterns

    model_keys = list(MODELS.keys()) if "all" in args.models else args.models
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for key in model_keys:
        if key not in MODELS:
            print(f"Unknown model key: {key}. Available: {', '.join(MODELS)}")
            continue

        model_path = MODELS[key]
        print(f"\n{'#'*60}")
        print(f"# Model: {key} ({model_path})")
        print(f"{'#'*60}")

        # Baseline
        baseline = bench_config(
            model_path=model_path,
            label=f"Baseline ({key})",
            patched=False,
            num_runs=args.runs,
            max_tokens=args.max_tokens,
        )

        # ZMLX patched
        patched = bench_config(
            model_path=model_path,
            label=f"ZMLX ({key})",
            patched=True,
            num_runs=args.runs,
            max_tokens=args.max_tokens,
            selective=selective,
        )

        # Compare
        print(f"\n{'='*60}")
        print(f"  RESULTS: {key}")
        print(f"{'='*60}")
        comparison = compare(baseline, patched)

        all_results[key] = {
            "model": model_path,
            "num_runs": args.runs,
            "max_tokens": args.max_tokens,
            "baseline": {
                "median_prompt_tps": baseline.median_prompt_tps,
                "median_gen_tps": baseline.median_gen_tps,
                "median_ttft_sec": baseline.median_ttft_sec,
                "median_total_sec": baseline.median_total_sec,
                "peak_memory_gb": baseline.peak_memory_gb,
                "runs": [asdict(r) for r in baseline.runs],
            },
            "zmlx": {
                "median_prompt_tps": patched.median_prompt_tps,
                "median_gen_tps": patched.median_gen_tps,
                "median_ttft_sec": patched.median_ttft_sec,
                "median_total_sec": patched.median_total_sec,
                "peak_memory_gb": patched.peak_memory_gb,
                "runs": [asdict(r) for r in patched.runs],
            },
            "comparison": comparison,
        }

    # Save all results
    results_path = OUTPUT_BASE / "inference_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Final summary across all models
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  CROSS-MODEL SUMMARY")
        print(f"{'='*60}")
        print(f"\n{'Model':<20} {'Prompt':>12} {'Gen':>12}")
        print("-" * 44)
        for key, res in all_results.items():
            c = res["comparison"]
            print(f"{key:<20} {c['prompt_speedup_x']:>11.2f}x {c['gen_speedup_x']:>11.2f}x")


if __name__ == "__main__":
    main()
