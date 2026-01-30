#!/usr/bin/env python3
"""Benchmark: ZMLX-patched vs baseline inference on Llama-3.2-1B.

Tests three configurations:
  1. Baseline (no patches)
  2. FUSED_ACTIVATIONS only (SwiGLU/GeGLU — safest for inference)
  3. All patches (activations + norms + softmax)

Targets a small model (Llama-3.2-1B-Instruct-4bit) for fast iteration
on any Apple Silicon Mac.

Usage:
    python benchmarks/llama_benchmark.py
    python benchmarks/llama_benchmark.py --runs 5
    python benchmarks/llama_benchmark.py --max-tokens 100
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
MODEL_PATH = "mlx-community/Llama-3.2-1B-Instruct-4bit"
MAX_TOKENS = 200
NUM_RUNS = 3
WARMUP_TOKENS = 5
OUTPUT_DIR = Path("benchmarks/results")

# ~400 token prompt (factual, deterministic-ish)
PROMPT = (
    "You are an expert systems architect. Explain in detail how a modern "
    "distributed database handles the following challenges:\n\n"
    "1. Data partitioning and sharding strategies\n"
    "2. Consensus protocols for strong consistency\n"
    "3. Conflict resolution in multi-leader replication\n"
    "4. Read-path optimization with caching layers\n"
    "5. Failure detection and automatic failover\n\n"
    "For each topic, describe the key algorithms, trade-offs, and real-world "
    "systems that implement them. Be thorough and precise."
)

CONFIGS = [
    {"name": "baseline", "patched": False, "selective": False},
    {"name": "fused_activations", "patched": True, "selective": True},
    {"name": "all_patches", "patched": True, "selective": False},
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tokens: int = 0
    generation_tps: float = 0.0
    ttft_sec: float = 0.0
    total_time_sec: float = 0.0
    peak_memory_gb: float = 0.0


@dataclass
class ConfigResult:
    name: str = ""
    model: str = ""
    runs: list[RunMetrics] = field(default_factory=list)

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
    print(f"  Loading {model_path} ...")
    model, tokenizer = mlx_lm.load(model_path)
    return model, tokenizer


def warmup_model(model, tokenizer):
    print(f"  Warming up ({WARMUP_TOKENS} tokens) ...")
    _ = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=WARMUP_TOKENS)
    mx.eval(mx.zeros(1))


def timed_generate(model, tokenizer, prompt: str, max_tokens: int) -> RunMetrics:
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


def apply_patches(model, selective: bool = False):
    from zmlx.patch import FUSED_ACTIVATIONS
    from zmlx.patch import patch as zmlx_patch

    if selective:
        print("  Applying ZMLX patches (fused activations only) ...")
        zmlx_patch(model, patterns=FUSED_ACTIVATIONS, verbose=True)
    else:
        print("  Applying ZMLX patches (all) ...")
        zmlx_patch(model, verbose=True)


def cleanup():
    gc.collect()
    if hasattr(mx, "clear_memory_cache"):
        mx.clear_memory_cache()


# ---------------------------------------------------------------------------
# Run one config
# ---------------------------------------------------------------------------
def bench_config(
    config: dict,
    model_path: str,
    num_runs: int,
    max_tokens: int,
    prompt: str,
) -> ConfigResult:
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"  Config: {name}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_path)

    if config["patched"]:
        apply_patches(model, selective=config["selective"])

    warmup_model(model, tokenizer)

    result = ConfigResult(name=name, model=model_path)

    for i in range(num_runs):
        print(f"\n  Run {i + 1}/{num_runs} ...")
        metrics = timed_generate(model, tokenizer, prompt, max_tokens)
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

    del model, tokenizer
    cleanup()

    return result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def print_comparison(results: dict[str, ConfigResult]):
    baseline = results.get("baseline")
    if baseline is None or baseline.median_gen_tps == 0:
        print("  (no baseline to compare against)")
        return

    print(f"\n{'Config':<25} {'Prompt':>10} {'Gen':>10} {'TTFT':>10} {'Mem':>10}")
    print("-" * 65)
    for name, res in results.items():
        if name == "baseline":
            print(
                f"{name:<25} "
                f"{res.median_prompt_tps:>9.1f}t "
                f"{res.median_gen_tps:>9.1f}t "
                f"{res.median_ttft_sec:>9.3f}s "
                f"{res.peak_memory_gb:>9.2f}G"
            )
        else:
            prompt_x = res.median_prompt_tps / baseline.median_prompt_tps if baseline.median_prompt_tps > 0 else 0
            gen_x = res.median_gen_tps / baseline.median_gen_tps if baseline.median_gen_tps > 0 else 0
            ttft_d = res.median_ttft_sec - baseline.median_ttft_sec
            mem_d = res.peak_memory_gb - baseline.peak_memory_gb
            print(
                f"{name:<25} "
                f"{prompt_x:>8.2f}x "
                f"{gen_x:>8.2f}x "
                f"{ttft_d:>+9.3f}s "
                f"{mem_d:>+9.2f}G"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ZMLX Llama-3.2-1B inference benchmark")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Runs per config")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="Tokens to generate")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Config names to run (baseline, fused_activations, all_patches)",
    )
    args = parser.parse_args()

    prompt = args.prompt or PROMPT
    selected = CONFIGS
    if args.configs:
        selected = [c for c in CONFIGS if c["name"] in args.configs]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, ConfigResult] = {}

    print(f"Model: {MODEL_PATH}")
    print(f"Runs:  {args.runs} | Max tokens: {args.max_tokens}")
    print(f"Configs: {[c['name'] for c in selected]}")

    for config in selected:
        results[config["name"]] = bench_config(
            config=config,
            model_path=MODEL_PATH,
            num_runs=args.runs,
            max_tokens=args.max_tokens,
            prompt=prompt,
        )

    # Comparison
    print(f"\n{'='*60}")
    print("  RESULTS COMPARISON")
    print(f"{'='*60}")
    print_comparison(results)

    # Save JSON
    output = {
        "model": MODEL_PATH,
        "num_runs": args.runs,
        "max_tokens": args.max_tokens,
    }
    for name, res in results.items():
        output[name] = {
            "median_prompt_tps": res.median_prompt_tps,
            "median_gen_tps": res.median_gen_tps,
            "median_ttft_sec": res.median_ttft_sec,
            "median_total_sec": res.median_total_sec,
            "peak_memory_gb": res.peak_memory_gb,
            "runs": [asdict(r) for r in res.runs],
        }

    if len(results) > 1 and "baseline" in results:
        baseline = results["baseline"]
        comparisons = {}
        for name, res in results.items():
            if name == "baseline":
                continue
            comparisons[name] = {
                "prompt_speedup_x": round(
                    res.median_prompt_tps / baseline.median_prompt_tps, 3
                ) if baseline.median_prompt_tps > 0 else 0,
                "gen_speedup_x": round(
                    res.median_gen_tps / baseline.median_gen_tps, 3
                ) if baseline.median_gen_tps > 0 else 0,
                "ttft_delta_sec": round(
                    res.median_ttft_sec - baseline.median_ttft_sec, 4
                ),
                "memory_delta_gb": round(
                    res.peak_memory_gb - baseline.peak_memory_gb, 3
                ),
            }
        output["comparisons"] = comparisons

    results_path = OUTPUT_DIR / "llama_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
