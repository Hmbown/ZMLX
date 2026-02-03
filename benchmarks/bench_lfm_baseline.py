#!/usr/bin/env python3
"""Baseline benchmark for LFM models (moonshot dequant fusion track).

Runs baseline (unpatched) and patched variants for a small set of LFM models
using mlx-lm stream generation metrics. Results are written under
benchmarks/results/ for comparison in later phases.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx_lm

from zmlx.patch import patch


OUTPUT_DIR = Path("benchmarks/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_QSWIGLU_ENV = "ZMLX_FUSED_QSWIGLU"


@dataclass
class RunMetrics:
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tokens: int = 0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    runtime_sec: float = 0.0


@dataclass
class BenchmarkMetrics:
    model: str
    patched: bool
    runs: list[RunMetrics]

    @property
    def summary(self) -> dict[str, float]:
        if not self.runs:
            return {}
        prompt_tps = sorted(r.prompt_tps for r in self.runs)
        gen_tps = sorted(r.generation_tps for r in self.runs)
        peak_mem = max(r.peak_memory_gb for r in self.runs)
        return {
            "median_prompt_tps": prompt_tps[len(prompt_tps) // 2],
            "median_generation_tps": gen_tps[len(gen_tps) // 2],
            "peak_memory_gb": peak_mem,
        }


def _clear_gpu() -> None:
    gc.collect()
    if hasattr(mx, "clear_memory_cache"):
        mx.clear_memory_cache()
    if hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _run_once(model, tokenizer, *, prompt: str, max_tokens: int) -> RunMetrics:
    start = time.time()
    response = None
    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        response = resp
    runtime = time.time() - start

    if response is None:
        return RunMetrics(runtime_sec=runtime)

    return RunMetrics(
        prompt_tokens=getattr(response, "prompt_tokens", 0),
        prompt_tps=getattr(response, "prompt_tps", 0.0),
        generation_tokens=getattr(response, "generation_tokens", 0),
        generation_tps=getattr(response, "generation_tps", 0.0),
        peak_memory_gb=getattr(response, "peak_memory", 0.0),
        runtime_sec=runtime,
    )


def run_benchmark(
    model_id: str,
    *,
    patched: bool,
    max_tokens: int,
    runs: int,
    prompt: str,
    enable_fused_qswiglu: bool,
) -> BenchmarkMetrics:
    print(f"\n{'=' * 64}")
    print(f"Model:   {model_id}")
    print(f"Patched: {patched}")
    print(f"{'=' * 64}")

    print("Loading model...")
    model, tokenizer = mlx_lm.load(model_id)

    prev_env = os.environ.get(_QSWIGLU_ENV)
    if enable_fused_qswiglu:
        os.environ[_QSWIGLU_ENV] = "1"
    else:
        os.environ.pop(_QSWIGLU_ENV, None)

    if patched:
        print("Applying ZMLX patches...")
        patch(model, verbose=False)

    print("Warmup...")
    _ = mlx_lm.generate(model, tokenizer, prompt="Hi", max_tokens=5)
    mx.eval(mx.zeros(1))

    metrics: list[RunMetrics] = []
    for i in range(runs):
        print(f"Run {i + 1}/{runs}...")
        metrics.append(_run_once(model, tokenizer, prompt=prompt, max_tokens=max_tokens))

    if prev_env is None:
        os.environ.pop(_QSWIGLU_ENV, None)
    else:
        os.environ[_QSWIGLU_ENV] = prev_env

    _clear_gpu()
    return BenchmarkMetrics(model=model_id, patched=patched, runs=metrics)


def _summarize_pair(baseline: BenchmarkMetrics, patched: BenchmarkMetrics) -> dict[str, float]:
    base = baseline.summary
    mod = patched.summary
    prompt_speedup = (mod.get("median_prompt_tps", 0.0) / base.get("median_prompt_tps", 1.0)) if base else 0.0
    gen_speedup = (mod.get("median_generation_tps", 0.0) / base.get("median_generation_tps", 1.0)) if base else 0.0
    return {
        "prompt_speedup_x": round(prompt_speedup, 3) if prompt_speedup else 0.0,
        "generation_speedup_x": round(gen_speedup, 3) if gen_speedup else 0.0,
        "baseline_prompt_tps": round(base.get("median_prompt_tps", 0.0), 2),
        "patched_prompt_tps": round(mod.get("median_prompt_tps", 0.0), 2),
        "baseline_generation_tps": round(base.get("median_generation_tps", 0.0), 2),
        "patched_generation_tps": round(mod.get("median_generation_tps", 0.0), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--enable-fused-qswiglu",
        action="store_true",
        help="Enable fused quantized SwiGLU (decode-only experimental path).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Summarize why mixture-of-experts models can be faster at decode. "
            "Mention routing and activation sparsity."
        ),
    )
    args = parser.parse_args()

    models = {
        "lfm2.5-1.2b": "mlx-community/LFM2.5-1.2B-Instruct-4bit",
        "lfm2-8b-a1b": "mlx-community/LFM2-8B-A1B-4bit",
    }

    results: dict[str, dict[str, object]] = {}
    for name, model_id in models.items():
        baseline = run_benchmark(
            model_id,
            patched=False,
            max_tokens=args.max_tokens,
            runs=args.runs,
            prompt=args.prompt,
            enable_fused_qswiglu=False,
        )
        patched = run_benchmark(
            model_id,
            patched=True,
            max_tokens=args.max_tokens,
            runs=args.runs,
            prompt=args.prompt,
            enable_fused_qswiglu=args.enable_fused_qswiglu,
        )
        results[name] = {
            "model_id": model_id,
            "baseline": [asdict(r) for r in baseline.runs],
            "patched": [asdict(r) for r in patched.runs],
            "summary": _summarize_pair(baseline, patched),
        }

        summary = results[name]["summary"]
        print("\nSummary:")
        print(f"  Prompt:   {summary['baseline_prompt_tps']} -> {summary['patched_prompt_tps']} tok/s")
        print(
            f"  Generate: {summary['baseline_generation_tps']} -> "
            f"{summary['patched_generation_tps']} tok/s"
        )

    output_file = OUTPUT_DIR / "lfm_baseline_results.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
