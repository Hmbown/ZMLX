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
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx_lm

from zmlx.device import detect_device
from zmlx.patch import patch

OUTPUT_DIR = Path("benchmarks/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_QSWIGLU_ENV = "ZMLX_FUSED_QSWIGLU"
_QSWIGLU_MODE_ENV = "ZMLX_FUSED_QSWIGLU_MODE"
_QSWIGLU_PROGRESSIVE_ENV = "ZMLX_FUSED_QSWIGLU_PROGRESSIVE"
_QSWIGLU_EPS_ENV = "ZMLX_FUSED_QSWIGLU_EPS"
_QSWIGLU_TG_ENV = "ZMLX_FUSED_QSWIGLU_TG"
_QSWIGLU_PER_GROUP_ENV = "ZMLX_FUSED_QSWIGLU_PER_GROUP"
_QSWIGLU_MAX_TOKENS_ENV = "ZMLX_FUSED_QSWIGLU_MAX_TOKENS"
_QSWIGLU_MAX_OUT_ENV = "ZMLX_FUSED_QSWIGLU_MAX_OUT"
_QSWIGLU_MAX_IN_ENV = "ZMLX_FUSED_QSWIGLU_MAX_IN"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _device_meta() -> dict[str, object]:
    device = detect_device()
    return {
        "family": device.family,
        "variant": device.variant,
        "gpu_cores": device.gpu_cores,
        "memory_gb": device.memory_gb,
        "has_ray_tracing": device.has_ray_tracing,
    }


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
    qswiglu_mode: str,
    qswiglu_progressive: bool,
    qswiglu_eps: float,
    qswiglu_tg: int,
    qswiglu_per_group: bool,
    qswiglu_max_tokens: int,
    qswiglu_max_out: int,
    qswiglu_max_in: int,
) -> BenchmarkMetrics:
    print(f"\n{'=' * 64}")
    print(f"Model:   {model_id}")
    print(f"Patched: {patched}")
    print(f"{'=' * 64}")

    print("Loading model...")
    model, tokenizer = mlx_lm.load(model_id)

    prev_env = {
        _QSWIGLU_ENV: os.environ.get(_QSWIGLU_ENV),
        _QSWIGLU_MODE_ENV: os.environ.get(_QSWIGLU_MODE_ENV),
        _QSWIGLU_PROGRESSIVE_ENV: os.environ.get(_QSWIGLU_PROGRESSIVE_ENV),
        _QSWIGLU_EPS_ENV: os.environ.get(_QSWIGLU_EPS_ENV),
        _QSWIGLU_TG_ENV: os.environ.get(_QSWIGLU_TG_ENV),
        _QSWIGLU_PER_GROUP_ENV: os.environ.get(_QSWIGLU_PER_GROUP_ENV),
        _QSWIGLU_MAX_TOKENS_ENV: os.environ.get(_QSWIGLU_MAX_TOKENS_ENV),
        _QSWIGLU_MAX_OUT_ENV: os.environ.get(_QSWIGLU_MAX_OUT_ENV),
        _QSWIGLU_MAX_IN_ENV: os.environ.get(_QSWIGLU_MAX_IN_ENV),
    }

    if qswiglu_mode == "force":
        os.environ[_QSWIGLU_ENV] = "1"
        os.environ.pop(_QSWIGLU_MODE_ENV, None)
    else:
        os.environ.pop(_QSWIGLU_ENV, None)
        os.environ[_QSWIGLU_MODE_ENV] = qswiglu_mode
    if qswiglu_progressive:
        os.environ[_QSWIGLU_PROGRESSIVE_ENV] = "1"
    else:
        os.environ.pop(_QSWIGLU_PROGRESSIVE_ENV, None)
    os.environ[_QSWIGLU_EPS_ENV] = str(qswiglu_eps)
    os.environ[_QSWIGLU_TG_ENV] = str(qswiglu_tg)
    if qswiglu_per_group:
        os.environ[_QSWIGLU_PER_GROUP_ENV] = "1"
    else:
        os.environ.pop(_QSWIGLU_PER_GROUP_ENV, None)
    os.environ[_QSWIGLU_MAX_TOKENS_ENV] = str(qswiglu_max_tokens)
    os.environ[_QSWIGLU_MAX_OUT_ENV] = str(qswiglu_max_out)
    os.environ[_QSWIGLU_MAX_IN_ENV] = str(qswiglu_max_in)

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

    for key, value in prev_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

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
        "--fused-qswiglu-mode",
        type=str,
        default="off",
        choices=("off", "auto", "force"),
        help="Quantized SwiGLU mode: off, auto, or force.",
    )
    parser.add_argument(
        "--fused-qswiglu-progressive",
        action="store_true",
        help="Enable progressive quantized SwiGLU refinement.",
    )
    parser.add_argument(
        "--fused-qswiglu-eps",
        type=float,
        default=0.0,
        help="Progressive refinement epsilon (lower = more refinement).",
    )
    parser.add_argument(
        "--fused-qswiglu-eps-sweep",
        type=str,
        default="",
        help="Comma-separated eps sweep (overrides --fused-qswiglu-eps).",
    )
    parser.add_argument(
        "--fused-qswiglu-tg",
        type=int,
        default=0,
        help="Threadgroup size for progressive quantized SwiGLU (0=off).",
    )
    parser.add_argument(
        "--fused-qswiglu-per-group",
        action="store_true",
        help="Use per-group progressive refinement (non-threadgroup only).",
    )
    parser.add_argument(
        "--fused-qswiglu-max-tokens",
        type=int,
        default=1,
        help="Max tokens to allow fused quantized SwiGLU (decode-like).",
    )
    parser.add_argument(
        "--fused-qswiglu-max-out",
        type=int,
        default=2048,
        help="Max output features for auto fused quantized SwiGLU.",
    )
    parser.add_argument(
        "--fused-qswiglu-max-in",
        type=int,
        default=2048,
        help="Max input features for auto fused quantized SwiGLU.",
    )
    parser.add_argument(
        "--fused-qswiglu-bits",
        type=int,
        default=4,
        help="Quantization bits for logging metadata (default: 4).",
    )
    parser.add_argument(
        "--fused-qswiglu-group-size",
        type=int,
        default=64,
        help="Quantization group size for logging metadata (default: 64).",
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

    eps_list = []
    if args.fused_qswiglu_eps_sweep:
        eps_list = [float(item.strip()) for item in args.fused_qswiglu_eps_sweep.split(",") if item.strip()]
    if not eps_list:
        eps_list = [args.fused_qswiglu_eps]

    qswiglu_config = {
        "mode": args.fused_qswiglu_mode,
        "progressive": args.fused_qswiglu_progressive,
        "threadgroup": args.fused_qswiglu_tg,
        "per_group": args.fused_qswiglu_per_group,
        "max_tokens": args.fused_qswiglu_max_tokens,
        "max_out": args.fused_qswiglu_max_out,
        "max_in": args.fused_qswiglu_max_in,
        "bits": args.fused_qswiglu_bits,
        "group_size": args.fused_qswiglu_group_size,
    }
    meta_common = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mlx_version": mx.__version__,
        "git_sha": _git_sha(),
        "device": _device_meta(),
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "qswiglu": qswiglu_config,
    }

    results: dict[str, object] = {"meta": meta_common, "models": {}}
    for name, model_id in models.items():
        baseline = run_benchmark(
            model_id,
            patched=False,
            max_tokens=args.max_tokens,
            runs=args.runs,
            prompt=args.prompt,
            qswiglu_mode="off",
            qswiglu_progressive=False,
            qswiglu_eps=0.0,
            qswiglu_tg=0,
            qswiglu_per_group=False,
            qswiglu_max_tokens=args.fused_qswiglu_max_tokens,
            qswiglu_max_out=args.fused_qswiglu_max_out,
            qswiglu_max_in=args.fused_qswiglu_max_in,
        )
        patched_entries: list[dict[str, object]] = []
        for eps in eps_list:
            patched = run_benchmark(
                model_id,
                patched=True,
                max_tokens=args.max_tokens,
                runs=args.runs,
                prompt=args.prompt,
                qswiglu_mode=args.fused_qswiglu_mode,
                qswiglu_progressive=args.fused_qswiglu_progressive,
                qswiglu_eps=eps,
                qswiglu_tg=args.fused_qswiglu_tg,
                qswiglu_per_group=args.fused_qswiglu_per_group,
                qswiglu_max_tokens=args.fused_qswiglu_max_tokens,
                qswiglu_max_out=args.fused_qswiglu_max_out,
                qswiglu_max_in=args.fused_qswiglu_max_in,
            )
            summary = _summarize_pair(baseline, patched)
            entry_config = dict(qswiglu_config)
            entry_config["eps"] = eps
            patched_entries.append(
                {
                    "eps": eps,
                    "config": entry_config,
                    "runs": [asdict(r) for r in patched.runs],
                    "summary": summary,
                }
            )

            print("\nSummary:")
            print(f"  eps={eps:g} Prompt:   {summary['baseline_prompt_tps']} -> {summary['patched_prompt_tps']} tok/s")
            print(
                f"  eps={eps:g} Generate: {summary['baseline_generation_tps']} -> "
                f"{summary['patched_generation_tps']} tok/s"
            )

        results["models"][name] = {
            "model_id": model_id,
            "baseline": {"runs": [asdict(r) for r in baseline.runs], "summary": baseline.summary},
            "patched": patched_entries,
        }

    output_file = OUTPUT_DIR / "lfm_baseline_results.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
