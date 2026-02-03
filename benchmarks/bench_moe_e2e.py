#!/usr/bin/env python3
"""End-to-end MoE inference benchmark: baseline vs ZMLX patches.

Compares three configurations on a quantized MoE model:
1. **Baseline** — vanilla mlx_lm.generate() (no patches)
2. **ZMLX v0.6.x** — existing MoE patch (fused gating + combine only)
3. **ZMLX + fused SwiGLU** — new patch with gather_qmm_swiglu for expert projections

Usage:
    python benchmarks/bench_moe_e2e.py
    python benchmarks/bench_moe_e2e.py --model mlx-community/Qwen3-30B-A3B-4bit
    python benchmarks/bench_moe_e2e.py --runs 5
    python benchmarks/bench_moe_e2e.py --max-tokens 500
    python benchmarks/bench_moe_e2e.py --json-out .benchmarks/qwen3_a3b.json
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import mlx.core as mx

try:
    from zmlx.device import detect_device
except Exception:
    detect_device = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
OUTPUT_DIR = Path("benchmarks/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = (
    "Explain the key differences between mixture-of-experts (MoE) and dense "
    "transformer architectures. Cover parameter efficiency, routing strategies, "
    "load balancing, and inference characteristics. Be detailed and thorough."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _device_meta() -> dict[str, object]:
    if detect_device is None:
        return {}
    device = detect_device()
    return {
        "family": device.family,
        "variant": device.variant,
        "gpu_cores": device.gpu_cores,
        "memory_gb": device.memory_gb,
        "has_ray_tracing": device.has_ray_tracing,
    }


def _infer_quant_meta(model_path: str, config: dict | None) -> dict[str, object]:
    bits = None
    group_size = None
    mode = None
    if isinstance(config, dict):
        quant = config.get("quantization") or config.get("quantization_config")
        if isinstance(quant, dict):
            bits = quant.get("bits", bits)
            group_size = quant.get("group_size", group_size)
            mode = quant.get("mode", quant.get("quantization_mode", mode))
    match = re.search(r"-(\d)bit", model_path)
    if match and bits is None:
        bits = int(match.group(1))
    match = re.search(r"(?:g|gs)(\d+)", model_path)
    if match and group_size is None:
        group_size = int(match.group(1))
    return {"bits": bits, "group_size": group_size, "mode": mode}


def load_model(model_path: str, *, return_config: bool = False):
    """Load model + tokenizer."""
    import mlx_lm
    print(f"  Loading {model_path} ...")
    if return_config:
        model, tokenizer, config = mlx_lm.load(model_path, return_config=True)
        return model, tokenizer, config
    model, tokenizer = mlx_lm.load(model_path)
    return model, tokenizer


def warmup(model, tokenizer, n: int = 2):
    """Warmup generations to compile Metal shaders."""
    import mlx_lm
    print("  Warming up ...")
    for _ in range(n):
        _ = mlx_lm.generate(model, tokenizer, prompt="Hi", max_tokens=5)
    mx.eval(mx.zeros(1))
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()


def timed_generate(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    """Run one generation and return timing metrics."""
    import mlx_lm

    wall_start = time.perf_counter()
    last_response = None

    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens
    ):
        last_response = response

    total_time = time.perf_counter() - wall_start

    if last_response is None:
        return {}

    return {
        "prompt_tokens": last_response.prompt_tokens,
        "prompt_tps": last_response.prompt_tps,
        "gen_tokens": last_response.generation_tokens,
        "gen_tps": last_response.generation_tps,
        "peak_memory_gb": last_response.peak_memory,
        "total_sec": total_time,
    }


def bench_config(
    model, tokenizer, label: str, num_runs: int, max_tokens: int
) -> dict:
    """Benchmark a single configuration, return aggregated metrics."""
    print(f"\n  --- {label} ---")
    warmup(model, tokenizer)

    runs = []
    for i in range(num_runs):
        metrics = timed_generate(model, tokenizer, PROMPT, max_tokens)
        runs.append(metrics)
        print(
            f"    Run {i + 1}/{num_runs}: "
            f"prompt={metrics.get('prompt_tps', 0):.1f} tok/s, "
            f"gen={metrics.get('gen_tps', 0):.1f} tok/s, "
            f"mem={metrics.get('peak_memory_gb', 0):.2f} GB"
        )

    if not runs:
        return {"label": label, "runs": []}

    return {
        "label": label,
        "runs": runs,
        "median_prompt_tps": statistics.median(r["prompt_tps"] for r in runs),
        "median_gen_tps": statistics.median(r["gen_tps"] for r in runs),
        "peak_memory_gb": max(r.get("peak_memory_gb", 0) for r in runs),
        "prompt_tokens": runs[0].get("prompt_tokens", 0),
        "gen_tokens": runs[0].get("gen_tokens", 0),
    }


def apply_zmlx_patch(
    model,
    *,
    disable_fused_swiglu: bool = False,
    fused_max_tokens: int | None = None,
    verbose: bool = True,
):
    """Apply ZMLX patches to the model.

    Args:
        disable_fused_swiglu: If True, temporarily disable the fused SwiGLU path
            to benchmark the existing MoE patch without it.
        verbose: Print patch details.
    """
    from zmlx.patch import patch as zmlx_patch

    if disable_fused_swiglu:
        # Temporarily override has_gather_qmm_swiglu to return False
        from zmlx.kernels import fused_moe
        original = fused_moe._HAS_GATHER_QMM_SWIGLU
        fused_moe._HAS_GATHER_QMM_SWIGLU = False
        try:
            zmlx_patch(model, moe_fused_swiglu_max_tokens=fused_max_tokens, verbose=verbose)
        finally:
            fused_moe._HAS_GATHER_QMM_SWIGLU = original
    else:
        zmlx_patch(model, moe_fused_swiglu_max_tokens=fused_max_tokens, verbose=verbose)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MoE E2E inference benchmark")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model path (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per config (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens to generate")
    parser.add_argument(
        "--fused-max-tokens",
        type=int,
        default=None,
        help="Override max token count for fused SwiGLU path",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Write summary JSON to this path (optional)",
    )
    args = parser.parse_args()

    has_fused = hasattr(mx, "gather_qmm_swiglu")
    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print(f"gather_qmm_swiglu available: {has_fused}")
    print(f"Model: {args.model}")
    print(f"Runs per config: {args.runs}")
    print(f"Max tokens: {args.max_tokens}")

    results = []

    # ------------------------------------------------------------------
    # Config 1: Baseline (no patches)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("CONFIG 1: Baseline (vanilla mlx_lm)")
    print(f"{'=' * 60}")
    model, tokenizer, model_config = load_model(args.model, return_config=True)
    r1 = bench_config(model, tokenizer, "Baseline", args.runs, args.max_tokens)
    results.append(r1)
    del model, tokenizer
    gc.collect()
    if hasattr(mx, "clear_memory_cache"):
        mx.clear_memory_cache()

    # ------------------------------------------------------------------
    # Config 2: ZMLX patch without fused SwiGLU (gating + combine only)
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("CONFIG 2: ZMLX patch (fused gating + combine, no fused SwiGLU)")
    print(f"{'=' * 60}")
    model, tokenizer = load_model(args.model)
    apply_zmlx_patch(
        model,
        disable_fused_swiglu=True,
        fused_max_tokens=args.fused_max_tokens,
    )
    r2 = bench_config(
        model, tokenizer, "ZMLX (gating+combine)", args.runs, args.max_tokens
    )
    results.append(r2)
    del model, tokenizer
    gc.collect()
    if hasattr(mx, "clear_memory_cache"):
        mx.clear_memory_cache()

    # ------------------------------------------------------------------
    # Config 3: ZMLX patch with fused SwiGLU
    # ------------------------------------------------------------------
    if has_fused:
        print(f"\n{'=' * 60}")
        print("CONFIG 3: ZMLX patch + gather_qmm_swiglu (fused expert projections)")
        print(f"{'=' * 60}")
        model, tokenizer = load_model(args.model)
        apply_zmlx_patch(
            model,
            disable_fused_swiglu=False,
            fused_max_tokens=args.fused_max_tokens,
        )
        r3 = bench_config(
            model, tokenizer, "ZMLX + fused SwiGLU", args.runs, args.max_tokens
        )
        results.append(r3)
        del model, tokenizer
        gc.collect()
        if hasattr(mx, "clear_memory_cache"):
            mx.clear_memory_cache()
    else:
        print("\nSkipping Config 3 — mx.gather_qmm_swiglu not available")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    baseline = results[0]
    print(
        f"\n{'Config':<35} {'Prompt tok/s':>14} {'Decode tok/s':>14} {'Memory GB':>12}"
    )
    print("-" * 75)
    for r in results:
        label = r.get("label", "?")
        pt = r.get("median_prompt_tps", 0)
        gt = r.get("median_gen_tps", 0)
        mem = r.get("peak_memory_gb", 0)
        print(f"{label:<35} {pt:>14.1f} {gt:>14.1f} {mem:>12.2f}")

    # Speedup table
    base_prompt = baseline.get("median_prompt_tps", 0)
    base_gen = baseline.get("median_gen_tps", 0)
    if base_prompt > 0 and base_gen > 0:
        print(f"\n{'Config':<35} {'Prompt Speedup':>14} {'Decode Speedup':>14}")
        print("-" * 63)
        for r in results[1:]:
            label = r.get("label", "?")
            ps = r.get("median_prompt_tps", 0) / base_prompt
            gs = r.get("median_gen_tps", 0) / base_gen
            print(f"{label:<35} {ps:>13.2f}x {gs:>13.2f}x")

    if args.json_out:
        quant_meta = _infer_quant_meta(args.model, model_config)
        payload = {
            "model": args.model,
            "mlx_version": mx.__version__,
            "device": str(mx.default_device()),
            "device_meta": _device_meta(),
            "git_sha": _git_sha(),
            "quantization": quant_meta,
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "fused_max_tokens": args.fused_max_tokens,
            "results": results,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON summary: {args.json_out}")


if __name__ == "__main__":
    main()
