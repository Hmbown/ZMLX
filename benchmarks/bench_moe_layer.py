#!/usr/bin/env python3
"""Kernel-level MoE layer benchmark: measures raw GPU time per forward pass.

Isolates a single MoE layer and times it with mx.synchronize() brackets,
removing all E2E noise (tokenizer, sampling, KV cache, attention).

Usage:
    python benchmarks/bench_moe_layer.py
    python benchmarks/bench_moe_layer.py --model mlx-community/Qwen3-30B-A3B-4bit
    python benchmarks/bench_moe_layer.py --seq-lens 1,4,16,64
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import mlx.core as mx


def load_model(model_path: str):
    import mlx_lm
    model, tokenizer = mlx_lm.load(model_path)
    return model, tokenizer


def find_moe_layer(model, layer_idx: int | None = None):
    """Find and return the first (or specified) MoE layer."""
    for i, layer in enumerate(model.layers):
        candidates = [
            getattr(layer, "feed_forward", None),
            getattr(layer, "mlp", None),
        ]
        for mod in candidates:
            if mod is None:
                continue
            if hasattr(mod, "gate") or hasattr(mod, "router"):
                if hasattr(mod, "switch_mlp") or hasattr(mod, "experts"):
                    if layer_idx is not None and i != layer_idx:
                        continue
                    return i, mod
    return None, None


def bench_layer(
    moe_layer,
    hidden_dim: int,
    seq_len: int = 1,
    warmup: int = 50,
    iters: int = 500,
    dtype=mx.float16,
) -> dict:
    """Benchmark a single MoE layer's forward pass.

    Returns dict with timing stats in microseconds.
    """
    x = mx.random.normal((1, seq_len, hidden_dim)).astype(dtype)
    mx.eval(x)

    sync = getattr(mx, "synchronize", None)
    if not callable(sync):
        def sync() -> None:
            mx.eval(mx.zeros(1))

    # Warmup
    for _ in range(warmup):
        out = moe_layer(x)
        mx.eval(out)
    sync()

    # Timed runs
    times_us = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        out = moe_layer(x)
        mx.eval(out)
        sync()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    times_us.sort()
    n = len(times_us)
    return {
        "p10": times_us[int(n * 0.10)],
        "p25": times_us[int(n * 0.25)],
        "p50": times_us[n // 2],
        "p75": times_us[int(n * 0.75)],
        "p90": times_us[int(n * 0.90)],
        "mean": statistics.mean(times_us),
        "stdev": statistics.stdev(times_us),
    }


def print_stats(label: str, stats: dict):
    print(
        f"  {label:.<45} "
        f"p50={stats['p50']:7.1f}us  "
        f"mean={stats['mean']:7.1f}us  "
        f"p10={stats['p10']:7.1f}  "
        f"p90={stats['p90']:7.1f}  "
        f"stdev={stats['stdev']:5.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Kernel-level MoE layer benchmark")
    parser.add_argument(
        "--model", default="mlx-community/LFM2-8B-A1B-4bit",
        help="Model path",
    )
    parser.add_argument(
        "--seq-lens", default="1,4,16",
        help="Comma-separated sequence lengths to test (default: 1,4,16)",
    )
    parser.add_argument("--iters", type=int, default=500, help="Iterations per config")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument(
        "--layer-idx", type=int, default=None,
        help="Specific layer index to benchmark (default: first MoE layer)",
    )
    args = parser.parse_args()

    seq_lens = [int(s.strip()) for s in args.seq_lens.split(",")]

    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print(f"gather_qmm_swiglu available: {hasattr(mx, 'gather_qmm_swiglu')}")
    print(f"Model: {args.model}")
    print(f"Iterations: {args.iters}, Warmup: {args.warmup}")
    print()

    results = []

    for seq_len in seq_lens:
        print(f"{'=' * 70}")
        print(f"SEQUENCE LENGTH = {seq_len} (simulating {'decode' if seq_len == 1 else 'prefill'})")
        print(f"{'=' * 70}")

        configs = []

        # --- Config 1: Baseline ---
        print("\n  Loading baseline model...")
        model, _ = load_model(args.model)
        idx, moe_layer = find_moe_layer(model, args.layer_idx)
        if moe_layer is None:
            print("  ERROR: No MoE layer found")
            return

        # Get hidden_dim from model config or infer from gate weight
        hidden_dim = getattr(model, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(model.args, "hidden_size", None) if hasattr(model, "args") else None
        if hidden_dim is None:
            # Infer from gate linear layer
            gate_attr = getattr(moe_layer, "gate", None) or getattr(moe_layer, "router", None)
            if gate_attr is not None and hasattr(gate_attr, "weight"):
                hidden_dim = gate_attr.weight.shape[-1]
            else:
                hidden_dim = 2048  # fallback
        print(f"  MoE layer index: {idx}, hidden_dim: {hidden_dim}")

        baseline = bench_layer(
            moe_layer, hidden_dim, seq_len=seq_len,
            warmup=args.warmup, iters=args.iters,
        )
        print_stats("Baseline (unpatched)", baseline)
        configs.append(("Baseline", baseline))
        del model
        gc.collect()
        if hasattr(mx, "clear_memory_cache"):
            mx.clear_memory_cache()

        # --- Config 2: ZMLX gating+combine only ---
        print("\n  Loading model for gating+combine patch...")
        model, _ = load_model(args.model)
        from zmlx.kernels import fused_moe
        original = fused_moe._HAS_GATHER_QMM_SWIGLU
        fused_moe._HAS_GATHER_QMM_SWIGLU = False
        try:
            from zmlx.patch import patch as zmlx_patch
            zmlx_patch(model, verbose=False)
        finally:
            fused_moe._HAS_GATHER_QMM_SWIGLU = original

        _, moe_layer = find_moe_layer(model, idx)
        gating = bench_layer(
            moe_layer, hidden_dim, seq_len=seq_len,
            warmup=args.warmup, iters=args.iters,
        )
        print_stats("ZMLX gating+combine", gating)
        configs.append(("Gating+combine", gating))
        del model
        gc.collect()
        if hasattr(mx, "clear_memory_cache"):
            mx.clear_memory_cache()

        # --- Config 3: ZMLX + fused SwiGLU ---
        if hasattr(mx, "gather_qmm_swiglu"):
            print("\n  Loading model for fused SwiGLU patch...")
            model, _ = load_model(args.model)
            from zmlx.patch import patch as zmlx_patch
            zmlx_patch(model, verbose=False)

            _, moe_layer = find_moe_layer(model, idx)
            fused = bench_layer(
                moe_layer, hidden_dim, seq_len=seq_len,
                warmup=args.warmup, iters=args.iters,
            )
            print_stats("ZMLX + fused SwiGLU", fused)
            configs.append(("Fused SwiGLU", fused))
            del model
            gc.collect()
            if hasattr(mx, "clear_memory_cache"):
                mx.clear_memory_cache()

        # Summary for this seq_len
        print(f"\n  Summary (seq_len={seq_len}):")
        base_p50 = configs[0][1]["p50"]
        print(f"  {'Config':<30} {'p50 (us)':>10} {'vs Base':>10} {'stdev':>8}")
        print(f"  {'-' * 58}")
        for name, stats in configs:
            speedup = base_p50 / stats["p50"] if stats["p50"] > 0 else 0
            print(
                f"  {name:<30} {stats['p50']:>10.1f} {speedup:>9.2f}x "
                f"{stats['stdev']:>7.1f}"
            )
        results.append((seq_len, configs))
        print()

    # Grand summary
    print(f"{'=' * 70}")
    print("GRAND SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'seq_len':<10} ", end="")
    config_names = [c[0] for c in results[0][1]]
    for name in config_names:
        print(f"{name:>20}", end="")
    print()
    print("-" * (10 + 20 * len(config_names)))

    for seq_len, configs in results:
        base_p50 = configs[0][1]["p50"]
        print(f"{seq_len:<10} ", end="")
        for _name, stats in configs:
            speedup = base_p50 / stats["p50"] if stats["p50"] > 0 else 0
            print(f"{stats['p50']:>10.1f}us {speedup:>6.2f}x", end="  ")
        print()


if __name__ == "__main__":
    main()
