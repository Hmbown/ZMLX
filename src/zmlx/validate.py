"""Model-agnostic fidelity and performance validation for ZMLX patches.

Usage::

    python -m zmlx.validate mlx-community/LFM2-8B-A1B-4bit \\
        --patterns moe_mlp residual_norm \\
        --max-tokens 200 --runs 3

Loads the model twice (baseline then patched), generates tokens with greedy
decoding (temp=0), and compares token-for-token fidelity plus throughput.
"""

from __future__ import annotations

import argparse
import gc
import statistics
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class _RunMetrics:
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    gen_tokens: int = 0
    gen_tps: float = 0.0
    peak_mem_gb: float = 0.0
    token_ids: list[int] = field(default_factory=list)


@dataclass
class _ConfigResult:
    label: str = ""
    runs: list[_RunMetrics] = field(default_factory=list)

    @property
    def median_prompt_tps(self) -> float:
        return statistics.median(r.prompt_tps for r in self.runs) if self.runs else 0.0

    @property
    def median_gen_tps(self) -> float:
        return statistics.median(r.gen_tps for r in self.runs) if self.runs else 0.0

    @property
    def peak_mem_gb(self) -> float:
        return max((r.peak_mem_gb for r in self.runs), default=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clear_gpu() -> None:
    gc.collect()
    if hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _generate_greedy(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict[str, Any] | None = None,
) -> _RunMetrics:
    """Generate tokens with greedy decoding and collect metrics."""
    import mlx_lm

    metrics = _RunMetrics()
    last = None
    token_ids: list[int] = []

    kwargs = gen_kwargs or {}
    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        **kwargs,
    ):
        last = resp
        if hasattr(resp, "token"):
            try:
                token_ids.append(int(resp.token))
            except (TypeError, ValueError):
                token_ids.append(resp.token)

    if last is None:
        return metrics

    metrics.prompt_tokens = last.prompt_tokens
    metrics.prompt_tps = last.prompt_tps
    metrics.gen_tokens = last.generation_tokens
    metrics.gen_tps = last.generation_tps
    metrics.peak_mem_gb = last.peak_memory
    if token_ids:
        gen_count = last.generation_tokens
        metrics.token_ids = (
            token_ids[:gen_count]
            if gen_count and len(token_ids) >= gen_count
            else token_ids
        )
    else:
        # Fallback: decode generated text if token stream wasn't available.
        full_text = last.text if hasattr(last, "text") else ""
        if full_text:
            metrics.token_ids = tokenizer.encode(full_text)
    return metrics


def _warmup(model, tokenizer, n: int = 2) -> None:
    import mlx_lm

    for _ in range(n):
        mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=5)
    mx.eval(mx.zeros(1))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def _bench_config(
    model_path: str,
    label: str,
    patterns: list[str] | None,
    profile: str | None,
    prompt: str,
    max_tokens: int,
    runs: int,
    gen_kwargs: dict[str, Any] | None,
) -> _ConfigResult:
    import mlx_lm

    from zmlx.patch import patch as zmlx_patch

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Loading {model_path} ...")
    loaded = mlx_lm.load(model_path)
    model, tokenizer = loaded[0], loaded[1]

    patched_count = 0
    if patterns is None and profile is None:
        print("  Applying: patch(model) with model-aware defaults")
        zmlx_patch(model, verbose=True)
    elif profile is not None:
        print(f"  Applying: patch(model, profile={profile!r})")
        zmlx_patch(model, profile=profile, verbose=True)
    elif patterns:
        print(f"  Applying patterns: {patterns}")
        zmlx_patch(model, patterns=patterns, verbose=True)
    else:
        print("  Applying patterns: [] (no patches)")
        result_attr = getattr(model, "_zmlx_patch_result", None)
        if result_attr is not None:
            patched_count = result_attr.patched_count
        print(f"  Patched {patched_count} modules")

    print("  Warming up ...")
    _warmup(model, tokenizer)

    result = _ConfigResult(label=label)

    for i in range(runs):
        print(f"  Run {i + 1}/{runs} ...", end="", flush=True)
        m = _generate_greedy(model, tokenizer, prompt, max_tokens, gen_kwargs)
        result.runs.append(m)
        print(
            f"  prompt={m.prompt_tps:.1f} tok/s  "
            f"gen={m.gen_tps:.1f} tok/s  "
            f"mem={m.peak_mem_gb:.2f} GB"
        )

    del model, tokenizer
    _clear_gpu()
    return result


def _compare_tokens(baseline: _RunMetrics, patched: _RunMetrics) -> tuple[int, int, int]:
    """Compare token ID sequences. Returns (match_count, total, first_diverge)."""
    b_ids = baseline.token_ids
    p_ids = patched.token_ids
    total = max(len(b_ids), len(p_ids))
    if total == 0:
        return 0, 0, -1
    matches = 0
    first_div = -1
    for i in range(min(len(b_ids), len(p_ids))):
        if b_ids[i] == p_ids[i]:
            matches += 1
        elif first_div == -1:
            first_div = i
    return matches, total, first_div


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m zmlx.validate",
        description="Model-agnostic fidelity and performance validation for ZMLX patches.",
    )
    parser.add_argument("model", help="HuggingFace model path (e.g. mlx-community/...)")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="ZMLX patch patterns to apply (default: FUSED_ACTIVATIONS)",
    )
    parser.add_argument(
        "--patch-profile",
        type=str,
        default=None,
        help="Patch profile to apply (e.g., qwen3)",
    )
    parser.add_argument("--max-tokens", type=int, default=200, help="Tokens to generate")
    parser.add_argument("--runs", type=int, default=3, help="Timed runs per config")
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Quantize KV cache to N bits during generation (env: ZMLX_KV_BITS).",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=None,
        help="Group size for KV cache quantization (env: ZMLX_KV_GROUP_SIZE).",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=None,
        help="Step to begin using a quantized KV cache (env: ZMLX_QUANTIZED_KV_START).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Explain the key differences between TCP and UDP protocols, "
            "including their use cases, reliability guarantees, and "
            "performance characteristics. Be thorough and precise."
        ),
    )
    args = parser.parse_args()

    if args.patterns is not None and args.patch_profile is not None:
        raise SystemExit("Use either --patterns or --patch-profile, not both.")

    from zmlx.kv_cache import kv_cache_kwargs

    gen_kwargs = kv_cache_kwargs(
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )

    # --- Baseline ---
    baseline = _bench_config(
        model_path=args.model,
        label="Baseline (unpatched)",
        patterns=[],
        profile=None,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        runs=args.runs,
        gen_kwargs=gen_kwargs,
    )

    # --- Patched ---
    patched = _bench_config(
        model_path=args.model,
        label="ZMLX Patched",
        patterns=args.patterns,
        profile=args.patch_profile,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        runs=args.runs,
        gen_kwargs=gen_kwargs,
    )

    # --- Fidelity ---
    # Compare first run token IDs (both greedy, should be deterministic)
    b_run = baseline.runs[0] if baseline.runs else _RunMetrics()
    p_run = patched.runs[0] if patched.runs else _RunMetrics()
    match_count, total, first_div = _compare_tokens(b_run, p_run)
    fidelity_pass = total > 0 and match_count == total

    # --- Report ---
    print(f"\n{'='*60}")
    print("  VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"  Model:    {args.model}")
    if args.patch_profile:
        patterns_label = f"profile={args.patch_profile}"
    else:
        patterns_label = args.patterns or "(default FUSED_ACTIVATIONS)"
    print(f"  Patterns: {patterns_label}")
    print(f"  Tokens:   {args.max_tokens} greedy (temp=0)")
    print()

    verdict = "PASS" if fidelity_pass else "FAIL"
    print(f"  FIDELITY: {match_count}/{total} tokens identical  [{verdict}]")
    if first_div >= 0:
        print(f"            First divergence at token {first_div}")
    print()

    print(f"  PERFORMANCE (median of {args.runs} runs):")
    print(f"  {'Config':<20} {'Prompt tok/s':>14} {'Decode tok/s':>14} {'Peak Mem':>10}")
    print(f"  {'-'*58}")
    print(
        f"  {'Baseline':<20} "
        f"{baseline.median_prompt_tps:>14.1f} "
        f"{baseline.median_gen_tps:>14.1f} "
        f"{baseline.peak_mem_gb:>9.2f} GB"
    )
    print(
        f"  {'Patched':<20} "
        f"{patched.median_prompt_tps:>14.1f} "
        f"{patched.median_gen_tps:>14.1f} "
        f"{patched.peak_mem_gb:>9.2f} GB"
    )
    if baseline.median_gen_tps > 0:
        speedup = patched.median_gen_tps / baseline.median_gen_tps
        print(f"  {'Speedup':<20} {'':>14} {speedup:>13.3f}x")

    print()


if __name__ == "__main__":
    main()
