#!/usr/bin/env python3
"""Benchmark MoE stream pool variants for list-of-experts models.

Runs baseline (unpatched), patched without streams, and patched with multiple
stream counts. Reports median prompt/decode tok/s plus token-fidelity checks.

Example:
  python benchmarks/bench_moe_streams.py --model mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit \
    --streams 1,2,4,8 --runs 5 --max-tokens 500 --json-out benchmarks/repro_capsules/mixtral_streams.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlx.core as mx

DEFAULT_MODEL = "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit"
DEFAULT_STREAMS = "1,2,4,8"
DEFAULT_PROMPT = (
    "Explain the key differences between mixture-of-experts (MoE) and dense "
    "transformer architectures. Cover parameter efficiency, routing strategies, "
    "load balancing, and inference characteristics. Be detailed and thorough."
)

_STREAMS_ENV = "ZMLX_MOE_STREAMS"
_STREAMS_REDUCE_ENV = "ZMLX_MOE_STREAMS_REDUCE"


@dataclass
class _RunMetrics:
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    peak_mem_gb: float = 0.0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    token_ids: list[int] | None = None


def _clear_gpu() -> None:
    gc.collect()
    if hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _set_stream_env(count: int, reduce_mode: str | None) -> None:
    if count <= 1:
        os.environ.pop(_STREAMS_ENV, None)
        os.environ.pop(_STREAMS_REDUCE_ENV, None)
        return
    else:
        os.environ[_STREAMS_ENV] = str(count)

    if reduce_mode and reduce_mode != "serial":
        os.environ[_STREAMS_REDUCE_ENV] = reduce_mode
    else:
        os.environ.pop(_STREAMS_REDUCE_ENV, None)


def _load_model(model_path: str):
    import mlx_lm

    print(f"  Loading {model_path} ...")
    model, tokenizer = mlx_lm.load(model_path)
    return model, tokenizer


def _warmup(model, tokenizer, n: int = 2) -> None:
    import mlx_lm

    for _ in range(n):
        _ = mlx_lm.generate(model, tokenizer, prompt="Hi", max_tokens=5)
    mx.eval(mx.zeros(1))
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()


def _generate_greedy(model, tokenizer, prompt: str, max_tokens: int) -> _RunMetrics:
    import mlx_lm

    metrics = _RunMetrics(token_ids=[])
    last = None

    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        last = resp
        if hasattr(resp, "token") and metrics.token_ids is not None:
            try:
                metrics.token_ids.append(int(resp.token))
            except (TypeError, ValueError):
                metrics.token_ids.append(resp.token)

    if last is None:
        return metrics

    metrics.prompt_tokens = getattr(last, "prompt_tokens", 0)
    metrics.prompt_tps = getattr(last, "prompt_tps", 0.0)
    metrics.gen_tokens = getattr(last, "generation_tokens", 0)
    metrics.gen_tps = getattr(last, "generation_tps", 0.0)
    metrics.peak_mem_gb = getattr(last, "peak_memory", 0.0)

    if metrics.token_ids is not None and metrics.token_ids and metrics.gen_tokens:
        metrics.token_ids = metrics.token_ids[: metrics.gen_tokens]
    elif metrics.token_ids is not None and hasattr(last, "text"):
        text = getattr(last, "text", "") or ""
        if text:
            try:
                metrics.token_ids = tokenizer.encode(text)
            except Exception:
                metrics.token_ids = []
    return metrics


def _compare_tokens(a: list[int] | None, b: list[int] | None) -> dict:
    if not a or not b:
        return {"matched": 0, "total": 0, "first_diverge": -1, "verdict": "UNKNOWN"}

    total = max(len(a), len(b))
    matches = 0
    first_div = -1
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            matches += 1
        elif first_div == -1:
            first_div = i
    verdict = "PASS" if total > 0 and matches == total else "FAIL"
    return {"matched": matches, "total": total, "first_diverge": first_div, "verdict": verdict}


def _summarize_runs(runs: list[_RunMetrics]) -> dict:
    if not runs:
        return {}

    prefill = [r.prompt_tps for r in runs]
    decode = [r.gen_tps for r in runs]
    return {
        "prefill_tok_s": prefill,
        "decode_tok_s": decode,
        "median_prefill": statistics.median(prefill),
        "median_decode": statistics.median(decode),
        "peak_mem_gb": max(r.peak_mem_gb for r in runs),
        "prompt_tokens": runs[0].prompt_tokens,
        "gen_tokens": runs[0].gen_tokens,
    }


def _detect_expert_style(model) -> tuple[str, int | None]:
    layers = getattr(model, "layers", []) or []
    for layer in layers:
        candidates = [
            getattr(layer, "feed_forward", None),
            getattr(layer, "mlp", None),
        ]
        for mod in candidates:
            if mod is None:
                continue
            if not (hasattr(mod, "gate") or hasattr(mod, "router")):
                continue
            experts = getattr(mod, "experts", None)
            if isinstance(experts, list):
                return "list", len(experts)
            if experts is not None and hasattr(experts, "gate_proj") and hasattr(experts, "up_proj"):
                return "switch", None
            if hasattr(mod, "switch_mlp"):
                return "switch", None
    return "unknown", None


def _apply_patch(model, *, verbose: bool) -> None:
    from zmlx.patch import patch as zmlx_patch

    zmlx_patch(model, verbose=verbose)


def _bench_config(
    label: str,
    model_path: str,
    *,
    patched: bool,
    streams: int,
    reduce_mode: str | None,
    prompt: str,
    max_tokens: int,
    runs: int,
    verbose: bool,
    capture: bool,
) -> tuple[dict, list[int] | None, dict]:
    _set_stream_env(streams if patched else 0, reduce_mode if patched else None)

    print(f"\n  --- {label} ---")
    model, tokenizer = _load_model(model_path)

    if patched:
        _apply_patch(model, verbose=verbose)

    style, expert_count = _detect_expert_style(model)
    if patched and streams > 1 and style != "list":
        print("  NOTE: expert style is not list-of-experts; stream pool is inactive.")

    print("  Warming up ...")
    _warmup(model, tokenizer)

    runs_out: list[_RunMetrics] = []
    for i in range(runs):
        if capture and i == 0 and hasattr(mx, "metal") and hasattr(mx.metal, "start_capture"):
            print("  Starting Metal capture ...")
            mx.metal.start_capture()
        metrics = _generate_greedy(model, tokenizer, prompt, max_tokens)
        if capture and i == 0 and hasattr(mx, "metal") and hasattr(mx.metal, "stop_capture"):
            mx.metal.stop_capture()
            print("  Metal capture stopped")
        runs_out.append(metrics)
        print(
            f"    Run {i + 1}/{runs}: "
            f"prompt={metrics.prompt_tps:.1f} tok/s, "
            f"gen={metrics.gen_tps:.1f} tok/s, "
            f"mem={metrics.peak_mem_gb:.2f} GB"
        )

    summary = _summarize_runs(runs_out)
    summary.update(
        {
            "label": label,
            "patched": patched,
            "streams": streams,
            "expert_style": style,
            "experts": expert_count,
        }
    )
    tokens = runs_out[0].token_ids if runs_out else None

    del model, tokenizer
    _clear_gpu()
    return summary, tokens, {"style": style, "experts": expert_count}


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
        ).strip()
    except Exception:
        return ""


def _format_command(args: argparse.Namespace) -> str:
    parts = ["python", "benchmarks/bench_moe_streams.py"]
    if args.model != DEFAULT_MODEL:
        parts += ["--model", args.model]
    if args.streams != DEFAULT_STREAMS:
        parts += ["--streams", args.streams]
    if args.runs != 3:
        parts += ["--runs", str(args.runs)]
    if args.max_tokens != 500:
        parts += ["--max-tokens", str(args.max_tokens)]
    if args.reduce != "serial":
        parts += ["--reduce", args.reduce]
    if args.capture_streams is not None:
        parts += ["--capture-streams", str(args.capture_streams)]
    if args.note:
        parts += ["--note", "<note>"]
    if args.json_out:
        json_out = Path(args.json_out)
        if json_out.is_absolute():
            try:
                repo_root = Path(__file__).resolve().parents[1]
                rel = json_out.relative_to(repo_root)
                parts += ["--json-out", f"<REPO_ROOT>/{rel.as_posix()}"]
            except ValueError:
                parts += ["--json-out", "<json-out>"]
        else:
            parts += ["--json-out", args.json_out]
    return " ".join(parts)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_streams(raw: str) -> list[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as err:
            raise SystemExit(f"Invalid stream count: {part}") from err
    if not values:
        return [1]
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE stream pool benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path")
    parser.add_argument(
        "--streams",
        default=DEFAULT_STREAMS,
        help="Comma-separated stream counts to test (default: 1,2,4,8)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per config")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt")
    parser.add_argument(
        "--reduce",
        default="serial",
        choices=("serial", "tree", "stack"),
        help="Stream reduction mode (only affects streams>1)",
    )
    parser.add_argument(
        "--capture-streams",
        type=int,
        default=None,
        help="Start/stop Metal capture on the first run of this stream count (use 0 for baseline)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose patching")
    parser.add_argument("--note", default="", help="Optional note for JSON capsule")
    parser.add_argument("--json-out", default=None, help="Write JSON summary here")
    args = parser.parse_args()

    streams = _parse_streams(args.streams)
    if 1 not in streams:
        streams = [1, *streams]

    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print(f"Model: {args.model}")
    print(f"Streams: {streams}")
    print(f"Runs per config: {args.runs}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Reduction: {args.reduce}")

    results: list[dict] = []
    stream_tokens: dict[int, list[int] | None] = {}

    # Baseline (unpatched)
    baseline_summary, baseline_tokens, _ = _bench_config(
        "Baseline (unpatched)",
        args.model,
        patched=False,
        streams=1,
        reduce_mode=None,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        runs=args.runs,
        verbose=args.verbose,
        capture=args.capture_streams == 0,
    )
    results.append(baseline_summary)

    # Patched, no streams (streams=1)
    patched_summary, patched_tokens, _ = _bench_config(
        "Patched (streams=1)",
        args.model,
        patched=True,
        streams=1,
        reduce_mode=args.reduce,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        runs=args.runs,
        verbose=args.verbose,
        capture=args.capture_streams == 1,
    )
    results.append(patched_summary)
    stream_tokens[1] = patched_tokens

    # Patched with streams > 1
    for count in streams:
        if count <= 1:
            continue
        summary, tokens, _ = _bench_config(
            f"Patched (streams={count})",
            args.model,
            patched=True,
            streams=count,
            reduce_mode=args.reduce,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            runs=args.runs,
            verbose=args.verbose,
            capture=args.capture_streams == count,
        )
        results.append(summary)
        stream_tokens[count] = tokens

    # Fidelity (baseline vs each config)
    fidelity = {
        "baseline_vs_patched": _compare_tokens(baseline_tokens, patched_tokens),
        "baseline_vs_streams": {},
    }
    for count, tokens in stream_tokens.items():
        if count <= 1:
            continue
        key = f"streams_{count}"
        fidelity["baseline_vs_streams"][key] = _compare_tokens(
            baseline_tokens,
            tokens,
        )

    # Summary table
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    base_decode = baseline_summary.get("median_decode", 0.0) or 0.0
    print(f"{'Config':<30} {'Prompt tok/s':>14} {'Decode tok/s':>14} {'Speedup':>10}")
    print("-" * 72)
    for summary in results:
        label = summary.get("label", "?")
        pt = summary.get("median_prefill", 0.0)
        gt = summary.get("median_decode", 0.0)
        speedup = (gt / base_decode) if base_decode > 0 else 0.0
        print(f"{label:<30} {pt:>14.1f} {gt:>14.1f} {speedup:>9.3f}x")

    # JSON output (repro capsule style)
    if args.json_out:
        try:
            import zmlx
            from zmlx.device import detect_device
        except Exception:
            zmlx = None
            detect_device = None

        meta = {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "device": "",
            "memory_gb": 0,
            "macos": platform.mac_ver()[0] or "",
            "mlx_version": mx.__version__,
            "zmlx_version": getattr(zmlx, "__version__", "") if zmlx else "",
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "command": _format_command(args),
            "note": args.note,
        }
        if detect_device:
            dev = detect_device()
            meta["device"] = dev.full_name
            meta["memory_gb"] = dev.memory_gb

        stream_results = {
            str(summary.get("streams")): summary
            for summary in results
            if summary.get("streams", 0) > 1
        }
        payload = {
            "meta": meta,
            "moe_streams": {
                "model": args.model,
                "runs": args.runs,
                "max_tokens": args.max_tokens,
                "reduce_mode": args.reduce,
                "streams": stream_results,
                "baseline": baseline_summary,
                "patched": patched_summary,
                "fidelity": fidelity,
                "results": results,
            },
        }
        _write_json(Path(args.json_out), payload)
        print(f"\nWrote JSON: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
