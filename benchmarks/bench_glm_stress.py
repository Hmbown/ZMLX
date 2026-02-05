#!/usr/bin/env python3
"""GLM-4.7-Flash stress benchmark (baseline vs patched).

This is the "full benchmark" protocol used for GLM-4.7-Flash:
  - Prompt diversity sweep (English technical, Chinese, code, math, creative)
  - Generation length sweep (default: 256, 1024, 2048)
  - Multiple runs per configuration (default: 5)
  - Token-by-token fidelity check (greedy decode)
  - Decode throughput + TTFT + per-token latency percentiles

Outputs:
  - A repro capsule JSON under ``benchmarks/repro_capsules/``
  - A full log under ``benchmarks/results/glm_stress/``

Example:
  # Quick sanity
  python benchmarks/bench_glm_stress.py --prompts english_technical,chinese --lengths 256,512 --runs 3

  # Full run (writes capsule + log)
  python benchmarks/bench_glm_stress.py --runs 5 --json-out benchmarks/repro_capsules/glm_stress_m4_<DATE>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import mlx.core as mx

DEFAULT_MODEL = "mlx-community/GLM-4.7-Flash-4bit"
DEFAULT_PROMPTS = "english_technical,chinese,code,math_reasoning,creative"
DEFAULT_LENGTHS = "256,1024,2048"


PROMPT_TEXT: dict[str, str] = {
    "english_technical": (
        "Explain how GLM-style MoE transformers route tokens to experts during decode. "
        "Cover gating math, load balancing, and why kernel launch overhead matters on Apple Silicon. "
        "Be technical and precise."
    ),
    "chinese": (
        "请用中文解释混合专家（MoE）模型在推理阶段如何进行路由与专家选择，并说明为什么在 Apple Silicon 上"
        "内存带宽和 kernel 启动开销会影响解码速度。要求严谨、清晰。"
    ),
    "code": (
        "Write a Python function `topk_router(logits, k)` that returns (indices, weights) where weights are "
        "softmax probabilities over the selected experts. Include a brief unit test that checks stability "
        "under ties (lower index wins)."
    ),
    "math_reasoning": (
        "Solve the following: Find all integer pairs (x, y) such that x^2 - 3y^2 = 1. "
        "Explain your reasoning and describe the general solution family."
    ),
    "creative": (
        "Write a short, vivid scene (300-500 words) about a late-night debugging session on a new GPU kernel. "
        "Use sensory details and keep the tone grounded, not comedic."
    ),
}


_T_CRIT_95: dict[int, float] = {
    # Two-sided 95% t critical values for df=1..30 (n = df + 1)
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass
class RunMetrics:
    prompt_tps: float
    gen_tps: float
    gen_tokens: int
    ttft_ms: float
    peak_mem_gb: float
    token_ms: list[float]
    token_ids: list[int]


class _Tee:
    def __init__(self, file: TextIO):
        self._file = file

    def print(self, *args: object, **kwargs: object) -> None:
        text = " ".join(str(a) for a in args)
        end = str(kwargs.get("end", "\n"))
        sys.stdout.write(text + end)
        sys.stdout.flush()
        self._file.write(text + end)
        self._file.flush()


def _sysctl_str(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return ""


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _git_short() -> str:
    sha = _git_sha()
    return sha[:7] if sha != "unknown" else sha


def _hardware_meta() -> tuple[str, int]:
    chip = _sysctl_str("machdep.cpu.brand_string") or platform.processor() or "unknown"
    mem_bytes = 0
    try:
        mem_bytes = int(_sysctl_str("hw.memsize") or "0")
    except Exception:
        mem_bytes = 0
    mem_gb = int(round(mem_bytes / (1024**3))) if mem_bytes else 0
    return chip, mem_gb


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    ys = sorted(float(x) for x in xs)
    rank = (p / 100.0) * (len(ys) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ys[lo]
    frac = rank - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def _ci95(xs: list[float]) -> list[float]:
    if not xs:
        return [0.0, 0.0]
    if len(xs) == 1:
        m = float(xs[0])
        return [m, m]
    n = len(xs)
    mean = float(statistics.mean(xs))
    stdev = float(statistics.stdev(xs))
    df = n - 1
    t = _T_CRIT_95.get(df, 1.96)
    se = stdev / math.sqrt(n)
    return [mean - t * se, mean + t * se]


def _summarize(prompt_name: str, max_tokens: int, config: str, runs: list[RunMetrics]) -> dict[str, Any]:
    gen_tps = [float(r.gen_tps) for r in runs]
    prompt_tps = [float(r.prompt_tps) for r in runs]
    ttft_ms = [float(r.ttft_ms) for r in runs]
    token_ms = [ms for r in runs for ms in r.token_ms]
    peak_mem = max((float(r.peak_mem_gb) for r in runs), default=0.0)

    return {
        "prompt_name": prompt_name,
        "max_tokens": int(max_tokens),
        "config": config,
        "n_runs": int(len(runs)),
        "median_gen_tps": float(statistics.median(gen_tps)) if gen_tps else 0.0,
        "mean_gen_tps": float(statistics.mean(gen_tps)) if gen_tps else 0.0,
        "stdev_gen_tps": float(statistics.stdev(gen_tps)) if len(gen_tps) > 1 else 0.0,
        "ci95_gen_tps": _ci95(gen_tps),
        "median_ttft_ms": float(statistics.median(ttft_ms)) if ttft_ms else 0.0,
        "median_prompt_tps": float(statistics.median(prompt_tps)) if prompt_tps else 0.0,
        "p50_token_ms": float(_percentile(token_ms, 50)),
        "p95_token_ms": float(_percentile(token_ms, 95)),
        "p99_token_ms": float(_percentile(token_ms, 99)),
        "peak_mem_gb": float(peak_mem),
    }


def _clear_gpu() -> None:
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _warmup(model: Any, tokenizer: Any, *, n: int) -> None:
    import mlx_lm

    for _ in range(n):
        _ = mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=5)
    mx.eval(mx.zeros(1))
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()


def _run_generate(model: Any, tokenizer: Any, prompt: str, *, max_tokens: int) -> RunMetrics:
    import mlx_lm

    start = time.perf_counter()
    token_ids: list[int] = []
    token_ms: list[float] = []
    ttft_ms = 0.0
    prev_token_t: float | None = None
    last = None

    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=int(max_tokens),
    ):
        last = resp
        if not hasattr(resp, "token"):
            continue

        now = time.perf_counter()
        if prev_token_t is None:
            ttft_ms = (now - start) * 1000.0
        else:
            token_ms.append((now - prev_token_t) * 1000.0)
        prev_token_t = now

        try:
            token_ids.append(int(resp.token))
        except (TypeError, ValueError):
            # Some mlx-lm builds expose token as already-int-like.
            token_ids.append(resp.token)  # type: ignore[arg-type]

    if last is None:
        return RunMetrics(
            prompt_tps=0.0,
            gen_tps=0.0,
            gen_tokens=0,
            ttft_ms=0.0,
            peak_mem_gb=0.0,
            token_ms=[],
            token_ids=[],
        )

    gen_tokens = int(getattr(last, "generation_tokens", 0) or 0)
    if gen_tokens and len(token_ids) >= gen_tokens:
        token_ids = token_ids[:gen_tokens]
    elif token_ids:
        gen_tokens = len(token_ids)
    else:
        text = getattr(last, "text", "") or ""
        if text:
            token_ids = tokenizer.encode(text)
            gen_tokens = len(token_ids)

    return RunMetrics(
        prompt_tps=float(getattr(last, "prompt_tps", 0.0) or 0.0),
        gen_tps=float(getattr(last, "generation_tps", 0.0) or 0.0),
        gen_tokens=gen_tokens,
        ttft_ms=ttft_ms,
        peak_mem_gb=float(getattr(last, "peak_memory", 0.0) or 0.0),
        token_ms=token_ms,
        token_ids=token_ids,
    )


def _compare_tokens(a: list[int], b: list[int]) -> tuple[int, int, int]:
    total = max(len(a), len(b))
    if total == 0:
        return 0, 0, -1
    matches = 0
    first_div = -1
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            matches += 1
        elif first_div == -1:
            first_div = i
    return matches, total, first_div


def _fmt_ci(ci: list[float]) -> str:
    if not ci or len(ci) != 2:
        return "—"
    return f"[{ci[0]:.1f}-{ci[1]:.1f}]"


def _fmt_speedup(x: float) -> str:
    if x <= 0:
        return "—"
    if x >= 1.07:
        return f"**{x:7.3f}x**"
    return f"{x:7.3f}x"


def _safe_relpath(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return p.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompts", type=str, default=DEFAULT_PROMPTS)
    parser.add_argument("--lengths", type=str, default=DEFAULT_LENGTHS)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup generations per model load (default: 2)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Write repro capsule JSON to this path (default: auto under benchmarks/repro_capsules/)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="benchmarks/results/glm_stress",
        help="Directory to write log files under (default: benchmarks/results/glm_stress)",
    )
    args = parser.parse_args()

    prompt_names = [p.strip() for p in args.prompts.split(",") if p.strip()]
    lengths: list[int] = []
    for part in args.lengths.split(","):
        part = part.strip()
        if not part:
            continue
        lengths.append(int(part))
    if not prompt_names:
        raise SystemExit("No prompts specified.")
    if not lengths:
        raise SystemExit("No lengths specified.")
    runs_per = int(args.runs)
    if runs_per <= 0:
        raise SystemExit("--runs must be > 0")

    missing = sorted(set(prompt_names) - set(PROMPT_TEXT))
    if missing:
        raise SystemExit(f"Unknown prompts: {missing}. Known: {sorted(PROMPT_TEXT)}")

    # Output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"glm_stress_full_{timestamp}.log"
    tee = _Tee(log_path.open("w", encoding="utf-8"))

    json_out = (args.json_out or "").strip()
    if not json_out:
        json_out = f"benchmarks/repro_capsules/glm_stress_m4_{datetime.now().strftime('%Y%m%d')}.json"
    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    chip, mem_gb = _hardware_meta()
    mlx_ver = getattr(mx, "__version__", "unknown")
    has_fused = hasattr(mx, "gather_qmm_swiglu")
    git_commit = _git_short()

    total_generations = len(prompt_names) * len(lengths) * runs_per * 2
    tee.print("GLM-4.7-Flash Stress Benchmark")
    tee.print(f"  Chip: {chip}")
    tee.print(f"  MLX: {mlx_ver}")
    tee.print(f"  Prompts: {prompt_names}")
    tee.print(f"  Lengths: {lengths}")
    tee.print(f"  Runs: {runs_per}")
    tee.print(f"  Total generations: {total_generations}")
    tee.print()

    # ------------------------------------------------------------------
    # Config 1: baseline
    # ------------------------------------------------------------------
    import mlx_lm

    tee.print("=" * 70)
    tee.print("  Loading: baseline")
    tee.print("=" * 70)
    tee.print("  Baseline (unpatched)")
    model, tokenizer = mlx_lm.load(args.model)[0:2]
    tee.print("  Warming up...")
    _warmup(model, tokenizer, n=int(args.warmup))

    baseline_runs: dict[str, list[RunMetrics]] = {}
    counter = 0
    total = len(prompt_names) * len(lengths) * runs_per
    for prompt_name in prompt_names:
        prompt = PROMPT_TEXT[prompt_name]
        for length in lengths:
            key = f"{prompt_name}@{length}"
            baseline_runs[key] = []
            for i in range(runs_per):
                counter += 1
                m = _run_generate(model, tokenizer, prompt, max_tokens=length)
                baseline_runs[key].append(m)
                tee.print(
                    f"  [{counter}/{total}] {prompt_name} @ {length} tok, run {i + 1}/{runs_per}...  "
                    f"gen={m.gen_tps:.1f} tok/s  ttft={m.ttft_ms:.1f}ms  mem={m.peak_mem_gb:.2f}GB"
                )

    del model, tokenizer
    _clear_gpu()

    # ------------------------------------------------------------------
    # Config 2: patched
    # ------------------------------------------------------------------
    tee.print()
    tee.print("=" * 70)
    tee.print("  Loading: patched")
    tee.print("=" * 70)
    tee.print("  Patching with ZMLX defaults...")
    model, tokenizer = mlx_lm.load(args.model)[0:2]
    from zmlx.patch import patch as zmlx_patch

    zmlx_patch(model, verbose=True)
    tee.print("  Warming up...")
    _warmup(model, tokenizer, n=int(args.warmup))

    patched_runs: dict[str, list[RunMetrics]] = {}
    counter = 0
    for prompt_name in prompt_names:
        prompt = PROMPT_TEXT[prompt_name]
        for length in lengths:
            key = f"{prompt_name}@{length}"
            patched_runs[key] = []
            for i in range(runs_per):
                counter += 1
                m = _run_generate(model, tokenizer, prompt, max_tokens=length)
                patched_runs[key].append(m)
                tee.print(
                    f"  [{counter}/{total}] {prompt_name} @ {length} tok, run {i + 1}/{runs_per}...  "
                    f"gen={m.gen_tps:.1f} tok/s  ttft={m.ttft_ms:.1f}ms  mem={m.peak_mem_gb:.2f}GB"
                )

    del model, tokenizer
    _clear_gpu()

    # ------------------------------------------------------------------
    # Summaries + fidelity
    # ------------------------------------------------------------------
    baseline_stats: dict[str, Any] = {}
    patched_stats: dict[str, Any] = {}
    fidelity: dict[str, Any] = {}

    for prompt_name in prompt_names:
        for length in lengths:
            key = f"{prompt_name}@{length}"
            b = baseline_runs.get(key, [])
            p = patched_runs.get(key, [])
            baseline_stats[key] = _summarize(prompt_name, length, "baseline", b)
            patched_stats[key] = _summarize(prompt_name, length, "patched", p)

            verdict = "ERROR"
            detail = "0/0"
            if b and p and len(b) == len(p):
                verdict = "PASS"
                total_tokens = 0
                for br, pr in zip(b, p, strict=True):
                    matches, total, first_div = _compare_tokens(br.token_ids, pr.token_ids)
                    if matches != total or len(br.token_ids) != len(pr.token_ids):
                        verdict = "FAIL"
                        extra = f" first_diverge={first_div}" if first_div >= 0 else ""
                        detail = f"{matches}/{total}{extra}"
                        break
                    total_tokens = total
                if verdict == "PASS":
                    detail = f"{total_tokens}/{total_tokens}"
            fidelity[key] = {"verdict": verdict, "detail": detail}

    # Print headline report (mirrors the legacy log format)
    tee.print()
    tee.print("=" * 80)
    tee.print("  GLM-4.7-Flash STRESS TEST RESULTS")
    tee.print("=" * 80)
    tee.print(f"  Chip: {chip}  |  MLX: {mlx_ver}")
    tee.print(f"  gather_qmm_swiglu: {bool(has_fused)}")
    tee.print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tee.print()
    tee.print("  DECODE THROUGHPUT (median tok/s)")
    tee.print("  Prompt               Tokens   Baseline    Patched  Speedup             CI95   Fidelity")
    tee.print("  " + "-" * 82)

    for length in lengths:
        for prompt_name in prompt_names:
            key = f"{prompt_name}@{length}"
            b_med = float(baseline_stats[key]["median_gen_tps"])
            p_med = float(patched_stats[key]["median_gen_tps"])
            speed = (p_med / b_med) if b_med > 0 else 0.0
            ci = _fmt_ci(list(patched_stats[key]["ci95_gen_tps"]))
            fid = fidelity[key]["verdict"]
            tee.print(
                f"  {prompt_name:<18} {length:>7}  {b_med:>9.1f}  {p_med:>9.1f}  "
                f"{_fmt_speedup(speed):>8}      {ci:>12}   {fid:>9}"
            )
        tee.print()

    tee.print("  PER-TOKEN LATENCY (ms, patched)")
    tee.print("  Prompt               Tokens      P50      P95      P99       TTFT")
    tee.print("  " + "-" * 64)
    for length in lengths:
        for prompt_name in prompt_names:
            key = f"{prompt_name}@{length}"
            ps = patched_stats[key]
            tee.print(
                f"  {prompt_name:<18} {length:>7}  "
                f"{ps['p50_token_ms']:>7.2f}  {ps['p95_token_ms']:>7.2f}  {ps['p99_token_ms']:>7.2f}  "
                f"{ps['median_ttft_ms']:>9.1f}ms"
            )
    tee.print()

    tee.print("  SPEEDUP BY GENERATION LENGTH (averaged across prompts)")
    tee.print("  Length   Avg Baseline    Avg Patched  Avg Speedup")
    tee.print("  " + "-" * 48)
    for length in lengths:
        b_vals = [float(baseline_stats[f"{p}@{length}"]["median_gen_tps"]) for p in prompt_names]
        p_vals = [float(patched_stats[f"{p}@{length}"]["median_gen_tps"]) for p in prompt_names]
        b_avg = float(statistics.mean(b_vals)) if b_vals else 0.0
        p_avg = float(statistics.mean(p_vals)) if p_vals else 0.0
        s = (p_avg / b_avg) if b_avg > 0 else 0.0
        tee.print(f"  {length:>6}  {b_avg:>13.1f}  {p_avg:>12.1f}    {s:>8.3f}x")
    tee.print()

    tee.print("  SPEEDUP BY PROMPT TYPE (averaged across lengths)")
    tee.print("  Prompt                 Avg Baseline    Avg Patched  Avg Speedup")
    tee.print("  " + "-" * 62)
    for prompt_name in prompt_names:
        b_vals = [
            float(baseline_stats[f"{prompt_name}@{gen_len}"]["median_gen_tps"]) for gen_len in lengths
        ]
        p_vals = [
            float(patched_stats[f"{prompt_name}@{gen_len}"]["median_gen_tps"]) for gen_len in lengths
        ]
        b_avg = float(statistics.mean(b_vals)) if b_vals else 0.0
        p_avg = float(statistics.mean(p_vals)) if p_vals else 0.0
        s = (p_avg / b_avg) if b_avg > 0 else 0.0
        tee.print(f"  {prompt_name:<22} {b_avg:>12.1f}  {p_avg:>12.1f}    {s:>8.3f}x")
    tee.print()

    all_pass = all(v.get("verdict") == "PASS" for v in fidelity.values())
    tee.print(f"  FIDELITY: {'ALL PASS' if all_pass else 'FAILURES'}")
    tee.print()

    capsule = {
        "metadata": {
            "model": args.model,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mlx_version": mlx_ver,
            "has_gather_qmm_swiglu": bool(has_fused),
            "chip": chip,
            "memory_gb": float(mem_gb) if mem_gb else 0.0,
            "zmlx_commit": git_commit,
            "git_sha": _git_sha(),
            "log": _safe_relpath(log_path),
        },
        "generation_lengths": lengths,
        "prompt_names": prompt_names,
        "prompt_texts": {k: PROMPT_TEXT[k] for k in prompt_names},
        "runs_per_config": runs_per,
        "baseline_stats": baseline_stats,
        "patched_stats": patched_stats,
        "fidelity": fidelity,
    }
    json_path.write_text(json.dumps(capsule, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    tee.print(f"  Saved capsule to {_safe_relpath(json_path)}")
    tee.print(f"  Saved log to {_safe_relpath(log_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
