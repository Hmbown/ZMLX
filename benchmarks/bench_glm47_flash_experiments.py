#!/usr/bin/env python3
"""GLM-4.7-Flash experiments: RoPE + shared-expert overlap.

Produces:
  - a repro capsule JSON (validate-style) under benchmarks/repro_capsules/
  - optional matrix entries appended to a JSONL ledger (benchmarks/matrix.jsonl)

This script is intended for quick iteration on GLM-4.7-Flash decode throughput
while preserving greedy token fidelity.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx

DEFAULT_MODEL = "mlx-community/GLM-4.7-Flash-4bit"
DEFAULT_PROMPT = (
    "Explain the key differences between TCP and UDP protocols, "
    "including their use cases, reliability guarantees, and "
    "performance characteristics. Be thorough and precise."
)


def _sysctl_str(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return ""


def _git_short_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return ""


def _hardware_meta() -> tuple[str, int]:
    chip = _sysctl_str("machdep.cpu.brand_string") or platform.processor() or "unknown"
    mem_bytes = 0
    try:
        mem_bytes = int(_sysctl_str("hw.memsize") or "0")
    except Exception:
        mem_bytes = 0
    mem_gb = int(round(mem_bytes / (1024**3))) if mem_bytes else 0
    return chip, mem_gb


def _round1(xs: list[float]) -> list[float]:
    return [round(float(x), 1) for x in xs]


def _summarize_config(result) -> dict:
    prefill = [float(r.prompt_tps) for r in result.runs]
    decode = [float(r.gen_tps) for r in result.runs]
    return {
        "prefill_tok_s": _round1(prefill),
        "decode_tok_s": _round1(decode),
        "median_prefill": round(statistics.median(prefill), 1) if prefill else 0.0,
        "median_decode": round(statistics.median(decode), 1) if decode else 0.0,
        "peak_mem_gb": round(float(result.peak_mem_gb), 2),
    }


@contextmanager
def _temp_env(env: dict[str, str | None]):
    old: dict[str, str | None] = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _capsule_key(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="bench_glm47_flash_experiments.py",
        description="Run GLM-4.7-Flash speed experiments and write a repro capsule JSON.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Quantize KV cache to N bits during generation (passed to zmlx.validate).",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=None,
        help="Group size for KV cache quantization (passed to zmlx.validate).",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=None,
        help="Step to begin using a quantized KV cache (passed to zmlx.validate).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        required=True,
        help="Output repro capsule path (e.g. benchmarks/repro_capsules/<name>.json)",
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default="benchmarks/matrix.jsonl",
        help="Matrix JSONL ledger path (set to '' to disable)",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=(
            "Variant names to run (default: all). "
            "Choices: control_swiglu_moe, glm47_rope, shared_experts_overlap_streams2"
        ),
    )
    args = parser.parse_args()

    from zmlx import __version__ as zmlx_version
    from zmlx.kv_cache import kv_cache_kwargs
    from zmlx.validate import _bench_config, _compare_tokens, _RunMetrics

    chip, mem_gb = _hardware_meta()
    macos_ver = platform.mac_ver()[0] or platform.platform()
    mlx_ver = getattr(mx, "__version__", "unknown")
    git_short = _git_short_commit()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    has_fused = hasattr(mx, "gather_qmm_swiglu")

    model_id = args.model
    prompt = args.prompt
    runs = int(args.runs)
    max_tokens = int(args.max_tokens)
    gen_kwargs = kv_cache_kwargs(
        kv_bits=args.kv_bits,
        kv_group_size=args.kv_group_size,
        quantized_kv_start=args.quantized_kv_start,
    )

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    meta: dict[str, object] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "device": chip,
        "memory_gb": mem_gb,
        "macos": macos_ver,
        "mlx_version": mlx_ver,
        "zmlx_version": zmlx_version,
        "git_commit": git_short,
        "python": py_ver,
        "has_gather_qmm_swiglu": bool(has_fused),
    }

    # Run baseline once.
    baseline = _bench_config(
        model_path=model_id,
        label="Baseline (unpatched)",
        patterns=[],
        profile=None,
        prompt=prompt,
        max_tokens=max_tokens,
        runs=runs,
        gen_kwargs=gen_kwargs,
    )
    baseline_stats = _summarize_config(baseline)
    b0 = baseline.runs[0] if baseline.runs else _RunMetrics()

    variants: list[dict] = [
        {
            "name": "control_swiglu_moe",
            "patterns": ["swiglu_mlp", "moe_mlp"],
            "env": {},
            "notes": "Control: current best patch set (no RoPE fusion).",
        },
        {
            "name": "control_swiglu_moe_residual_norm",
            "patterns": ["swiglu_mlp", "moe_mlp", "residual_norm"],
            "env": {},
            "notes": (
                "Experimental: residual_norm fusion (breaks greedy token fidelity on "
                "GLM-4.7-Flash in current testing)."
            ),
        },
        {
            "name": "glm47_rope",
            "patterns": ["swiglu_mlp", "moe_mlp", "glm47_rope"],
            "env": {},
            "notes": "Decode-only fused RoPE+concat for glm4_moe_lite attention.",
        },
        {
            "name": "shared_experts_overlap_streams2",
            "patterns": ["swiglu_mlp", "moe_mlp"],
            "env": {
                "ZMLX_MOE_STREAMS": "2",
                "ZMLX_MOE_SHARED_EXPERTS_OVERLAP": "1",
            },
            "notes": "Experimental: overlap shared_experts(x) on a separate stream.",
        },
    ]
    if args.variants is not None:
        wanted = {v.strip() for v in args.variants if v.strip()}
        known = {v["name"] for v in variants}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(f"Unknown variants: {unknown}. Known: {sorted(known)}")
        variants = [v for v in variants if v["name"] in wanted]

    capsule: dict[str, object] = {"meta": meta}

    # Optional matrix logging
    ledger_path = (args.ledger or "").strip() or None
    if ledger_path is not None:
        from zmlx.matrix.models import _infer_architecture, _infer_family
        from zmlx.matrix.schema import MatrixEntry
        from zmlx.matrix.storage import append as matrix_append

        family = _infer_family(model_id)
        architecture = _infer_architecture(model_id, family)
        hw = f"{chip} {mem_gb}GB" if mem_gb else chip

    for v in variants:
        name = v["name"]
        patterns = v["patterns"]
        env = v["env"]
        note = v["notes"]

        cmd = f"python -m zmlx.validate {model_id} --patterns {' '.join(patterns)} --runs {runs} --max-tokens {max_tokens}"
        if args.kv_bits is not None:
            cmd += f" --kv-bits {int(args.kv_bits)}"
        if args.kv_group_size is not None:
            cmd += f" --kv-group-size {int(args.kv_group_size)}"
        if args.quantized_kv_start is not None:
            cmd += f" --quantized-kv-start {int(args.quantized_kv_start)}"
        if env:
            prefix = " ".join(f"{k}={v}" for k, v in env.items())
            cmd = f"{prefix} {cmd}"
        meta[f"command_{name}"] = cmd

        with _temp_env(env):
            patched = _bench_config(
                model_path=model_id,
                label=f"ZMLX Patched ({name})",
                patterns=patterns,
                profile=None,
                prompt=prompt,
                max_tokens=max_tokens,
                runs=runs,
                gen_kwargs=gen_kwargs,
            )

        p0 = patched.runs[0] if patched.runs else _RunMetrics()
        match, total, _ = _compare_tokens(b0, p0)
        fidelity = {
            "matched": int(match),
            "total": int(total),
            "verdict": "PASS" if total > 0 and match == total else "FAIL",
        }

        patched_stats = _summarize_config(patched)
        b_dec = float(baseline_stats["median_decode"])
        p_dec = float(patched_stats["median_decode"])
        b_pre = float(baseline_stats["median_prefill"])
        p_pre = float(patched_stats["median_prefill"])

        modules_patched = int(getattr(patched, "patched_count", 0) or 0)
        patterns_applied = sorted(getattr(patched, "pattern_counts", {}).keys())

        entry = {
            "model": model_id,
            "variant": name,
            "max_tokens": max_tokens,
            "gen_tokens": int(getattr(b0, "gen_tokens", 0) or 0),
            "runs": runs,
            "fidelity": fidelity,
            "baseline": baseline_stats,
            "patched": {
                "patterns": patterns_applied,
                "modules_patched": modules_patched,
                **patched_stats,
            },
            "summary": {
                "decode_speedup": round(p_dec / b_dec, 4) if b_dec > 0 else 0.0,
                "prefill_change": round(p_pre / b_pre - 1.0, 4) if b_pre > 0 else 0.0,
            },
            "notes": note,
        }

        capsule[_capsule_key(name)] = entry

        if ledger_path is not None:
            matrix_entry = MatrixEntry(
                model_id=model_id,
                model_family=family,
                architecture=architecture,
                patterns_applied=patterns_applied,
                hardware=hw,
                notes=f"{note} capsule={json_out.as_posix()}",
                macos_version=macos_ver,
                mlx_version=mlx_ver,
                zmlx_version=zmlx_version,
                zmlx_commit=git_short,
                python_version=platform.python_version(),
                custom_mlx=bool(has_fused),
                timestamp=datetime.now(timezone.utc).isoformat(),
                fidelity=str(fidelity["verdict"]),
                fidelity_detail=f"{match}/{total} tokens identical",
                modules_patched=modules_patched,
                decode_tps_baseline=b_dec,
                decode_tps_patched=p_dec,
                decode_speedup=float(entry["summary"]["decode_speedup"]),
                decode_runs_baseline=list(baseline_stats["decode_tok_s"]),
                decode_runs_patched=list(patched_stats["decode_tok_s"]),
                prefill_tps_baseline=b_pre,
                prefill_tps_patched=p_pre,
                prefill_change=float(entry["summary"]["prefill_change"]),
                peak_mem_baseline_gb=float(baseline_stats["peak_mem_gb"]),
                peak_mem_patched_gb=float(patched_stats["peak_mem_gb"]),
                max_tokens=max_tokens,
                gen_tokens=int(getattr(p0, "gen_tokens", 0) or 0),
                runs=runs,
            )
            matrix_append(matrix_entry, ledger_path)

    json_out.write_text(json.dumps(capsule, indent=2, sort_keys=False) + "\n")
    print(f"\nWrote capsule: {json_out}")


if __name__ == "__main__":
    main()
