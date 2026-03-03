#!/usr/bin/env python3
"""Reproducible foundry campaign runner for MoE combine templates.

This script runs matched-case template comparisons for ``moe_combine`` and
writes:
1. Per-seed attempt logs.
2. Combined attempt log.
3. Aggregate summary JSON (pairwise wins, sign-test p-values, segments).
4. Human-readable Markdown report.

Example:
    python benchmarks/run_moe_combine_foundry_campaign.py \
      --seeds 20260210 20260211 20260212 20260213 \
      --cases-per-seed 72
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import comb
from pathlib import Path
from statistics import median, pstdev
from typing import Any

from zmlx.foundry.harness.cache import CompileCache
from zmlx.foundry.harness.evaluate import evaluate_attempt
from zmlx.foundry.ops.moe_combine import MoECombineOp
from zmlx.foundry.taxonomy import KernelCandidate

DEFAULT_TEMPLATES = ["t0_basic", "t1_k8_unrolled", "t2_row_tile"]
DEFAULT_SEEDS = [20260210, 20260211, 20260212, 20260213]

# Decode-oriented shape regime.
BATCHES = [1, 2, 4, 8]
SEQS = [1, 2, 4, 8, 16, 32]
HIDDENS = [768, 1024, 1536, 2048, 3072, 4096]
N_EXPERTS = [8, 16, 32, 64]
KS = [2, 4]

# Align with foundry dtype bias (about 45/35/20).
DTYPE_BUCKETS = ["float16"] * 45 + ["bfloat16"] * 35 + ["float32"] * 20

TG_SIZES = [32, 64, 128, 256]
UNROLLS = [1, 2, 4, 8]


@dataclass(frozen=True)
class CampaignConfig:
    seeds: list[int]
    cases_per_seed: int
    templates: list[str]
    backend: str
    correctness_tests: int
    warmup: int
    repeats: int
    bench_timeout_s: float
    out_dir: Path


def _token_bucket(tokens: int) -> str:
    if tokens <= 8:
        return "small_1_8"
    if tokens <= 32:
        return "medium_9_32"
    return "large_33_256"


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    i = (len(xs) - 1) * q
    lo = math.floor(i)
    hi = math.ceil(i)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {}
    xs = [float(v) for v in values]
    q1 = _quantile(xs, 0.25)
    q3 = _quantile(xs, 0.75)
    return {
        "n": len(xs),
        "min": min(xs),
        "p10": _quantile(xs, 0.10),
        "median": median(xs),
        "p90": _quantile(xs, 0.90),
        "max": max(xs),
        "mean": sum(xs) / len(xs),
        "stdev": pstdev(xs) if len(xs) > 1 else 0.0,
        "q1": q1,
        "q3": q3,
        "iqr": (q3 - q1) if q1 is not None and q3 is not None else None,
    }


def _sign_test_pvalue(wins_a: int, wins_b: int) -> float:
    n = wins_a + wins_b
    if n <= 0:
        return 1.0
    m = min(wins_a, wins_b)
    p_low = sum(comb(n, i) for i in range(0, m + 1)) * (0.5**n)
    p_high = sum(comb(n, i) for i in range(n - m, n + 1)) * (0.5**n)
    return float(min(1.0, p_low + p_high))


def _attempt_ok(record: dict[str, Any]) -> bool:
    return bool(
        record.get("build", {}).get("ok")
        and record.get("correctness", {}).get("ok")
        and record.get("bench", {}).get("ok")
    )


def _fmt_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _sample_case(rng: random.Random) -> dict[str, Any]:
    batch = rng.choice(BATCHES)
    seq = rng.choice(SEQS)
    hidden = rng.choice(HIDDENS)
    n_experts = rng.choice(N_EXPERTS)
    k = min(rng.choice(KS), n_experts)
    dtype = rng.choice(DTYPE_BUCKETS)
    layout = {"contiguous": (rng.random() >= 0.18), "strides": []}
    knobs = {
        "tg_size": rng.choice(TG_SIZES),
        "unroll": rng.choice(UNROLLS),
        "fast_math": bool(rng.getrandbits(1)),
        "inject_compile_error": False,
        "inject_incorrect": False,
    }
    shape = {
        "batch": batch,
        "seq": seq,
        "hidden": hidden,
        "n_experts": n_experts,
        "k": k,
    }
    return {"dtype": dtype, "layout": layout, "knobs": knobs, "shape": shape}


def _run_seed(cfg: CampaignConfig, seed: int, op: MoECombineOp) -> list[dict[str, Any]]:
    session_dir = cfg.out_dir / f"seed_{seed}"
    session_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = session_dir / "attempts.ndjson"

    rng = random.Random(seed)
    cache = CompileCache()
    records: list[dict[str, Any]] = []

    with attempts_path.open("w", encoding="utf-8") as f:
        for case_idx in range(cfg.cases_per_seed):
            case = _sample_case(rng)
            order_rng = random.Random((seed * 1_000_003) ^ case_idx)
            order = order_rng.sample(cfg.templates, k=len(cfg.templates))

            for order_pos, template_id in enumerate(order):
                candidate = KernelCandidate(
                    op="moe_combine",
                    dtype=case["dtype"],
                    shape=case["shape"],
                    layout=case["layout"],
                    template_id=template_id,
                    knobs=case["knobs"],
                )
                rec = evaluate_attempt(
                    session_dir=session_dir,
                    backend_name=cfg.backend,
                    candidate=candidate,
                    op=op,
                    cache=cache,
                    correctness_tests=cfg.correctness_tests,
                    correctness_seed=case_idx,
                    warmup=cfg.warmup,
                    repeats=cfg.repeats,
                    bench_timeout_s=cfg.bench_timeout_s,
                )
                rec["campaign"] = {
                    "name": "moe_combine_targeted_pairwise_randorder",
                    "seed": seed,
                    "case_index": case_idx,
                    "order_pos": order_pos,
                    "order": order,
                }
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (case_idx + 1) % 12 == 0:
                attempts = (case_idx + 1) * len(cfg.templates)
                print(
                    f"seed={seed} case={case_idx + 1}/{cfg.cases_per_seed} "
                    f"attempts_written={attempts}",
                    flush=True,
                )

    return records


def _build_summary(
    records: list[dict[str, Any]],
    templates: list[str],
    source_paths: list[str],
) -> dict[str, Any]:
    by_template: dict[str, list[dict[str, Any]]] = {t: [] for t in templates}
    for r in records:
        by_template[r["kernel"]["template_id"]].append(r)

    overall: dict[str, Any] = {}
    for t, recs in by_template.items():
        ok = [r for r in recs if _attempt_ok(r)]
        p50 = [float(r["bench"]["latency_ms"]["p50"]) for r in ok]
        overall[t] = {
            "attempts": len(recs),
            "ok": len(ok),
            "ok_rate": (len(ok) / len(recs)) if recs else 0.0,
            "p50_ms": _stats(p50),
        }

    order_bias: dict[str, Any] = {}
    for t in templates:
        order_bias[t] = {}
        for pos in range(len(templates)):
            vals = [
                float(r["bench"]["latency_ms"]["p50"])
                for r in by_template[t]
                if _attempt_ok(r) and int(r["campaign"]["order_pos"]) == pos
            ]
            order_bias[t][str(pos)] = _stats(vals)

    cases: dict[tuple[int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        c = r["campaign"]
        key = (int(c["seed"]), int(c["case_index"]))
        cases[key][r["kernel"]["template_id"]] = r

    case_rows: list[dict[str, Any]] = []
    for key, data in cases.items():
        if not all(t in data for t in templates):
            continue
        if not all(_attempt_ok(data[t]) for t in templates):
            continue

        p50 = {t: float(data[t]["bench"]["latency_ms"]["p50"]) for t in templates}
        winner = min(p50, key=lambda t: p50[t])
        shape = data[templates[0]]["shape"]
        tokens = int(shape["batch"]) * int(shape["seq"])
        case_rows.append(
            {
                "seed": key[0],
                "case_index": key[1],
                "winner": winner,
                "tokens": tokens,
                "dtype": data[templates[0]]["dtype"],
                "contiguous": bool(data[templates[0]]["layout"].get("contiguous", True)),
                "p50": p50,
            }
        )

    winner_counts = {t: 0 for t in templates}
    for c in case_rows:
        winner_counts[c["winner"]] += 1

    pairwise: dict[str, Any] = {}
    for a, b in combinations(templates, 2):
        wins_a = 0
        wins_b = 0
        ratios: list[float] = []
        for c in case_rows:
            pa = c["p50"][a]
            pb = c["p50"][b]
            if pa < pb:
                wins_a += 1
            elif pb < pa:
                wins_b += 1
            if pa > 0:
                ratios.append(pb / pa)
        n = wins_a + wins_b
        pairwise[f"{a}_vs_{b}"] = {
            "n": n,
            "wins": {a: wins_a, b: wins_b},
            "win_rate": {
                a: (wins_a / n) if n else 0.0,
                b: (wins_b / n) if n else 0.0,
            },
            "ratio_b_over_a": _stats(ratios),
            "sign_test_pvalue": _sign_test_pvalue(wins_a, wins_b),
            "interpretation": f"ratio < 1 favors {b}; ratio > 1 favors {a}",
        }

    segments: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in case_rows:
        segments[("tokens", _token_bucket(int(c["tokens"])))].append(c)
        segments[("dtype", str(c["dtype"]))].append(c)
        layout = "contiguous" if c["contiguous"] else "strided"
        segments[("layout", layout)].append(c)

    segment_out: dict[str, Any] = {}
    for (kind, name), vals in sorted(segments.items()):
        wc = {t: 0 for t in templates}
        for v in vals:
            wc[v["winner"]] += 1
        segment_out[f"{kind}:{name}"] = {
            "n_cases": len(vals),
            "winner_counts": wc,
            "winner_rates": {
                t: (wc[t] / len(vals)) if vals else 0.0 for t in templates
            },
        }

    return {
        "campaign": "moe_combine_targeted_pairwise_randorder",
        "source_paths": source_paths,
        "attempts": len(records),
        "cases": len(case_rows),
        "templates": templates,
        "overall_per_template": overall,
        "order_position_bias": order_bias,
        "winner_counts": winner_counts,
        "winner_rates": {
            t: (winner_counts[t] / len(case_rows)) if case_rows else 0.0
            for t in templates
        },
        "pairwise": pairwise,
        "segments": segment_out,
    }


def _render_report(summary: dict[str, Any]) -> str:
    templates = list(summary["templates"])
    lines: list[str] = []
    lines.append("# MoE Combine Foundry Campaign Report")
    lines.append("")
    lines.append(f"- Attempts: {summary['attempts']}")
    lines.append(f"- Matched cases: {summary['cases']}")
    lines.append("")
    lines.append("## Overall")
    for t in templates:
        o = summary["overall_per_template"][t]
        s = o["p50_ms"]
        med = s.get("median")
        iqr = s.get("iqr")
        lines.append(
            f"- `{t}`: median `{_fmt_float(med)} ms`, IQR `{_fmt_float(iqr)} ms`, "
            f"ok `{o['ok']}/{o['attempts']}`"
        )
    lines.append("")
    lines.append("## Winner Rates")
    for t in templates:
        wc = summary["winner_counts"][t]
        wr = summary["winner_rates"][t]
        lines.append(f"- `{t}`: {wc}/{summary['cases']} ({wr:.1%})")
    lines.append("")
    lines.append("## Pairwise (Sign Test)")
    for key in sorted(summary["pairwise"]):
        p = summary["pairwise"][key]
        a, b = key.split("_vs_")
        ratio_med = p["ratio_b_over_a"].get("median")
        lines.append(
            f"- `{a}` vs `{b}`: wins `{a}={p['wins'][a]}`, `{b}={p['wins'][b]}`, "
            f"median `{b}/{a}={_fmt_float(ratio_med)}`, "
            f"p=`{p['sign_test_pvalue']:.3e}`"
        )
    lines.append("")
    lines.append("## Token Segments")
    for key in sorted(summary["segments"]):
        if not key.startswith("tokens:"):
            continue
        r = summary["segments"][key]["winner_rates"]
        lines.append(
            f"- `{key}`: "
            + ", ".join(f"{t}={r[t]:.1%}" for t in templates)
        )
    return "\n".join(lines) + "\n"


def _default_out_dir() -> Path:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return Path("sessions") / f"foundry_moe_combine_campaign_{ts}"


def _parse_args() -> CampaignConfig:
    parser = argparse.ArgumentParser(
        description="Run reproducible foundry campaign for moe_combine templates."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--cases-per-seed", type=int, default=72)
    parser.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    parser.add_argument("--backend", choices=["mlx", "mock"], default="mlx")
    parser.add_argument("--correctness-tests", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--bench-timeout", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    templates = list(dict.fromkeys(args.templates))
    if len(templates) < 2:
        parser.error("--templates must include at least two unique template IDs")

    out_dir = args.out_dir if args.out_dir is not None else _default_out_dir()
    return CampaignConfig(
        seeds=list(args.seeds),
        cases_per_seed=int(args.cases_per_seed),
        templates=templates,
        backend=str(args.backend),
        correctness_tests=int(args.correctness_tests),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
        bench_timeout_s=float(args.bench_timeout),
        out_dir=out_dir,
    )


def main() -> None:
    cfg = _parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {cfg.out_dir}")
    print(f"Seeds: {cfg.seeds}")
    print(
        f"Templates: {cfg.templates} | backend={cfg.backend} | "
        f"cases/seed={cfg.cases_per_seed}"
    )
    print()

    op = MoECombineOp()
    available_templates = set(op.templates())
    missing = [t for t in cfg.templates if t not in available_templates]
    if missing:
        raise SystemExit(
            "Unknown template(s): "
            + ", ".join(missing)
            + " | available: "
            + ", ".join(sorted(available_templates))
        )

    records: list[dict[str, Any]] = []
    source_paths: list[str] = []

    for seed in cfg.seeds:
        seed_records = _run_seed(cfg, seed, op)
        records.extend(seed_records)
        source_paths.append(str(cfg.out_dir / f"seed_{seed}" / "attempts.ndjson"))

    combined_dir = cfg.out_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_attempts = combined_dir / "attempts.ndjson"
    with combined_attempts.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = _build_summary(records, cfg.templates, source_paths)
    summary_path = combined_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = combined_dir / "report.md"
    report_path.write_text(_render_report(summary), encoding="utf-8")

    print()
    print(f"Wrote attempts: {combined_attempts}")
    print(f"Wrote summary:  {summary_path}")
    print(f"Wrote report:   {report_path}")
    print(f"Winner counts:  {summary['winner_counts']}")


if __name__ == "__main__":
    main()
