"""Promotion gating for kernel discovery results."""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import KernelCandidate


@dataclass
class PromotionPolicy:
    min_speedup_p10: float = 1.01
    noise_guard: float = 0.5
    max_noise_pct: float = 0.2


@dataclass
class ShapeDecision:
    op_name: str
    dtype: str
    shape_signature: dict[str, Any]
    candidate_id: str
    latency_us: float
    speedup_median: float
    speedup_p10: float
    speedup_p90: float
    speedup_noise_pct: float
    ok: bool
    reason: str


@dataclass
class PromotionSelection:
    payload: dict[str, Any]
    decisions: list[ShapeDecision]

    @property
    def promoted_count(self) -> int:
        return len(self.payload.get("entries", []))


@dataclass
class ValidationResult:
    fidelity_ok: bool
    match_count: int
    total: int
    control: dict[str, Any]
    candidate: dict[str, Any]
    median_gen_ratio: float
    median_prompt_ratio: float


def _now_date() -> str:
    return time.strftime("%Y-%m-%d")


def _shape_key(shape_sig: dict[str, Any]) -> str:
    return json.dumps(shape_sig, sort_keys=True, separators=(",", ":"))


def _speedup_stats(case: dict[str, Any]) -> tuple[float, float, float, float]:
    median = float(case.get("speedup_vs_ref", 0.0))
    p10 = float(case.get("speedup_p10", median))
    p90 = float(case.get("speedup_p90", median))
    noise = case.get("speedup_noise_pct")
    if isinstance(noise, (int, float)):
        noise_pct = float(noise)
    elif median > 0 and p90 >= p10:
        noise_pct = float((p90 - p10) / max(1e-9, median))
    else:
        noise_pct = 0.0
    return median, p10, p90, noise_pct


def _decision_for_case(case: dict[str, Any], policy: PromotionPolicy) -> tuple[bool, str]:
    speedup_median, speedup_p10, speedup_p90, noise_pct = _speedup_stats(case)
    min_gain = max(0.0, policy.min_speedup_p10 - 1.0)
    spread = max(0.0, speedup_p90 - speedup_p10)
    required_gain = max(min_gain, spread * policy.noise_guard)
    gain = speedup_median - 1.0

    if speedup_p10 < policy.min_speedup_p10:
        return False, f"speedup_p10={speedup_p10:.3f} < {policy.min_speedup_p10:.3f}"
    if gain < required_gain:
        return False, f"gain={gain:.3f} < noise_guard={required_gain:.3f}"
    if noise_pct > policy.max_noise_pct:
        return False, f"noise_pct={noise_pct:.3f} > {policy.max_noise_pct:.3f}"
    return True, "ok"


def select_promoted_entries(
    candidates: list[KernelCandidate],
    *,
    runtime_env: dict[str, Any] | None = None,
    policy: PromotionPolicy | None = None,
) -> PromotionSelection:
    policy = policy or PromotionPolicy()
    bench = [c for c in candidates if c.status == "benchmarked"]
    bench = sorted(bench, key=lambda cand: cand.candidate_id)
    best_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    decisions: list[ShapeDecision] = []

    for cand in bench:
        dtype_name = str(cand.metrics.get("dtype", cand.notes.get("dtype", "float16")))
        per_shape = cand.metrics.get("per_shape", [])
        if not isinstance(per_shape, list):
            continue
        for case in per_shape:
            if not isinstance(case, dict):
                continue
            shape_sig = case.get("shape", {})
            if not isinstance(shape_sig, dict):
                continue
            latency_us = float(case.get("latency_us", cand.metrics.get("latency_us", float("inf"))))
            speedup_median, speedup_p10, speedup_p90, noise_pct = _speedup_stats(case)
            ok, reason = _decision_for_case(case, policy)
            decisions.append(
                ShapeDecision(
                    op_name=cand.op_name,
                    dtype=dtype_name,
                    shape_signature=shape_sig,
                    candidate_id=cand.candidate_id,
                    latency_us=latency_us,
                    speedup_median=speedup_median,
                    speedup_p10=speedup_p10,
                    speedup_p90=speedup_p90,
                    speedup_noise_pct=noise_pct,
                    ok=ok,
                    reason=reason,
                )
            )
            if not ok:
                continue

            entry_key = (cand.op_name, dtype_name, _shape_key(shape_sig))
            current = best_by_key.get(entry_key)
            score = (-speedup_median, latency_us, cand.candidate_id)
            if current is not None and score >= current["score"]:
                continue

            best_by_key[entry_key] = {
                "score": score,
                "candidate": cand,
                "shape_signature": shape_sig,
                "dtype": dtype_name,
                "latency_us": latency_us,
                "speedup_vs_ref": speedup_median,
                "speedup_p10": speedup_p10,
                "speedup_p90": speedup_p90,
                "speedup_noise_pct": noise_pct,
                "correctness_max_abs_err": case.get(
                    "max_abs_err", cand.metrics.get("correctness_max_abs_err")
                ),
                "correctness_max_rel_err": case.get(
                    "max_rel_err", cand.metrics.get("correctness_max_rel_err")
                ),
            }

    entries: list[dict[str, Any]] = []
    env = runtime_env or {}
    for op_name, dtype_name, shape_key in sorted(best_by_key):
        _ = shape_key
        selected = best_by_key[(op_name, dtype_name, shape_key)]
        cand = selected["candidate"]
        entries.append(
            {
                "key": {
                    "op_name": op_name,
                    "mlx_version": str(env.get("mlx_version", "unknown")),
                    "device_arch": str(env.get("device_arch", "unknown")),
                    "device_name": str(env.get("device_name", "unknown")),
                    "dtype": dtype_name,
                    "shape_signature": selected["shape_signature"],
                },
                "candidate_id": cand.candidate_id,
                "func_name": cand.func_name,
                "metal_source": cand.metal_source,
                "inputs_spec": cand.inputs_spec,
                "outputs_spec": cand.outputs_spec,
                "template_params": cand.template_params,
                "launch_params": cand.launch_params,
                "source_hash": cand.source_hash,
                "metrics": {
                    "latency_us": selected["latency_us"],
                    "speedup_vs_ref": selected["speedup_vs_ref"],
                    "speedup_p10": selected["speedup_p10"],
                    "speedup_p90": selected["speedup_p90"],
                    "speedup_noise_pct": selected["speedup_noise_pct"],
                    "correctness_max_abs_err": selected["correctness_max_abs_err"],
                    "correctness_max_rel_err": selected["correctness_max_rel_err"],
                },
            }
        )

    payload = {
        "schema_version": "2",
        "runtime": env,
        "entries": entries,
    }
    return PromotionSelection(payload=payload, decisions=decisions)


@contextlib.contextmanager
def _temp_env(overrides: dict[str, str | None]) -> Iterator[None]:
    prior: dict[str, str | None] = {}
    for key, value in overrides.items():
        prior[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _summarize_config(result: Any) -> dict[str, Any]:
    return {
        "prompt_tps": [float(r.prompt_tps) for r in result.runs],
        "gen_tps": [float(r.gen_tps) for r in result.runs],
        "median_prompt_tps": float(result.median_prompt_tps),
        "median_gen_tps": float(result.median_gen_tps),
        "peak_mem_gb": float(result.peak_mem_gb),
        "patched_modules": int(result.patched_count),
        "pattern_counts": dict(result.pattern_counts),
    }


def run_promotion_validation(
    *,
    model_id: str,
    patterns: list[str] | None,
    patch_profile: str | None,
    prompt: str,
    max_tokens: int,
    runs: int,
    gen_kwargs: dict[str, Any] | None,
    discovered_path: str | Path,
) -> ValidationResult:
    from zmlx.validate import _bench_config, _compare_tokens, _RunMetrics

    if patterns is not None and patch_profile is not None:
        raise ValueError("Use either patterns or patch_profile, not both.")

    control: Any
    candidate: Any

    with _temp_env({"ZMLX_USE_DISCOVERED_KERNELS": "0", "ZMLX_DISCOVERED_KERNELS_PATH": None}):
        control = _bench_config(
            model_path=model_id,
            label="ZMLX Control",
            patterns=patterns,
            profile=patch_profile,
            prompt=prompt,
            max_tokens=max_tokens,
            runs=runs,
            gen_kwargs=gen_kwargs,
        )

    with _temp_env(
        {
            "ZMLX_USE_DISCOVERED_KERNELS": "1",
            "ZMLX_DISCOVERED_KERNELS_PATH": str(discovered_path),
        }
    ):
        candidate = _bench_config(
            model_path=model_id,
            label="ZMLX Patched + Discovered Kernels",
            patterns=patterns,
            profile=patch_profile,
            prompt=prompt,
            max_tokens=max_tokens,
            runs=runs,
            gen_kwargs=gen_kwargs,
        )

    b_run = control.runs[0] if control.runs else _RunMetrics()
    p_run = candidate.runs[0] if candidate.runs else _RunMetrics()
    match_count, total, _ = _compare_tokens(b_run, p_run)
    fidelity_ok = total > 0 and match_count == total

    control_summary = _summarize_config(control)
    candidate_summary = _summarize_config(candidate)

    median_gen_ratio = (
        candidate_summary["median_gen_tps"] / control_summary["median_gen_tps"]
        if control_summary["median_gen_tps"] > 0
        else 0.0
    )
    median_prompt_ratio = (
        candidate_summary["median_prompt_tps"] / control_summary["median_prompt_tps"]
        if control_summary["median_prompt_tps"] > 0
        else 0.0
    )

    return ValidationResult(
        fidelity_ok=fidelity_ok,
        match_count=int(match_count),
        total=int(total),
        control=control_summary,
        candidate=candidate_summary,
        median_gen_ratio=float(median_gen_ratio),
        median_prompt_ratio=float(median_prompt_ratio),
    )


def build_promotion_capsule(
    *,
    model_id: str,
    patterns: list[str] | None,
    patch_profile: str | None,
    max_tokens: int,
    runs: int,
    prompt: str,
    discovered_path: str | Path,
    validation: ValidationResult,
    note: str = "",
) -> dict[str, Any]:
    patterns_label = patterns if patterns is not None else "(default)"
    if patch_profile:
        patterns_label = f"profile={patch_profile}"
    meta = {
        "date": _now_date(),
        "model": model_id,
        "max_tokens": int(max_tokens),
        "runs": int(runs),
        "patterns": patterns_label,
        "prompt": prompt,
        "discovered_kernels_path": str(discovered_path),
    }
    if note:
        meta["note"] = note
    fidelity_verdict = "PASS" if validation.fidelity_ok else "FAIL"
    capsule = {
        "meta": meta,
        "control": validation.control,
        "discovered": validation.candidate,
        "fidelity": {
            "matched": validation.match_count,
            "total": validation.total,
            "verdict": fidelity_verdict,
        },
        "delta": {
            "median_gen_tps_ratio": validation.median_gen_ratio,
            "median_prompt_tps_ratio": validation.median_prompt_ratio,
            "median_gen_tps_pct": (validation.median_gen_ratio - 1.0) * 100.0,
        },
    }
    return capsule
