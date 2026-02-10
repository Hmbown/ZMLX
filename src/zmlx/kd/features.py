"""Feature extraction for candidate graph scheduling."""

from __future__ import annotations

import math
from typing import Any

from .types import KernelCandidate


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0


def candidate_feature_map(candidate: KernelCandidate) -> dict[str, float]:
    """Normalize parameters/metrics into a flat numeric feature map."""
    feat: dict[str, float] = {}

    for key, value in sorted(candidate.template_params.items()):
        feat[f"tp.{key}"] = _as_float(value)
    for key, value in sorted(candidate.launch_params.items()):
        feat[f"lp.{key}"] = _as_float(value)
    for key, value in sorted(candidate.features.items()):
        feat[f"fx.{key}"] = _as_float(value)

    latency = _as_float(candidate.metrics.get("latency_us", 0.0))
    speedup = _as_float(candidate.metrics.get("speedup_vs_ref", 0.0))
    compiled = 1.0 if candidate.status in {"compiled", "correct", "benchmarked"} else 0.0
    benchmarked = 1.0 if candidate.status == "benchmarked" else 0.0

    feat["mt.latency_us"] = latency
    feat["mt.speedup_vs_ref"] = speedup
    feat["mt.compiled"] = compiled
    feat["mt.benchmarked"] = benchmarked

    if latency > 0:
        feat["mt.inv_latency"] = 1.0 / latency
    else:
        feat["mt.inv_latency"] = 0.0

    return feat


def collect_feature_keys(candidates: list[KernelCandidate]) -> list[str]:
    keys: set[str] = set()
    for cand in candidates:
        keys.update(candidate_feature_map(cand).keys())
    return sorted(keys)


def candidate_vector(candidate: KernelCandidate, keys: list[str]) -> list[float]:
    feat = candidate_feature_map(candidate)
    return [feat.get(k, 0.0) for k in keys]


def l2_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("distance vectors must have same dimensionality")
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b, strict=True)))


def cosine_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("distance vectors must have same dimensionality")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos = dot / (na * nb)
    cos = max(-1.0, min(1.0, cos))
    return 1.0 - cos


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError("all vectors must share dimensionality")
        for i, value in enumerate(vec):
            out[i] += value
    return [value / len(vectors) for value in out]
