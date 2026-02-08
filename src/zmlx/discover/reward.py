"""Reward computation for kernel search."""

from __future__ import annotations

import math


def geometric_mean(values: list[float]) -> float:
    """Compute the geometric mean of positive values.

    Returns 0.0 if the list is empty or contains non-positive values.
    """
    if not values:
        return 0.0
    log_sum = 0.0
    for v in values:
        if v <= 0:
            return 0.0
        log_sum += math.log(v)
    return math.exp(log_sum / len(values))


def compute_reward(timings_us: list[float], baseline_us: float) -> float:
    """Compute reward as baseline / geometric_mean(candidate timings).

    Returns a value clamped to [0, 10].  A reward > 1.0 indicates the
    candidate is faster than baseline.

    Args:
        timings_us: Per-iteration timings in microseconds.
        baseline_us: Baseline median timing in microseconds.

    Returns:
        Reward value in [0, 10].
    """
    if baseline_us <= 0 or not timings_us:
        return 0.0
    gm = geometric_mean(timings_us)
    if gm <= 0:
        return 0.0
    raw = baseline_us / gm
    return max(0.0, min(10.0, raw))
