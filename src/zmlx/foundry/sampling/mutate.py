"""Knob mutation for elite-evolution sampling.

Operates on a generic knob space (Dict[str, List[Any]]) rather than
hardcoded per-op types.  A single knob field is perturbed per mutation
by stepping +/- 1 in its ordered value list.  Boolean knobs are toggled.
"""
from __future__ import annotations

import copy
import random
from typing import Any


def _mutate_choice(rng: random.Random, val: Any, choices: list[Any]) -> Any:
    """Step the current value one position in *choices* (wrapping)."""
    if val not in choices:
        return rng.choice(choices)
    idx = choices.index(val)
    step = rng.choice([-1, 1])
    return choices[(idx + step) % len(choices)]


def mutate(
    knobs: dict[str, Any],
    knob_space: dict[str, list[Any]],
    rng: random.Random,
    *,
    inject_error_prob: float = 0.02,
    inject_incorrect_prob: float = 0.04,
) -> dict[str, Any]:
    """Mutate one random knob field from *knobs*, guided by *knob_space*.

    Parameters
    ----------
    knobs : dict
        Current knob values (will be deep-copied, not mutated in place).
    knob_space : dict
        Mapping of knob name to its ordered domain (list of valid values).
        Boolean knobs are detected automatically and toggled.
    rng : random.Random
        Seeded RNG for reproducibility.
    inject_error_prob : float
        Probability of injecting a deliberate compile error (for dataset
        diversity -- training on failures is valuable).
    inject_incorrect_prob : float
        Probability of injecting a deliberate correctness error.

    Returns
    -------
    dict
        New knobs dict with exactly one field mutated.
    """
    k = copy.deepcopy(knobs)

    # Only mutate fields that exist in both the knob space and the current knobs
    mutable_fields = [f for f in knob_space if f in k]
    if not mutable_fields:
        # Nothing to mutate -- return a copy with possible fault injection
        return _maybe_inject(k, rng, inject_error_prob, inject_incorrect_prob)

    field = rng.choice(mutable_fields)
    domain = list(knob_space[field])

    # Boolean toggle shortcut
    if set(domain) == {True, False} or set(domain) == {0, 1}:
        k[field] = not bool(k.get(field, False))
    else:
        k[field] = _mutate_choice(rng, k.get(field), domain)

    return _maybe_inject(k, rng, inject_error_prob, inject_incorrect_prob)


def _maybe_inject(
    k: dict[str, Any],
    rng: random.Random,
    error_prob: float,
    incorrect_prob: float,
) -> dict[str, Any]:
    """Optionally inject compile-error or correctness-error flags."""
    r = rng.random()
    if r < error_prob:
        k["inject_compile_error"] = True
        k["inject_incorrect"] = False
    elif r < error_prob + incorrect_prob:
        k["inject_compile_error"] = False
        k["inject_incorrect"] = True
    else:
        k["inject_compile_error"] = False
        k["inject_incorrect"] = False
    return k
