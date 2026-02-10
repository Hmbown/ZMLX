"""Candidate mutation operators (graph edges)."""

from __future__ import annotations

import random
from typing import Any

from .types import KernelCandidate


def _mutate_key(
    current: dict[str, Any],
    space: dict[str, tuple[Any, ...]],
    key: str,
    rng: random.Random,
) -> dict[str, Any]:
    values = list(space[key])
    if not values:
        return dict(current)

    new = dict(current)
    cur = current.get(key)
    if cur not in values:
        new[key] = values[0]
        return new

    idx = values.index(cur)
    if len(values) == 1:
        return new

    if isinstance(cur, (int, float)) and len(values) >= 2:
        moves = []
        if idx > 0:
            moves.append(values[idx - 1])
        if idx + 1 < len(values):
            moves.append(values[idx + 1])
        if moves:
            new[key] = rng.choice(moves)
            return new

    candidates = [value for value in values if value != cur]
    new[key] = rng.choice(candidates)
    return new


def initial_population(
    *,
    op_module: Any,
    shape: dict[str, Any],
    dtype_name: str,
    seed: int,
    count: int,
) -> list[KernelCandidate]:
    """Generate deterministic initial candidates for one op/shape/dtype."""
    rng = random.Random(seed)
    out: list[KernelCandidate] = []
    seen: set[str] = set()

    base_tpl = op_module.seed_template_params()
    base_launch = op_module.seed_launch_params()

    seed_candidate = op_module.make_candidate(
        template_params=base_tpl,
        launch_params=base_launch,
        shape=shape,
        dtype_name=dtype_name,
        parent_id=None,
    )
    out.append(seed_candidate)
    seen.add(seed_candidate.candidate_id)

    attempts = 0
    while len(out) < max(1, count) and attempts < max(100, count * 20):
        attempts += 1
        tpl = dict(base_tpl)
        launch = dict(base_launch)

        n_tpl = rng.randint(1, max(1, len(op_module.TEMPLATE_PARAM_SPACE)))
        n_launch = rng.randint(1, max(1, len(op_module.LAUNCH_PARAM_SPACE)))

        tpl_keys = rng.sample(sorted(op_module.TEMPLATE_PARAM_SPACE), k=n_tpl)
        launch_keys = rng.sample(sorted(op_module.LAUNCH_PARAM_SPACE), k=n_launch)

        for key in tpl_keys:
            tpl = _mutate_key(tpl, op_module.TEMPLATE_PARAM_SPACE, key, rng)
        for key in launch_keys:
            launch = _mutate_key(launch, op_module.LAUNCH_PARAM_SPACE, key, rng)

        candidate = op_module.make_candidate(
            template_params=tpl,
            launch_params=launch,
            shape=shape,
            dtype_name=dtype_name,
            parent_id=seed_candidate.candidate_id,
        )
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        out.append(candidate)

    return out


def neighbor_mutations(
    *,
    parent: KernelCandidate,
    op_module: Any,
    shape: dict[str, Any],
    dtype_name: str,
    seed: int,
    count: int,
) -> list[KernelCandidate]:
    """Generate neighbor candidates by mutating template/launch knobs."""
    rng = random.Random(seed)
    out: list[KernelCandidate] = []
    seen = {parent.candidate_id}

    for _ in range(max(1, count * 5)):
        if len(out) >= count:
            break

        tpl = dict(parent.template_params)
        launch = dict(parent.launch_params)

        n_tpl = rng.randint(1, max(1, len(op_module.TEMPLATE_PARAM_SPACE)))
        n_launch = rng.randint(0, max(1, len(op_module.LAUNCH_PARAM_SPACE)))

        tpl_keys = rng.sample(sorted(op_module.TEMPLATE_PARAM_SPACE), k=n_tpl)
        for key in tpl_keys:
            tpl = _mutate_key(tpl, op_module.TEMPLATE_PARAM_SPACE, key, rng)

        if n_launch > 0:
            launch_keys = rng.sample(sorted(op_module.LAUNCH_PARAM_SPACE), k=n_launch)
            for key in launch_keys:
                launch = _mutate_key(launch, op_module.LAUNCH_PARAM_SPACE, key, rng)

        candidate = op_module.make_candidate(
            template_params=tpl,
            launch_params=launch,
            shape=shape,
            dtype_name=dtype_name,
            parent_id=parent.candidate_id,
        )
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        out.append(candidate)

    return out
