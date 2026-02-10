"""Approximate Hamiltonian-tour scheduling for candidate diversity."""

from __future__ import annotations

import random

from .features import centroid, l2_distance
from .graph import KNNGraph
from .types import KernelCandidate


def build_hamiltonian_tour(
    graph: KNNGraph,
    n_nodes: int,
    *,
    seed: int,
) -> list[int]:
    """Build an approximate Hamiltonian tour with greedy nearest walk.

    If the kNN backbone disconnects, this routine allows teleport jumps to the
    nearest unvisited node in global distance space.
    """
    if n_nodes <= 0:
        return []

    rng = random.Random(seed)
    start = rng.randrange(n_nodes)
    unvisited = set(range(n_nodes))
    tour = [start]
    unvisited.remove(start)
    current = start

    while unvisited:
        preferred = [idx for idx in graph.nearest(current) if idx in unvisited]
        if preferred:
            next_idx = min(preferred, key=lambda j: (graph.distance(current, j), j))
        else:
            next_idx = min(unvisited, key=lambda j: (graph.distance(current, j), j))
        tour.append(next_idx)
        unvisited.remove(next_idx)
        current = next_idx

    return tour


def build_union_tours(
    graph: KNNGraph,
    n_nodes: int,
    *,
    n_tours: int,
    seed: int,
) -> list[list[int]]:
    """Build 2-4 diverse tours for union-of-cycles scheduling."""
    if n_nodes <= 0:
        return []
    tours: list[list[int]] = []
    for i in range(max(1, n_tours)):
        tours.append(build_hamiltonian_tour(graph, n_nodes, seed=seed + i * 7919))
    return tours


def interleave_tours(tours: list[list[int]]) -> list[int]:
    """Interleave multiple tours into a single visitation sequence."""
    if not tours:
        return []
    max_len = max(len(t) for t in tours)
    out: list[int] = []
    seen: set[int] = set()
    for pos in range(max_len):
        for tour in tours:
            if pos >= len(tour):
                continue
            idx = tour[pos]
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
    return out


def _position_map(tour: list[int]) -> dict[int, int]:
    return {idx: pos for pos, idx in enumerate(tour)}


def _cyclic_gap(pos_a: int, pos_b: int, length: int) -> int:
    raw = abs(pos_a - pos_b)
    return min(raw, length - raw)


def schedule_batch(
    *,
    candidates: list[KernelCandidate],
    vectors: list[list[float]],
    graph: KNNGraph,
    tours: list[list[int]],
    batch_size: int,
    step: int,
    seed: int,
    exploit_fraction: float = 0.25,
    novelty_fraction: float = 0.25,
    min_tour_gap: int = 3,
) -> list[int]:
    """Pick a diverse batch using tour spacing + exploit/novelty reservations."""
    if batch_size <= 0:
        return []

    unevaluated = {i for i, c in enumerate(candidates) if c.status == "new"}
    if not unevaluated:
        return []

    selected: list[int] = []

    # Exploitation: neighbors of current best.
    scored = [
        (idx, float(c.metrics.get("speedup_vs_ref", 0.0)))
        for idx, c in enumerate(candidates)
        if c.status in {"correct", "benchmarked"}
    ]
    best_idx = max(scored, key=lambda item: item[1])[0] if scored else None

    exploit_quota = int(round(batch_size * exploit_fraction))
    if best_idx is not None and exploit_quota > 0:
        for idx in graph.nearest(best_idx):
            if idx in unevaluated and idx not in selected:
                selected.append(idx)
            if len(selected) >= exploit_quota:
                break

    # Novelty: farthest points from best cluster centroid.
    novelty_quota = int(round(batch_size * novelty_fraction))
    if novelty_quota > 0 and vectors:
        top_scored = sorted(scored, key=lambda item: item[1], reverse=True)
        top_ids = [idx for idx, _ in top_scored[: max(1, len(top_scored) // 4 or 1)]]
        if top_ids:
            center = centroid([vectors[i] for i in top_ids])
            novelty_pool = sorted(i for i in unevaluated if i not in selected)
            novelty_distance = {i: l2_distance(vectors[i], center) for i in novelty_pool}
            far = sorted(
                novelty_pool,
                key=lambda i: (-novelty_distance[i], i),
            )
            for idx in far[:novelty_quota]:
                selected.append(idx)

    # Diversity from interleaved tours.
    order = interleave_tours(tours)
    if not order:
        order = sorted(unevaluated)
    if order:
        offset = step % len(order)
        order = order[offset:] + order[:offset]

    primary = tours[0] if tours else order
    pos = _position_map(primary)

    def _is_spaced(candidate_idx: int) -> bool:
        if candidate_idx not in pos:
            return True
        pa = pos[candidate_idx]
        for chosen in selected:
            if chosen not in pos:
                continue
            pb = pos[chosen]
            if _cyclic_gap(pa, pb, len(primary)) < max(1, min_tour_gap):
                return False
        return True

    for idx in order:
        if idx not in unevaluated or idx in selected:
            continue
        if not _is_spaced(idx):
            continue
        selected.append(idx)
        if len(selected) >= batch_size:
            break

    if len(selected) < batch_size:
        rng = random.Random(seed + step)
        remainder = [i for i in sorted(unevaluated) if i not in selected]
        rng.shuffle(remainder)
        for idx in remainder:
            selected.append(idx)
            if len(selected) >= batch_size:
                break

    return selected[:batch_size]
