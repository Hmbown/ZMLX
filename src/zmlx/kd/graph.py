"""kNN graph construction for candidate scheduling."""

from __future__ import annotations

from dataclasses import dataclass

from .features import cosine_distance


@dataclass
class KNNGraph:
    """A dense distance-aware k-nearest-neighbor graph."""

    neighbors: dict[int, list[int]]
    distances: dict[tuple[int, int], float]

    def distance(self, i: int, j: int) -> float:
        if i == j:
            return 0.0
        key = (i, j) if (i, j) in self.distances else (j, i)
        return self.distances.get(key, float("inf"))

    def nearest(self, i: int) -> list[int]:
        return self.neighbors.get(i, [])


def build_knn_graph(vectors: list[list[float]], k: int = 8) -> KNNGraph:
    """Build a kNN graph with cosine distance in feature space."""
    n = len(vectors)
    if n == 0:
        return KNNGraph(neighbors={}, distances={})

    k_eff = max(1, min(k, n - 1)) if n > 1 else 0
    distances: dict[tuple[int, int], float] = {}
    neighbors: dict[int, list[int]] = {}

    for i in range(n):
        pairs: list[tuple[float, int]] = []
        for j in range(n):
            if i == j:
                continue
            d = cosine_distance(vectors[i], vectors[j])
            distances[(i, j)] = d
            pairs.append((d, j))
        pairs.sort(key=lambda item: item[0])
        neighbors[i] = [idx for _, idx in pairs[:k_eff]]

    return KNNGraph(neighbors=neighbors, distances=distances)
