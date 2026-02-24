"""Search policies for knob-space exploration.

Generalized from Discover's RmsnormKnobSpace-specific policies to accept
any ``Dict[str, List[Any]]`` knob space.  The CEM math (smoothing, elite
fraction, minimum probability) is preserved identically.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SampledVariant:
    """A single sampled point in the knob space."""
    knobs: dict[str, Any]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sample_categorical(
    rng: random.Random,
    items: Sequence[Any],
    probs: Sequence[float],
) -> Any:
    """Sample one item from *items* according to *probs*."""
    r = rng.random()
    acc = 0.0
    for item, p in zip(items, probs, strict=False):
        acc += p
        if r <= acc:
            return item
    return items[-1]


# ---------------------------------------------------------------------------
# RandomPolicy
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Uniform-random sampling over a discrete knob space.

    Parameters
    ----------
    knob_space : dict
        Mapping of knob name to its domain (list of allowed values).
        Example: ``{"tg_size": [64, 128, 256], "vec": [1, 2, 4]}``
    """

    def __init__(self, knob_space: dict[str, list[Any]]) -> None:
        self.knob_space = knob_space

    def state(self) -> dict[str, Any]:
        return {"type": "random"}

    def sample(self, rng: random.Random, n: int) -> list[SampledVariant]:
        """Draw *n* independent uniform samples."""
        out: list[SampledVariant] = []
        for _ in range(int(n)):
            knobs: dict[str, Any] = {}
            for key, domain in self.knob_space.items():
                knobs[key] = rng.choice(list(domain))
            out.append(SampledVariant(knobs=knobs))
        return out

    def update(self, variants: Sequence[tuple[SampledVariant, float]]) -> None:
        """No-op: random policy does not learn."""
        return


# ---------------------------------------------------------------------------
# CEMPolicy
# ---------------------------------------------------------------------------

class CEMPolicy:
    """Cross-Entropy Method over independent categorical knob dimensions.

    Maintains a factored probability distribution (one categorical per knob)
    and updates toward elite samples with exponential smoothing.

    Parameters
    ----------
    knob_space : dict
        Mapping of knob name to its domain (list of allowed values).
    elite_frac : float
        Fraction of samples to treat as elite in each update.
    smoothing : float
        Weight on the *old* distribution during update (higher = more
        conservative, slower adaptation).
    min_prob : float
        Floor probability for any value in any dimension, to prevent
        premature convergence and maintain exploration.
    """

    def __init__(
        self,
        knob_space: dict[str, list[Any]],
        *,
        elite_frac: float = 0.25,
        smoothing: float = 0.7,
        min_prob: float = 0.02,
    ) -> None:
        self.knob_space = knob_space
        self.elite_frac = float(elite_frac)
        self.smoothing = float(smoothing)
        self.min_prob = float(min_prob)

        # Initialize uniform distribution over each knob dimension
        self._dist: dict[str, dict[Any, float]] = {}
        for key, domain in knob_space.items():
            n = len(domain)
            self._dist[key] = {v: 1.0 / n for v in domain}

    # -- Serialization / deserialization ------------------------------------

    @staticmethod
    def from_state(
        knob_space: dict[str, list[Any]],
        state: Mapping[str, Any],
    ) -> CEMPolicy:
        """Reconstruct a CEMPolicy from a previously saved state dict."""
        pol = CEMPolicy(
            knob_space,
            elite_frac=float(state.get("elite_frac", 0.25)),
            smoothing=float(state.get("smoothing", 0.7)),
            min_prob=float(state.get("min_prob", 0.02)),
        )
        dist = state.get("dist", {})
        for key, entries in dist.items():
            if key not in pol._dist or not isinstance(entries, list):
                continue
            new: dict[Any, float] = {}
            for e in entries:
                if not isinstance(e, Mapping):
                    continue
                if "value" not in e or "prob" not in e:
                    continue
                new[e["value"]] = float(e["prob"])
            if new:
                pol._dist[key] = new
                pol._renormalize(key)
        return pol

    def state(self) -> dict[str, Any]:
        """Serialize the policy state for persistence."""
        return {
            "type": "cem",
            "elite_frac": self.elite_frac,
            "smoothing": self.smoothing,
            "min_prob": self.min_prob,
            "dist": {
                k: [{"value": kk, "prob": float(vv)} for kk, vv in d.items()]
                for k, d in self._dist.items()
            },
        }

    # -- Sampling -----------------------------------------------------------

    def sample(self, rng: random.Random, n: int) -> list[SampledVariant]:
        """Draw *n* samples from the current distribution."""
        out: list[SampledVariant] = []
        for _ in range(int(n)):
            knobs: dict[str, Any] = {}
            for key, domain in self.knob_space.items():
                knobs[key] = self._sample_key(rng, key, list(domain))
            out.append(SampledVariant(knobs=knobs))
        return out

    # -- Update (the CEM step) ---------------------------------------------

    def update(self, variants: Sequence[tuple[SampledVariant, float]]) -> None:
        """Update the distribution toward elite samples.

        Parameters
        ----------
        variants : sequence of (SampledVariant, reward) tuples
            Higher reward is better.
        """
        if not variants:
            return

        # Sort by reward descending
        sorted_pairs = sorted(variants, key=lambda x: x[1], reverse=True)
        elite_n = max(1, int(len(sorted_pairs) * self.elite_frac))
        elite = sorted_pairs[:elite_n]

        # Count elite frequencies per knob dimension
        counts: dict[str, dict[Any, int]] = {
            key: {} for key in self._dist
        }

        for v, _reward in elite:
            for key in self._dist:
                val = v.knobs.get(key)
                if val is not None:
                    counts[key][val] = counts[key].get(val, 0) + 1

        # Update each categorical with exponential smoothing
        for key, old in self._dist.items():
            total = float(sum(counts[key].values()))
            if total <= 0.0:
                continue

            new: dict[Any, float] = {}
            for k in old.keys():
                target = float(counts[key].get(k, 0)) / total
                new[k] = self.smoothing * old[k] + (1.0 - self.smoothing) * target

            # Enforce minimum probability to maintain exploration
            for k in new:
                new[k] = max(new[k], self.min_prob)

            self._dist[key] = new
            self._renormalize(key)

    # -- Internal helpers ---------------------------------------------------

    def _renormalize(self, key: str) -> None:
        """Re-normalize the distribution for *key* to sum to 1."""
        d = self._dist[key]
        s = float(sum(d.values()))
        if s <= 0:
            n = float(len(d))
            for k in d:
                d[k] = 1.0 / n
            return
        for k in d:
            d[k] = d[k] / s

    def _sample_key(
        self,
        rng: random.Random,
        key: str,
        domain: Sequence[Any],
    ) -> Any:
        """Sample one value for knob *key* from the current distribution."""
        probs = [float(self._dist[key].get(k, 0.0)) for k in domain]
        s = float(sum(probs))
        if s <= 0.0:
            probs = [1.0 / len(domain)] * len(domain)
        else:
            probs = [p / s for p in probs]
        return _sample_categorical(rng, domain, probs)
