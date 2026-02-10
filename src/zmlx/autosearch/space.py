"""Config space with typed knobs, constraints, and mutation operators."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Knob:
    """A single tunable parameter with a finite set of valid values."""

    name: str
    values: tuple[Any, ...]
    default: Any

    def __post_init__(self) -> None:
        if self.default not in self.values:
            raise ValueError(f"default {self.default!r} not in values {self.values}")


@dataclass(frozen=True)
class ConfigSpace:
    """Parameterized search space with hard constraints."""

    knobs: tuple[Knob, ...]
    constraints: tuple[Callable[[dict[str, Any]], bool], ...] = ()

    @property
    def total_configs(self) -> int:
        """Upper bound on configs (before constraint filtering)."""
        n = 1
        for k in self.knobs:
            n *= len(k.values)
        return n

    def default_config(self) -> dict[str, Any]:
        return {k.name: k.default for k in self.knobs}

    def is_valid(self, config: dict[str, Any]) -> bool:
        for c in self.constraints:
            if not c(config):
                return False
        return True

    def sample_random(self, rng: random.Random) -> dict[str, Any]:
        """Sample a random valid config (retries up to 100 times)."""
        for _ in range(100):
            config = {k.name: rng.choice(k.values) for k in self.knobs}
            if self.is_valid(config):
                return config
        return self.default_config()

    def mutate(self, config: dict[str, Any], rng: random.Random,
               rate: float = 0.3) -> dict[str, Any]:
        """Mutate one or more knobs with probability ``rate`` each."""
        new = dict(config)
        for k in self.knobs:
            if rng.random() < rate:
                new[k.name] = rng.choice(k.values)
        for _ in range(50):
            if self.is_valid(new):
                return new
            # Re-mutate invalid configs
            new = dict(config)
            for k in self.knobs:
                if rng.random() < rate:
                    new[k.name] = rng.choice(k.values)
        return config  # fallback to original

    def crossover(self, a: dict[str, Any], b: dict[str, Any],
                  rng: random.Random) -> dict[str, Any]:
        """Uniform crossover: pick each knob from a or b with equal probability."""
        child = {}
        for k in self.knobs:
            child[k.name] = a[k.name] if rng.random() < 0.5 else b[k.name]
        if self.is_valid(child):
            return child
        # Fallback: try swapping one knob at a time
        for k in self.knobs:
            child[k.name] = b[k.name] if child[k.name] == a[k.name] else a[k.name]
            if self.is_valid(child):
                return child
        return a  # give up

    def enumerate_all(self) -> list[dict[str, Any]]:
        """Enumerate all valid configs. Only sensible for small spaces."""
        configs: list[dict[str, Any]] = [{}]
        for k in self.knobs:
            new_configs = []
            for c in configs:
                for v in k.values:
                    nc = dict(c)
                    nc[k.name] = v
                    new_configs.append(nc)
            configs = new_configs
        return [c for c in configs if self.is_valid(c)]

    def config_key(self, config: dict[str, Any]) -> tuple[Any, ...]:
        """Hashable key for deduplication."""
        return tuple(config[k.name] for k in self.knobs)
