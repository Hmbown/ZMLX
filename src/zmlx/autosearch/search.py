"""Evolutionary and exhaustive search engines for kernel autotuning."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from .harness import ScoredConfig
from .space import ConfigSpace


class EvolutionarySearch:
    """Measurement-driven evolutionary search over a config space."""

    def __init__(
        self,
        space: ConfigSpace,
        template_fn: Callable[[dict[str, Any]], tuple[str, tuple[int, int, int], tuple[int, int, int]]],
        population_size: int = 24,
        elite_frac: float = 0.25,
        mutation_rate: float = 0.3,
        seed: int = 42,
    ):
        self.space = space
        self.template_fn = template_fn
        self.pop_size = population_size
        self.elite_frac = elite_frac
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)

    def initialize(self) -> list[dict[str, Any]]:
        """Create initial population: default + random valid configs."""
        population = [self.space.default_config()]
        seen = {self.space.config_key(population[0])}

        while len(population) < self.pop_size:
            config = self.space.sample_random(self.rng)
            key = self.space.config_key(config)
            if key not in seen:
                seen.add(key)
                population.append(config)

        return population

    def run(
        self,
        evaluate_fn: Callable[[dict[str, Any], str, tuple[int, int, int], tuple[int, int, int]], ScoredConfig],
        generations: int = 10,
        verbose: bool = False,
    ) -> list[ScoredConfig]:
        """Run evolutionary search.

        Args:
            evaluate_fn: (config, source, grid, threadgroup) -> ScoredConfig
            generations: Number of generations to run.
            verbose: Print progress.

        Returns:
            All scored configs sorted by reward (descending).
        """
        all_scored: list[ScoredConfig] = []
        seen_keys: set[tuple[Any, ...]] = set()

        # Initialize
        population = self.initialize()

        for gen in range(generations):
            if verbose:
                print(f"  Generation {gen + 1}/{generations} ({len(population)} candidates)")

            # Evaluate new configs
            gen_scored: list[ScoredConfig] = []
            for config in population:
                key = self.space.config_key(config)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                source, grid, tg = self.template_fn(config)
                sc = evaluate_fn(config, source, grid, tg)
                gen_scored.append(sc)
                all_scored.append(sc)

                if verbose and sc.eval_result.correct:
                    print(f"    {_fmt_config(config)} -> {sc.speedup:.2f}x "
                          f"({sc.eval_result.median_us:.1f}us)")
                elif verbose:
                    err = sc.eval_result.compile_error or sc.eval_result.correctness_error or "?"
                    print(f"    {_fmt_config(config)} -> FAIL: {err[:50]}")

            if not gen_scored:
                if verbose:
                    print("    (all configs already evaluated)")
                continue

            # Select elites from all scored so far
            valid = [s for s in all_scored if s.eval_result.correct]
            if not valid:
                # All failed — just re-randomize
                population = [self.space.sample_random(self.rng) for _ in range(self.pop_size)]
                continue

            valid.sort(key=lambda s: s.reward, reverse=True)
            n_elite = max(1, int(len(valid) * self.elite_frac))
            elites = valid[:n_elite]

            if verbose:
                best = elites[0]
                print(f"    Best so far: {best.speedup:.2f}x ({_fmt_config(best.config)})")

            # Next generation
            population = []
            # Keep elites
            for e in elites:
                population.append(e.config)

            # Mutations of elites
            while len(population) < int(self.pop_size * 0.6):
                parent = self.rng.choice(elites)
                child = self.space.mutate(parent.config, self.rng, self.mutation_rate)
                population.append(child)

            # Crossovers
            while len(population) < int(self.pop_size * 0.85):
                a, b = self.rng.sample(elites, min(2, len(elites)))
                if a is b and len(elites) > 1:
                    b = self.rng.choice(elites)
                child = self.space.crossover(a.config, b.config, self.rng)
                population.append(child)

            # Random fill
            while len(population) < self.pop_size:
                population.append(self.space.sample_random(self.rng))

        all_scored.sort(key=lambda s: s.reward, reverse=True)
        return all_scored


class ExhaustiveSearch:
    """Try every valid config in the space."""

    def __init__(
        self,
        space: ConfigSpace,
        template_fn: Callable[[dict[str, Any]], tuple[str, tuple[int, int, int], tuple[int, int, int]]],
    ):
        self.space = space
        self.template_fn = template_fn

    def run(
        self,
        evaluate_fn: Callable[[dict[str, Any], str, tuple[int, int, int], tuple[int, int, int]], ScoredConfig],
        verbose: bool = False,
    ) -> list[ScoredConfig]:
        """Evaluate all valid configs."""
        configs = self.space.enumerate_all()
        if verbose:
            print(f"  Exhaustive search: {len(configs)} valid configs")

        results: list[ScoredConfig] = []
        for i, config in enumerate(configs):
            source, grid, tg = self.template_fn(config)
            sc = evaluate_fn(config, source, grid, tg)
            results.append(sc)

            if verbose and sc.eval_result.correct:
                print(f"    [{i + 1}/{len(configs)}] {_fmt_config(config)} -> "
                      f"{sc.speedup:.2f}x ({sc.eval_result.median_us:.1f}us)")
            elif verbose:
                err = sc.eval_result.compile_error or sc.eval_result.correctness_error or "?"
                print(f"    [{i + 1}/{len(configs)}] {_fmt_config(config)} -> FAIL: {err[:50]}")

        results.sort(key=lambda s: s.reward, reverse=True)
        return results


def auto_search(
    space: ConfigSpace,
    template_fn: Callable[[dict[str, Any]], tuple[str, tuple[int, int, int], tuple[int, int, int]]],
    evaluate_fn: Callable[[dict[str, Any], str, tuple[int, int, int], tuple[int, int, int]], ScoredConfig],
    *,
    generations: int = 10,
    population_size: int = 24,
    exhaustive_threshold: int = 200,
    seed: int = 42,
    verbose: bool = False,
) -> list[ScoredConfig]:
    """Auto-select search strategy and run.

    Uses exhaustive search if the valid config count is below threshold,
    otherwise evolutionary search.
    """
    all_valid = space.enumerate_all()
    if len(all_valid) <= exhaustive_threshold:
        if verbose:
            print(f"  Config space has {len(all_valid)} valid configs — using exhaustive search")
        searcher = ExhaustiveSearch(space, template_fn)
        return searcher.run(evaluate_fn, verbose=verbose)
    else:
        if verbose:
            print(f"  Config space has {len(all_valid)}+ configs — using evolutionary search "
                  f"(pop={population_size}, gens={generations})")
        searcher_evo = EvolutionarySearch(
            space, template_fn,
            population_size=population_size,
            seed=seed,
        )
        return searcher_evo.run(evaluate_fn, generations=generations, verbose=verbose)


def _fmt_config(config: dict[str, Any]) -> str:
    """Format a config dict as a compact string."""
    parts = []
    for k, v in config.items():
        if isinstance(v, bool):
            parts.append(f"{k}={'Y' if v else 'N'}")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)
