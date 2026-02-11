"""Generalized kernel candidate sampler.

Adapted from DataFoundry's hardcoded op-to-knobs dispatch to accept a
generic knob_space dict from the op registry.  Supports four sampling
modes: random, coverage, mutation, and mix (cyclic blend of all three).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ndjson import iter_records
from ..taxonomy import KernelCandidate
from .coverage import layout_for_index, shape_for_index
from .mutate import mutate

# ---------------------------------------------------------------------------
# Dtype sampling
# ---------------------------------------------------------------------------

DTYPES: list[str] = ["float16", "bfloat16", "float32"]


def dtype_for_index(i: int, rng: random.Random) -> str:
    """Biased dtype sampling: ~45% fp16, ~35% bf16, ~20% fp32."""
    r = rng.random()
    if r < 0.45:
        return "float16"
    if r < 0.80:
        return "bfloat16"
    return "float32"


# ---------------------------------------------------------------------------
# Elite loading (for mutation mode)
# ---------------------------------------------------------------------------

def load_elite(
    session_dir: Path,
    op: str,
    *,
    top_k: int = 32,
) -> list[dict[str, Any]]:
    """Load the top-k fastest correct attempts from the session log.

    Returns a list of dicts with ``template_id`` and ``knobs`` keys.
    """
    path = session_dir / "attempts.ndjson"
    if not path.exists():
        return []

    good: list[tuple[float, dict[str, Any]]] = []
    for rec in iter_records(path):
        if rec.get("op") != op:
            continue
        if not rec.get("correctness", {}).get("ok", False):
            continue
        bench = rec.get("bench", {})
        if not bench.get("ok", False):
            continue
        p50 = bench.get("latency_ms", {}).get("p50")
        if p50 is None:
            continue
        try:
            p50f = float(p50)
        except (TypeError, ValueError):
            continue
        good.append((p50f, rec))

    good.sort(key=lambda t: t[0])
    elites: list[dict[str, Any]] = []
    for _, rec in good[:top_k]:
        elites.append({
            "template_id": rec.get("kernel", {}).get("template_id"),
            "knobs": rec.get("kernel", {}).get("knobs", {}),
        })
    return elites


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

@dataclass
class Sampler:
    """Generates KernelCandidate instances using configurable strategies.

    Parameters
    ----------
    op : str
        Operation name (e.g. "rmsnorm", "swiglu").
    knob_space : dict
        Mapping of knob name to list of valid values for that knob.
        Obtained from the op registry rather than hardcoded here.
    templates : list of str
        Available template IDs for this op.
    mode : str
        Sampling mode: "random", "coverage", "mutation", or "mix".
    seed : int
        Base seed for deterministic sampling.
    session_dir : Path
        Session directory for loading elite candidates (mutation mode).
    extra_shape_dims : dict, optional
        Extra dimension ladders passed through to shape_for_index.
    """

    op: str
    knob_space: dict[str, list[Any]]
    templates: list[str]
    mode: str  # random | coverage | mutation | mix
    seed: int
    session_dir: Path
    extra_shape_dims: dict[str, list[int]] | None = None

    # Internal state (set in __post_init__)
    rng: random.Random = field(init=False, repr=False)
    _mix_cycle: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def sample_knobs_random(self, template_id: str) -> dict[str, Any]:
        """Uniform random sample from the knob space."""
        knobs: dict[str, Any] = {}
        for key, domain in self.knob_space.items():
            knobs[key] = self.rng.choice(list(domain))
        return knobs

    def sample_knobs_coverage(self, template_id: str, i: int) -> dict[str, Any]:
        """Stratified sweep through knob values for systematic coverage.

        Cycles through each dimension's domain at different rates using
        prime-offset strides, ensuring broad coverage over many iterations.
        """
        knobs: dict[str, Any] = {}
        # Use different stride primes per dimension to avoid alignment
        stride_primes = [1, 3, 7, 11, 13, 17, 19, 23, 29, 31]
        for idx, (key, domain) in enumerate(self.knob_space.items()):
            stride = stride_primes[idx % len(stride_primes)]
            knobs[key] = domain[(i * stride) % len(domain)]

        # Low-probability fault injection for dataset diversity
        knobs["inject_compile_error"] = (i % 97 == 0)
        knobs["inject_incorrect"] = (i % 113 == 0) and not knobs["inject_compile_error"]
        return knobs

    def next_candidate(self, i: int) -> KernelCandidate:
        """Generate the *i*-th candidate.

        The mode determines how shapes, layouts, and knobs are selected:
        - **random**: all dimensions sampled uniformly
        - **coverage**: deterministic cycling for systematic exploration
        - **mutation**: perturb elite candidates
        - **mix**: cycles coverage -> random -> random -> mutation
        """
        mode = self.mode
        if mode == "mix":
            cyc = self._mix_cycle % 4
            self._mix_cycle += 1
            mode = ["coverage", "random", "random", "mutation"][cyc]

        dtype = dtype_for_index(i, self.rng)
        template_id = self.templates[i % len(self.templates)]

        if mode == "coverage":
            shape = shape_for_index(
                i, self.rng, self.op, extra_dims=self.extra_shape_dims,
            )
            layout = layout_for_index(i, self.rng)
            knobs = self.sample_knobs_coverage(template_id, i)

        elif mode == "mutation":
            elites = load_elite(self.session_dir, self.op, top_k=32)
            if elites:
                base = self.rng.choice(elites)
                template_id = base.get("template_id") or template_id
                knobs = mutate(
                    base.get("knobs", {}),
                    self.knob_space,
                    self.rng,
                )
            else:
                # No elites yet -- fall back to random knobs
                knobs = self.sample_knobs_random(template_id)
            # Mutation also samples new shapes to maintain shape coverage
            shape = shape_for_index(
                i, self.rng, self.op, extra_dims=self.extra_shape_dims,
            )
            layout = layout_for_index(i, self.rng)

        else:
            # random
            shape = shape_for_index(
                i, self.rng, self.op, extra_dims=self.extra_shape_dims,
            )
            layout = layout_for_index(i, self.rng)
            knobs = self.sample_knobs_random(template_id)

        return KernelCandidate(
            op=self.op,
            dtype=dtype,
            shape=shape,
            layout=layout,
            template_id=template_id,
            knobs=knobs,
        )
