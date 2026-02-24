"""Shape and layout coverage generators for systematic exploration.

Generates deterministic sequences of shapes and layouts to ensure coverage
across the (batch, seq, hidden) space, with occasional odd/adversarial
shapes to test edge cases.
"""
from __future__ import annotations

import random
from typing import Any

# -- Standard shape ladders for LLM inference workloads -----------------------

HIDDEN_SIZES: list[int] = [768, 1024, 1536, 2048, 2560, 3072, 4096, 5120, 6144, 8192]
SEQ_LENS: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
BATCHES: list[int] = [1, 2, 4, 8]

# Edge-case sizes to probe non-power-of-two handling, off-by-one, etc.
ODD_HIDDEN: list[int] = [769, 1025, 1537, 2053, 2561, 3073, 4097]
ODD_SEQ: list[int] = [3, 5, 7, 9, 17, 33, 65, 127, 257, 513, 1025]


def shape_for_index(
    i: int,
    rng: random.Random,
    op: str,
    *,
    extra_dims: dict[str, list[int]] | None = None,
) -> dict[str, int]:
    """Deterministic shape for index *i*, cycling through standard ladders.

    Parameters
    ----------
    i : int
        Iteration index used for deterministic cycling.
    rng : random.Random
        Seeded RNG for occasional odd-shape injection.
    op : str
        Operation name; used to add op-specific shape keys.
    extra_dims : dict, optional
        Additional dimension ladders keyed by name.  Each value list is
        cycled through independently.  E.g. ``{"n_experts": [8, 16, 32]}``.
    """
    use_odd = (i % 11 == 0)  # ~9% of samples get adversarial shapes
    batch = BATCHES[i % len(BATCHES)]

    if use_odd:
        hidden = rng.choice(ODD_HIDDEN)
        seq = rng.choice(ODD_SEQ)
    else:
        hidden = HIDDEN_SIZES[(i // len(SEQ_LENS)) % len(HIDDEN_SIZES)]
        seq = SEQ_LENS[i % len(SEQ_LENS)]

    shape: dict[str, int] = {"batch": int(batch), "seq": int(seq), "hidden": int(hidden)}

    # Op-specific shape augmentations
    if op == "swiglu":
        shape["hidden_in"] = int(2 * hidden)

    # Plug in any caller-supplied extra dimension ladders
    if extra_dims:
        for dim_name, ladder in extra_dims.items():
            shape[dim_name] = ladder[i % len(ladder)]

    return shape


def layout_for_index(i: int, rng: random.Random) -> dict[str, Any]:
    """Deterministic layout for index *i*.

    ~15% of samples are non-contiguous (strided) to exercise Metal
    stride-handling code paths.
    """
    strided = rng.random() < 0.15
    return {"contiguous": (not strided), "strides": []}
