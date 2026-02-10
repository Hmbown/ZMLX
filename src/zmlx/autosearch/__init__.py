"""ZMLX Autosearch: structured kernel autotuner (no LLM required).

Usage::

    python -m zmlx.autosearch run swiglu --D 1536 --generations 10
    python -m zmlx.autosearch list
    python -m zmlx.autosearch export swiglu --D 1536 -o best.py
"""

from __future__ import annotations

from .harness import Harness, ScoredConfig
from .search import EvolutionarySearch, ExhaustiveSearch, auto_search
from .space import ConfigSpace, Knob
from .templates import TEMPLATES, get_space, get_template_fn

__all__ = [
    "ConfigSpace",
    "EvolutionarySearch",
    "ExhaustiveSearch",
    "Harness",
    "Knob",
    "ScoredConfig",
    "TEMPLATES",
    "auto_search",
    "get_space",
    "get_template_fn",
]
