"""ZMLX Discover: LLM-guided Metal kernel search.

Usage::

    from zmlx.discover import TARGETS, MockBackend, run_search

    # List targets
    for name in TARGETS:
        print(name)

    # Quick test with mock backend
    python -m zmlx.discover search moe_combine --llm mock --steps 2 -v
"""

from __future__ import annotations

from .candidates import (
    EvalResult,
    InputSpec,
    KernelCandidate,
    KernelSpec,
    OutputSpec,
    SearchSpace,
)
from .llm import ClaudeBackend, LLMBackend, MockBackend, OpenAIBackend
from .session import Session
from .targets import TARGETS

__all__ = [
    "ClaudeBackend",
    "EvalResult",
    "InputSpec",
    "KernelCandidate",
    "KernelSpec",
    "LLMBackend",
    "MockBackend",
    "OpenAIBackend",
    "OutputSpec",
    "SearchSpace",
    "Session",
    "TARGETS",
]
