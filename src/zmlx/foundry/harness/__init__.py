"""Evaluation harness for ZMLX foundry kernel candidates.

Compile, correctness-check, and benchmark Metal kernels against a
reference implementation using a generic ``KernelOp`` protocol.

Public API:
    evaluate_attempt  -- main orchestrator (compile + correctness + bench)
    MLXBackend        -- real Metal backend (requires Apple Silicon + mlx)
    MockBackend       -- in-process mock for CI / non-GPU environments
"""
from __future__ import annotations

from .backend import MLXBackend, MockBackend
from .evaluate import evaluate_attempt

__all__ = [
    "evaluate_attempt",
    "MLXBackend",
    "MockBackend",
]
