"""ZMLX Foundry -- kernel template system and evaluation harness.

The foundry unifies four worktree repos (Discover, Lab, Foundry, DataFoundry)
into a single module for Metal kernel dataset generation, evaluation, and
training data export.

Submodules:
    taxonomy    -- Foundation types (KernelClass, OpSpec, KernelCandidate, etc.)
    ids         -- Stable identifiers for attempts and cache keys
    ndjson      -- Append-only NDJSON logging with crash tolerance
    ops         -- Op registry (16 ops across 9 kernel classes)
    templates   -- Metal template discovery, loading, and rendering
    harness     -- Compile, correctness-check, and benchmark orchestrator
    sampling    -- Random/coverage/mutation/CEM candidate sampling
    session     -- Session directory management and NDJSON logging
    scheduler   -- Curriculum-based staged op unlocking
    plugins     -- Protocol-based extensibility (entry-point and local)
    reports     -- Coverage analysis and Pareto extraction
    export      -- Training JSONL export
    workers     -- Multi-process worker orchestration

CLI:
    python -m zmlx.foundry run      # generate kernel dataset
    python -m zmlx.foundry report   # coverage + pareto reports
    python -m zmlx.foundry export   # export training JSONL
    python -m zmlx.foundry list     # list registered ops
"""
from __future__ import annotations

__all__ = [
    # Foundation
    "taxonomy",
    "ids",
    "ndjson",
    # Ops
    "ops",
    # Templates
    "templates",
    # Harness
    "harness",
    # Sampling
    "sampling",
    # Session & scheduling
    "session",
    "scheduler",
    # Plugins
    "plugins",
    # Reports & export
    "reports",
    "export",
    # Workers
    "workers",
]
