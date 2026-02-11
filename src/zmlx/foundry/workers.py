"""Multi-process worker orchestration for foundry runs.

Spawns N workers that each run an independent evaluate loop over
candidates from a shared op/scheduler configuration.  Each worker
writes to its own NDJSON shard (``attempts.worker<N>.ndjson``) to
avoid contention; after all workers complete, logs are merged into
a single deduplicated ``attempts.ndjson``.

Adapted from DataFoundry's ``run.py`` multi-worker path.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .harness.cache import CompileCache
from .harness.evaluate import evaluate_attempt, write_record
from .ndjson import merge_worker_logs
from .ops import get_registry
from .sampling.sampler import Sampler
from .scheduler import CurriculumScheduler
from .session import Session
from .templates import list_templates

# ---------------------------------------------------------------------------
# Worker config (picklable across processes)
# ---------------------------------------------------------------------------


@dataclass
class WorkerConfig:
    """Serializable configuration for a single worker process."""

    session_dir: str
    worker_id: int
    num_workers: int
    ops: list[str]
    n_attempts: int
    mode: str  # random | coverage | mutation | mix
    seed: int
    backend: str  # mlx | mock
    correctness_tests: int
    warmup: int
    repeats: int
    bench_timeout_s: float
    stage: int


# ---------------------------------------------------------------------------
# Single-worker loop
# ---------------------------------------------------------------------------


def run_worker(cfg: WorkerConfig) -> dict[str, Any]:
    """Execute the evaluation loop for one worker.

    This function is designed to be the target of ``mp.Process`` or
    called directly for single-worker runs.

    Returns a summary dict with ``worker_id``, ``n_attempted``,
    ``n_written``, ``n_skipped``, and ``elapsed_s``.
    """
    session = Session(
        session_dir=Path(cfg.session_dir),
        worker_id=cfg.worker_id,
        num_workers=cfg.num_workers,
    )

    registry = get_registry()
    scheduler = CurriculumScheduler(ops=cfg.ops)
    scheduler.set_stage(cfg.stage)
    available = scheduler.available_ops()

    cache = CompileCache()
    t0 = time.monotonic()
    n_attempted = 0
    n_written = 0
    n_skipped = 0

    for op_name in available:
        op = registry.get(op_name)
        if op is None:
            continue

        templates = list_templates(op_name)
        if not templates:
            templates = ["ref"]

        # Build a merged, flattened knob space from all templates.
        # Op.knob_space(template_id) returns {"knob": {"type":..., "values":[...]}}
        # Sampler expects {"knob": [val1, val2, ...]}
        merged_knob_space: dict[str, list[Any]] = {}
        if hasattr(op, "knob_space"):
            for tid in templates:
                try:
                    raw = op.knob_space(tid)
                except Exception:
                    continue
                for k, v in raw.items():
                    if isinstance(v, dict) and "values" in v:
                        vals = list(v["values"])
                    elif isinstance(v, dict) and v.get("type") == "bool":
                        vals = [True, False]
                    elif isinstance(v, list):
                        vals = list(v)
                    else:
                        continue
                    if k not in merged_knob_space:
                        merged_knob_space[k] = vals
                    else:
                        # Union of values, preserving order
                        existing = set(merged_knob_space[k])
                        for val in vals:
                            if val not in existing:
                                merged_knob_space[k].append(val)
                                existing.add(val)

        sampler = Sampler(
            op=op_name,
            knob_space=merged_knob_space,
            templates=templates,
            mode=cfg.mode,
            seed=cfg.seed + cfg.worker_id * 1_000_000,
            session_dir=session.session_dir,
            extra_shape_dims=getattr(op, "extra_shape_dims", None),
        )

        per_op_budget = max(1, cfg.n_attempts // max(len(available), 1))

        for i in range(per_op_budget):
            candidate = sampler.next_candidate(i)

            # Shard: only this worker handles candidates where
            # hash(attempt_index) % num_workers == worker_id
            global_idx = n_attempted
            if cfg.num_workers > 1 and (global_idx % cfg.num_workers) != cfg.worker_id:
                n_attempted += 1
                n_skipped += 1
                continue

            try:
                record = evaluate_attempt(
                    session_dir=session.session_dir,
                    backend_name=cfg.backend,
                    candidate=candidate,
                    op=op,
                    cache=cache,
                    correctness_tests=cfg.correctness_tests,
                    warmup=cfg.warmup,
                    repeats=cfg.repeats,
                    bench_timeout_s=cfg.bench_timeout_s,
                )
            except Exception as exc:
                record = {
                    "id": f"error_{cfg.worker_id}_{n_attempted}",
                    "op": op_name,
                    "error": str(exc),
                    "build": {"ok": False},
                    "correctness": {"ok": False},
                    "bench": {"ok": False},
                }

            rid = record.get("id", "")
            if session.try_claim(rid):
                write_record(
                    session_dir=session.session_dir,
                    record=record,
                    worker_id=cfg.worker_id if cfg.num_workers > 1 else None,
                )
                n_written += 1

            n_attempted += 1

    elapsed = time.monotonic() - t0
    return {
        "worker_id": cfg.worker_id,
        "n_attempted": n_attempted,
        "n_written": n_written,
        "n_skipped": n_skipped,
        "elapsed_s": round(elapsed, 2),
    }


def _worker_entry(cfg: WorkerConfig) -> None:
    """Entry point for spawned worker processes (writes result to stdout)."""
    import json
    result = run_worker(cfg)
    print(json.dumps(result), flush=True)


# ---------------------------------------------------------------------------
# Multi-worker orchestrator
# ---------------------------------------------------------------------------


def spawn_workers(
    *,
    session_dir: str,
    ops: list[str],
    n_attempts: int,
    num_workers: int = 1,
    mode: str = "mix",
    seed: int = 42,
    backend: str = "mlx",
    correctness_tests: int = 3,
    warmup: int = 10,
    repeats: int = 50,
    bench_timeout_s: float = 10.0,
    stage: int = 4,
) -> dict[str, Any]:
    """Run a foundry session with *num_workers* parallel workers.

    For ``num_workers == 1``, runs in the current process (no subprocess
    overhead).  For ``num_workers > 1``, spawns child processes and merges
    the per-worker NDJSON shards on completion.

    Returns a summary dict with ``session_dir``, ``workers`` (list of
    per-worker results), ``merged_path``, and ``total_elapsed_s``.
    """
    session_path = Path(session_dir)
    session_path.mkdir(parents=True, exist_ok=True)

    configs = [
        WorkerConfig(
            session_dir=session_dir,
            worker_id=wid,
            num_workers=num_workers,
            ops=ops,
            n_attempts=n_attempts,
            mode=mode,
            seed=seed,
            backend=backend,
            correctness_tests=correctness_tests,
            warmup=warmup,
            repeats=repeats,
            bench_timeout_s=bench_timeout_s,
            stage=stage,
        )
        for wid in range(num_workers)
    ]

    t0 = time.monotonic()

    if num_workers == 1:
        results = [run_worker(configs[0])]
    else:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(run_worker, configs)

    # Merge worker shards
    merged = merge_worker_logs(session_path)
    total_elapsed = time.monotonic() - t0

    return {
        "session_dir": str(session_path),
        "num_workers": num_workers,
        "workers": results,
        "merged_path": str(merged),
        "total_elapsed_s": round(total_elapsed, 2),
    }
