"""Run archive, lineage tracking, and NDJSON logging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import KernelCandidate


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class RunArchive:
    """Persist candidate lineage and per-step evaluations."""

    out_dir: Path
    op_name: str
    seed: int
    budget: int
    dtype_name: str
    shape_suite: str
    runtime_env: dict[str, Any]

    candidates: dict[str, KernelCandidate] = field(default_factory=dict)
    lineage: dict[str, list[str]] = field(default_factory=dict)
    evaluated_order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ndjson_path = self.out_dir / "run.ndjson"
        self.meta_path = self.out_dir / "run_meta.json"

        meta = {
            "schema_version": "1",
            "created_at": _now_iso(),
            "op_name": self.op_name,
            "seed": self.seed,
            "budget": self.budget,
            "dtype": self.dtype_name,
            "shape_suite": self.shape_suite,
            "runtime": self.runtime_env,
        }
        self.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def register_candidate(self, candidate: KernelCandidate) -> bool:
        if candidate.candidate_id in self.candidates:
            return False
        self.candidates[candidate.candidate_id] = candidate
        parent = candidate.parent_id
        if parent is not None:
            self.lineage.setdefault(parent, []).append(candidate.candidate_id)
        return True

    def get_candidate(self, candidate_id: str) -> KernelCandidate | None:
        return self.candidates.get(candidate_id)

    def all_candidates(self) -> list[KernelCandidate]:
        return sorted(self.candidates.values(), key=lambda cand: cand.candidate_id)

    def append_event(self, record: dict[str, Any]) -> None:
        with self.ndjson_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def log_evaluation(self, *, step: int, candidate: KernelCandidate) -> None:
        self.evaluated_order.append(candidate.candidate_id)
        record = {
            "ts": _now_iso(),
            "event": "evaluation",
            "step": int(step),
            "seed": int(self.seed),
            "op_name": candidate.op_name,
            "candidate_id": candidate.candidate_id,
            "parent_id": candidate.parent_id,
            "status": candidate.status,
            "func_name": candidate.func_name,
            "metal_source": candidate.metal_source,
            "inputs_spec": candidate.inputs_spec,
            "outputs_spec": candidate.outputs_spec,
            "template_params": candidate.template_params,
            "launch_params": candidate.launch_params,
            "features": candidate.features,
            "metrics": candidate.metrics,
            "notes": candidate.notes,
        }
        self.append_event(record)

    def log_failure(
        self,
        *,
        step: int,
        candidate: KernelCandidate,
        reason: str,
    ) -> None:
        record = {
            "ts": _now_iso(),
            "event": "failure",
            "step": int(step),
            "seed": int(self.seed),
            "op_name": candidate.op_name,
            "candidate_id": candidate.candidate_id,
            "parent_id": candidate.parent_id,
            "reason": reason,
            "status": candidate.status,
            "template_params": candidate.template_params,
            "launch_params": candidate.launch_params,
            "metrics": candidate.metrics,
        }
        self.append_event(record)

    def top_benchmarked(self, limit: int = 10) -> list[KernelCandidate]:
        bench = [c for c in self.candidates.values() if c.status == "benchmarked"]
        bench.sort(key=lambda c: (float(c.metrics.get("latency_us", float("inf"))), c.candidate_id))
        return bench[:limit]
