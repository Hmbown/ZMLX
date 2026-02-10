"""Core data types for kernel discovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

CandidateStatus = Literal["new", "compiled", "correct", "benchmarked", "failed"]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(data: Any) -> str:
    payload = _canonical_json(data).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class KernelCandidate:
    """A concrete kernel variant with source and launch parameters."""

    op_name: str
    candidate_id: str
    metal_source: str
    func_name: str
    inputs_spec: list[dict[str, Any]]
    outputs_spec: list[dict[str, Any]]
    template_params: dict[str, Any]
    launch_params: dict[str, Any]
    features: dict[str, float] = field(default_factory=dict)
    status: CandidateStatus = "new"
    metrics: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            self.candidate_id = self.build_candidate_id(
                op_name=self.op_name,
                metal_source=self.metal_source,
                func_name=self.func_name,
                inputs_spec=self.inputs_spec,
                outputs_spec=self.outputs_spec,
                template_params=self.template_params,
                launch_params=self.launch_params,
            )

    @staticmethod
    def build_candidate_id(
        *,
        op_name: str,
        metal_source: str,
        func_name: str,
        inputs_spec: list[dict[str, Any]],
        outputs_spec: list[dict[str, Any]],
        template_params: dict[str, Any],
        launch_params: dict[str, Any],
    ) -> str:
        """Create a deterministic candidate ID from normalized kernel descriptors."""
        normalized = {
            "op_name": op_name,
            "metal_source": metal_source,
            "func_name": func_name,
            "inputs_spec": inputs_spec,
            "outputs_spec": outputs_spec,
            "template_params": template_params,
            "launch_params": launch_params,
        }
        digest = _stable_hash(normalized)
        return f"{op_name}_{digest[:16]}"

    @property
    def source_hash(self) -> str:
        return _stable_hash({"metal_source": self.metal_source})

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_name": self.op_name,
            "candidate_id": self.candidate_id,
            "metal_source": self.metal_source,
            "func_name": self.func_name,
            "inputs_spec": self.inputs_spec,
            "outputs_spec": self.outputs_spec,
            "template_params": self.template_params,
            "launch_params": self.launch_params,
            "features": self.features,
            "status": self.status,
            "metrics": self.metrics,
            "parent_id": self.parent_id,
            "notes": self.notes,
            "source_hash": self.source_hash,
        }
