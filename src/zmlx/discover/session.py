"""Session persistence for kernel search."""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionMetadata:
    """Metadata about a search session."""

    session_id: str = ""
    target_name: str = ""
    llm_backend: str = ""
    device_chip: str = ""
    device_memory_gb: int = 0
    os_version: str = ""
    started_at: str = ""
    updated_at: str = ""
    total_steps: int = 0
    total_candidates: int = 0
    total_evaluated: int = 0
    best_reward: float = 0.0
    best_speedup: float = 0.0
    best_source: str = ""
    baseline_us: float = 0.0


@dataclass
class Session:
    """Persistent session state for a kernel search."""

    schema_version: str = "1.0"
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    tree_data: dict[str, Any] = field(default_factory=dict)
    eval_history: list[dict[str, Any]] = field(default_factory=list)
    candidate_sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def new(cls, target_name: str, backend: str, device_info: dict[str, Any]) -> Session:
        """Create a new session with populated metadata."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        return cls(
            metadata=SessionMetadata(
                session_id=uuid.uuid4().hex[:16],
                target_name=target_name,
                llm_backend=backend,
                device_chip=device_info.get("chip", ""),
                device_memory_gb=device_info.get("memory_gb", 0),
                os_version=f"{platform.system()} {platform.release()}",
                started_at=now,
                updated_at=now,
            ),
        )

    def save(self, path: str | Path) -> None:
        """Save session to a JSON file."""
        self.metadata.updated_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        data = {
            "schema_version": self.schema_version,
            "metadata": asdict(self.metadata),
            "tree_data": self.tree_data,
            "eval_history": self.eval_history,
            "candidate_sources": self.candidate_sources,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Session:
        """Load session from a JSON file."""
        data = json.loads(Path(path).read_text())
        meta_d = data.get("metadata", {})
        meta = SessionMetadata(**meta_d)
        return cls(
            schema_version=data.get("schema_version", "1.0"),
            metadata=meta,
            tree_data=data.get("tree_data", {}),
            eval_history=data.get("eval_history", []),
            candidate_sources=data.get("candidate_sources", {}),
        )
