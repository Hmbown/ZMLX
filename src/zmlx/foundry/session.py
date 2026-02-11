"""Session management for foundry runs.

A Session owns a session directory, manages NDJSON logging, compile-cache
tracking, and existing-ID dedup.  It combines the session-setup patterns
from datafoundry's ``run.py``, the kernel-foundry's ``SessionWriter``, and
the general NDJSON utilities already in this package.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ndjson import append_record, load_existing_ids


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp string."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@dataclass
class Session:
    """Manages a single foundry session on disk.

    Directory layout::

        <session_dir>/
            attempts.ndjson          # primary log
            attempts.worker*.ndjson  # per-worker shards (multi-worker)
            kernels/                 # saved Metal source files
            cache/                   # compile-cache metadata
            reports/                 # generated reports

    Parameters
    ----------
    session_dir : Path
        Root directory for this session.
    worker_id : int
        Worker ID for multi-worker runs (0 for single-worker).
    num_workers : int
        Total number of workers.
    """

    session_dir: Path
    worker_id: int = 0
    num_workers: int = 1

    # Internal state
    _existing_ids: set[str] = field(init=False, repr=False, default_factory=set)
    _compile_cache: dict[str, bool] = field(init=False, repr=False, default_factory=dict)
    _written: int = field(init=False, repr=False, default=0)
    _created_at: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        self.session_dir = Path(self.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "kernels").mkdir(exist_ok=True)
        (self.session_dir / "cache").mkdir(exist_ok=True)
        (self.session_dir / "reports").mkdir(exist_ok=True)

        self._existing_ids = load_existing_ids(self.session_dir)
        self._created_at = _utc_now_iso()

    # -- Properties ---------------------------------------------------------

    @property
    def attempts_path(self) -> Path:
        """Path to this worker's NDJSON log."""
        if self.num_workers > 1:
            return self.session_dir / f"attempts.worker{self.worker_id}.ndjson"
        return self.session_dir / "attempts.ndjson"

    @property
    def kernels_dir(self) -> Path:
        return self.session_dir / "kernels"

    @property
    def n_existing(self) -> int:
        """Number of previously seen attempt IDs."""
        return len(self._existing_ids)

    @property
    def n_written(self) -> int:
        """Number of records written in this session instance."""
        return self._written

    # -- Dedup / claim ------------------------------------------------------

    def is_known(self, attempt_id: str) -> bool:
        """Check whether an attempt ID has already been recorded."""
        return attempt_id in self._existing_ids

    def try_claim(self, attempt_id: str) -> bool:
        """Atomically claim an attempt ID.  Returns True if newly claimed."""
        if attempt_id in self._existing_ids:
            return False
        self._existing_ids.add(attempt_id)
        return True

    # -- Writing records ----------------------------------------------------

    def write_record(self, record: dict[str, Any]) -> None:
        """Append *record* to the session's NDJSON log.

        Automatically stamps ``session``, ``worker_id``, and ``created_at``
        if not already present.
        """
        record.setdefault("session", self.session_dir.name)
        record.setdefault("worker_id", self.worker_id)
        record.setdefault("created_at", _utc_now_iso())
        append_record(self.attempts_path, record)
        self._written += 1

    def save_kernel_source(self, attempt_id: str, source: str) -> Path:
        """Persist Metal source to ``kernels/<attempt_id>.metal``."""
        path = self.kernels_dir / f"{attempt_id}.metal"
        path.write_text(source, encoding="utf-8")
        return path

    # -- Compile cache helpers ----------------------------------------------

    def cache_key(self, **kwargs: Any) -> str:
        """Compute a SHA-256 compile-cache key from arbitrary keyword args."""
        payload = json.dumps(kwargs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def is_cached(self, key: str) -> bool:
        """Check if a compile result is cached (in-memory fast path)."""
        if key in self._compile_cache:
            return True
        # Check on-disk marker
        marker = self.session_dir / "cache" / f"{key}.ok"
        if marker.exists():
            self._compile_cache[key] = True
            return True
        return False

    def mark_cached(self, key: str) -> None:
        """Record that a compile succeeded for *key*."""
        self._compile_cache[key] = True
        marker = self.session_dir / "cache" / f"{key}.ok"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()

    # -- Session metadata ---------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Summary metadata for reporting."""
        return {
            "session_dir": str(self.session_dir),
            "session_name": self.session_dir.name,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "created_at": self._created_at,
            "n_existing": self.n_existing,
            "n_written": self.n_written,
        }
