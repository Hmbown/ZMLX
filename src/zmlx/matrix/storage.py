"""JSONL append-only ledger for matrix entries."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schema import MatrixEntry, MatrixSnapshot

_DEFAULT_LEDGER = "benchmarks/matrix.jsonl"


def _resolve_path(path: str | None = None) -> Path:
    if path is not None:
        return Path(path)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    return repo_root / _DEFAULT_LEDGER


def append(entry: MatrixEntry, path: str | None = None) -> None:
    """Append one MatrixEntry as a JSON line to the ledger."""
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry.to_dict(), separators=(",", ":")) + "\n")


def load_all(path: str | None = None) -> list[MatrixEntry]:
    """Read all entries from the JSONL ledger."""
    p = _resolve_path(path)
    if not p.exists():
        return []
    entries: list[MatrixEntry] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(MatrixEntry.from_dict(json.loads(line)))
    return entries


def latest(path: str | None = None) -> dict[tuple[str, str], MatrixEntry]:
    """Return the latest entry per (model_id, hardware) key.

    Later entries in the file override earlier ones.
    """
    entries = load_all(path)
    result: dict[tuple[str, str], MatrixEntry] = {}
    for e in entries:
        result[(e.model_id, e.hardware)] = e
    return result


def snapshot(
    path: str | None = None,
    hardware: str | None = None,
) -> MatrixSnapshot:
    """Build a MatrixSnapshot from the ledger, optionally filtered by hardware."""
    lat = latest(path)
    entries = list(lat.values())
    if hardware:
        entries = [e for e in entries if e.hardware == hardware]
    entries.sort(key=lambda e: (e.model_family, e.model_id))
    date = ""
    if entries:
        date = max(e.timestamp for e in entries if e.timestamp) or ""
    return MatrixSnapshot(entries=entries, hardware=hardware or "", date=date)


def clear(path: str | None = None) -> None:
    """Remove the ledger file."""
    p = _resolve_path(path)
    if p.exists():
        os.remove(p)
