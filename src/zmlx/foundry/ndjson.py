"""Append-only NDJSON log for attempt records.

Adapted from DataFoundry's logging/ndjson.py.  Each record is a single
JSON line, fsynced after write so partial records from crashes are
confined to the last line (which ``iter_records`` tolerates).
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def dumps(record: dict[str, Any]) -> str:
    """Compact, deterministic-order JSON serialisation."""
    return json.dumps(
        record,
        sort_keys=False,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = dumps(record) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Iterate NDJSON records, tolerating partial/corrupt lines."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_existing_ids(session_dir: Path) -> set[str]:
    """Collect all attempt IDs already recorded in a session directory."""
    ids: set[str] = set()
    if not session_dir.exists():
        return ids
    candidates = [session_dir / "attempts.ndjson"]
    candidates += sorted(session_dir.glob("attempts.worker*.ndjson"))
    for p in candidates:
        for rec in iter_records(p):
            rid = rec.get("id")
            if isinstance(rid, str):
                ids.add(rid)
    return ids


def merge_worker_logs(
    session_dir: Path, *, out_name: str = "attempts.ndjson"
) -> Path:
    """Merge per-worker shards into a single deduplicated log."""
    session_dir.mkdir(parents=True, exist_ok=True)
    out_path = session_dir / out_name

    seen: set[str] = set()
    for rec in iter_records(out_path):
        rid = rec.get("id")
        if isinstance(rid, str):
            seen.add(rid)

    worker_paths = sorted(session_dir.glob("attempts.worker*.ndjson"))
    if not worker_paths:
        return out_path

    with open(out_path, "a", encoding="utf-8") as out:
        for wp in worker_paths:
            for rec in iter_records(wp):
                rid = rec.get("id")
                if not isinstance(rid, str) or rid in seen:
                    continue
                out.write(dumps(rec) + "\n")
                seen.add(rid)
        out.flush()
        os.fsync(out.fileno())
    return out_path
