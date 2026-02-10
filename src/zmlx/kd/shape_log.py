"""Runtime shape logging for kernel discovery."""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_ENV_SHAPE_LOG = "ZMLX_KD_SHAPE_LOG"
_ENV_SHAPE_LOG_MODE = "ZMLX_KD_SHAPE_LOG_MODE"
_ENV_SHAPE_LOG_FLUSH_EVERY = "ZMLX_KD_SHAPE_LOG_FLUSH_EVERY"
_ENV_SHAPE_LOG_MAX = "ZMLX_KD_SHAPE_LOG_MAX"

_lock = threading.Lock()
_counts: dict[tuple[str, str, str], int] = {}
_event_count = 0
_registered = False


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _log_path() -> Path | None:
    raw = os.environ.get(_ENV_SHAPE_LOG, "").strip()
    return Path(raw) if raw else None


def _mode() -> str:
    raw = os.environ.get(_ENV_SHAPE_LOG_MODE, "count").strip().lower()
    return raw if raw in {"count", "event"} else "count"


def _flush_every() -> int:
    raw = os.environ.get(_ENV_SHAPE_LOG_FLUSH_EVERY, "").strip()
    if not raw:
        return 256
    try:
        return max(1, int(raw))
    except Exception:
        return 256


def _max_events() -> int | None:
    raw = os.environ.get(_ENV_SHAPE_LOG_MAX, "").strip()
    if not raw:
        return None
    try:
        return max(1, int(raw))
    except Exception:
        return None


def _canon_shape(shape_signature: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(shape_signature):
        value = shape_signature[key]
        try:
            out[str(key)] = int(value)
        except Exception:
            out[str(key)] = str(value)
    return out


def _append_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _flush_counts() -> None:
    path = _log_path()
    if path is None:
        return
    records: list[dict[str, Any]] = []
    for (op_name, dtype_name, shape_key), count in _counts.items():
        try:
            shape_sig = json.loads(shape_key)
        except Exception:
            shape_sig = {}
        records.append(
            {
                "ts": _now_iso(),
                "op_name": op_name,
                "dtype": dtype_name,
                "shape_signature": shape_sig,
                "count": int(count),
            }
        )
    _append_records(path, records)
    _counts.clear()


def _ensure_atexit() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    atexit.register(_flush_counts)


def record_shape(
    op_name: str,
    dtype: str,
    shape_signature: dict[str, Any],
    *,
    source: str | None = None,
) -> None:
    path = _log_path()
    if path is None:
        return
    if not isinstance(shape_signature, dict):
        return

    sig = _canon_shape(shape_signature)
    shape_key = json.dumps(sig, sort_keys=True, separators=(",", ":"))
    key = (str(op_name), str(dtype), shape_key)
    mode = _mode()
    max_events = _max_events()

    with _lock:
        global _event_count
        if max_events is not None and _event_count >= max_events:
            return
        _event_count += 1

        if mode == "event":
            record = {
                "ts": _now_iso(),
                "op_name": str(op_name),
                "dtype": str(dtype),
                "shape_signature": sig,
                "count": 1,
            }
            if source:
                record["source"] = source
            _append_records(path, [record])
            return

        _counts[key] = _counts.get(key, 0) + 1

        if _event_count % _flush_every() == 0:
            _flush_counts()
        _ensure_atexit()
