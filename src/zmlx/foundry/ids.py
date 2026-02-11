"""Stable identifiers for attempts, cache keys, and shape classes.

Adapted from DataFoundry's logging/ids.py.  Every function is pure
(deterministic, no side effects) and depends only on the stdlib.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Source normalisation
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[ \t]+")


def normalize_source(src: str) -> str:
    """Normalize Metal source for stable hashing.

    * Converts line endings to ``\\n``.
    * Collapses runs of horizontal whitespace in non-indented lines.
    * Strips trailing blank lines.
    """
    src = src.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for line in src.split("\n"):
        line = line.rstrip()
        # Non-indented lines: collapse internal runs of spaces/tabs.
        if line.lstrip() == line:
            line = _WHITESPACE_RE.sub(" ", line)
        lines.append(line)
    # Remove trailing blank lines.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Canonical JSON (deterministic serialisation for hashing)
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


# ---------------------------------------------------------------------------
# Shape class -- human-readable dimension fingerprint
# ---------------------------------------------------------------------------


def shape_class(
    op: str, dtype: str, shape: dict[str, int], layout: dict[str, Any]
) -> str:
    """Replay-stable shape class string.

    Later summarisation can bucket these; we keep exact dims here.
    """
    parts = [f"b{shape.get('batch', 1)}", f"s{shape.get('seq', 1)}"]
    if op == "swiglu":
        parts.append(f"h{shape.get('hidden', 0)}")
        parts.append(f"hin{shape.get('hidden_in', shape.get('hidden', 0) * 2)}")
    else:
        parts.append(f"h{shape.get('hidden', 0)}")
    parts.append(
        "strided" if not bool(layout.get("contiguous", True)) else "contig"
    )
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Attempt ID and compile cache key
# ---------------------------------------------------------------------------


def attempt_id(
    *,
    op: str,
    dtype: str,
    shape_class: str,
    template_id: str,
    knobs: dict[str, Any],
    normalized_source: str,
) -> str:
    payload = {
        "op": op,
        "dtype": dtype,
        "shape_class": shape_class,
        "template_id": template_id,
        "knobs": knobs,
        "source": normalized_source,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def cache_key(
    *,
    op: str,
    dtype: str,
    template_id: str,
    knobs: dict[str, Any],
    normalized_source: str,
) -> str:
    payload = {
        "op": op,
        "dtype": dtype,
        "template_id": template_id,
        "knobs": knobs,
        "source": normalized_source,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
