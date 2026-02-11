"""Mustache-style ``{{VAR}}`` template renderer for Metal kernel sources.

Adapted from DataFoundry's harness/render.py.  Templates may optionally
contain ``//---HEADER---`` and ``//---BODY---`` markers to split the
source into a header (``#include`` directives) and body (kernel logic).
The ``mx.fast.metal_kernel`` API accepts these separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ids import normalize_source

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedTemplate:
    """The output of template rendering.

    Attributes:
        header: Metal source for ``#include`` / ``using namespace`` preamble.
        source: Metal kernel body (the code that runs per-thread).
        full_text: Combined ``header + //---BODY--- + source`` for logging.
        normalized_full_text: Whitespace-normalized version for stable hashing.
    """

    header: str
    source: str
    full_text: str
    normalized_full_text: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_template(text: str) -> tuple[str, str]:
    """Split raw template text at ``//---HEADER---`` / ``//---BODY---`` markers.

    If either marker is missing the entire text is treated as body and
    the header is empty.  This gracefully handles templates that omit
    the markers.
    """
    if "//---HEADER---" not in text or "//---BODY---" not in text:
        return "", text
    header = text.split("//---HEADER---", 1)[1].split("//---BODY---", 1)[0]
    body = text.split("//---BODY---", 1)[1]
    return header.strip() + "\n", body.strip() + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_template(
    path: Path, values: dict[str, Any]
) -> RenderedTemplate:
    """Render a ``.metal`` template with ``{{KEY}}`` substitution.

    Parameters:
        path: Absolute path to the ``.metal`` template file.
        values: Mapping of placeholder names to replacement values.
                Each value is converted to ``str`` before substitution.

    Returns:
        A ``RenderedTemplate`` with the rendered header, body, combined
        full text, and a normalised copy of the full text suitable for
        cache-key hashing.
    """
    raw = path.read_text(encoding="utf-8")
    header, body = _split_template(raw)

    def _substitute(s: str) -> str:
        out = s
        for k, v in values.items():
            out = out.replace("{{" + k + "}}", str(v))
        return out

    header_rendered = _substitute(header)
    body_rendered = _substitute(body)
    full = f"{header_rendered}\n//---BODY---\n{body_rendered}"
    normalised = normalize_source(full)
    return RenderedTemplate(
        header=header_rendered,
        source=body_rendered,
        full_text=full,
        normalized_full_text=normalised,
    )
