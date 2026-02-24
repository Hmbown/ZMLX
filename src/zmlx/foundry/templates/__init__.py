"""Template discovery and loading for foundry Metal kernel templates.

Each op has a subdirectory (e.g. ``templates/rmsnorm/``) containing one
or more ``.metal`` template files.  Template IDs are derived from the
filename stem (e.g. ``t0_basic.metal`` -> ``"t0_basic"``).
"""
from __future__ import annotations

from pathlib import Path

# The templates directory is this file's parent.
_TEMPLATES_DIR = Path(__file__).resolve().parent


def get_template_path(op: str, template_id: str) -> Path:
    """Return the absolute Path to a ``.metal`` template file.

    Raises ``FileNotFoundError`` if the template does not exist on disk.
    """
    path = _TEMPLATES_DIR / op / f"{template_id}.metal"
    if not path.exists():
        raise FileNotFoundError(
            f"Template not found: {path}  "
            f"(op={op!r}, template_id={template_id!r})"
        )
    return path


def list_templates(op: str) -> list[str]:
    """Return sorted template IDs available for *op*.

    An empty list is returned if the op subdirectory does not exist.
    """
    op_dir = _TEMPLATES_DIR / op
    if not op_dir.is_dir():
        return []
    return sorted(p.stem for p in op_dir.glob("*.metal"))


def load_template(op: str, template_id: str) -> str:
    """Read and return the raw text of a template file."""
    path = get_template_path(op, template_id)
    return path.read_text(encoding="utf-8")
