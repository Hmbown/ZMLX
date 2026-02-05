"""zmlx.matrix — Kernel test matrix: catalog, run, report."""

from .models import ModelInfo, load_catalog, print_catalog
from .report import print_heatmap, to_csv, to_html
from .schema import MatrixEntry, MatrixSnapshot
from .storage import append, latest, load_all, snapshot

__all__ = [
    "ModelInfo",
    "MatrixEntry",
    "MatrixSnapshot",
    "load_catalog",
    "print_catalog",
    "load_all",
    "latest",
    "snapshot",
    "append",
    "print_heatmap",
    "to_csv",
    "to_html",
]
