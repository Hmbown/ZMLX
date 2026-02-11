"""Foundry reporting modules: coverage analysis and Pareto extraction."""
from __future__ import annotations

from .coverage import build_coverage, write_coverage_reports
from .pareto import best_kernel_by_p50, extract_pareto_front

__all__ = [
    "build_coverage",
    "write_coverage_reports",
    "extract_pareto_front",
    "best_kernel_by_p50",
]
