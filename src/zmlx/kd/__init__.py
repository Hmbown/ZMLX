"""Kernel discovery subsystem for ZMLX."""

from .cli import main
from .types import KernelCandidate

__all__ = ["KernelCandidate", "main"]
