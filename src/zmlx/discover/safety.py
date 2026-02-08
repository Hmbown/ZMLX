"""Safety utilities for kernel evaluation."""

from __future__ import annotations

import re
import signal
import sys
from collections.abc import Generator
from contextlib import contextmanager


class KernelEvalError(Exception):
    """Base class for kernel evaluation errors."""


class CompilationError(KernelEvalError):
    """Kernel failed to compile."""


class CorrectnessError(KernelEvalError):
    """Kernel produced incorrect results."""


class KernelTimeoutError(KernelEvalError):
    """Kernel evaluation timed out."""


@contextmanager
def eval_timeout(seconds: float) -> Generator[None, None, None]:
    """Context manager that raises KernelTimeoutError after *seconds*.

    Uses ``signal.SIGALRM`` on Unix.  On other platforms (Windows) this is
    a no-op — the caller should use other mechanisms for timeout.
    """
    if sys.platform == "win32" or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum: int, _frame: object) -> None:
        raise KernelTimeoutError(f"Kernel evaluation timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# --- Static source checks ---

_INFINITE_LOOP_RE = re.compile(
    r"\bwhile\s*\(\s*(true|1)\s*\)", re.IGNORECASE
)
_LARGE_THREADGROUP_ALLOC_RE = re.compile(
    r"threadgroup\s+\w+\s+\w+\[(\d+)\]"
)
_MAX_THREADGROUP_ALLOC = 32768  # 32 KB limit (conservative)


def validate_metal_source(source: str) -> list[str]:
    """Run static checks on Metal source code.

    Returns a list of warning strings.  An empty list means no issues found.
    """
    warnings: list[str] = []

    if _INFINITE_LOOP_RE.search(source):
        warnings.append("Potential infinite loop detected: while(true) or while(1)")

    for m in _LARGE_THREADGROUP_ALLOC_RE.finditer(source):
        size = int(m.group(1))
        if size > _MAX_THREADGROUP_ALLOC:
            warnings.append(
                f"Large threadgroup allocation: [{size}] exceeds {_MAX_THREADGROUP_ALLOC}"
            )

    # Check for obviously dangerous patterns
    if "device_memory_size" in source.lower():
        warnings.append("Suspicious access to device_memory_size")

    return warnings
