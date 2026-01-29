from __future__ import annotations

import platform
from typing import Any


def is_macos() -> bool:
    return platform.system() == "Darwin"

def is_arm64() -> bool:
    # On Apple silicon this is usually 'arm64'
    return platform.machine() in ("arm64", "aarch64")

def is_supported_host() -> bool:
    # MLX PyPI is primarily Apple-silicon macOS, but MLX can also have CUDA/CPU backends on Linux.
    # For this repo's Metal path, we require macOS + arm64.
    return is_macos() and is_arm64()

def import_mx() -> Any:
    """Import mlx.core lazily to keep import errors friendly."""
    import mlx.core as mx  # type: ignore
    return mx
