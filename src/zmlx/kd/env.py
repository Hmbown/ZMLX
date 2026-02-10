"""Runtime environment fingerprinting for kernel pinning."""

from __future__ import annotations

from importlib import metadata
from typing import Any


def _mlx_version() -> str:
    try:
        return metadata.version("mlx")
    except Exception:
        return "unknown"


def runtime_fingerprint() -> dict[str, Any]:
    """Return the runtime selector for discovered-kernel pinning."""
    try:
        import mlx.core as mx

        device = mx.device_info()
        device_name = str(device.get("device_name", "unknown"))
        device_arch = str(device.get("architecture", "unknown"))
    except Exception:
        device_name = "unknown"
        device_arch = "unknown"

    return {
        "mlx_version": _mlx_version(),
        "device_name": device_name,
        "device_arch": device_arch,
    }
