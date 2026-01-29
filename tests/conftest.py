import platform

import pytest


def _metal_available() -> bool:
    try:
        import mlx.core as mx
    except Exception:
        return False
    metal = getattr(mx, "metal", None)
    if metal is None:
        return False
    fn = getattr(metal, "is_available", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False

def pytest_collection_modifyitems(config, items):
    # Skip Metal-dependent tests when not on macOS arm64 or when Metal isn't available.
    is_macos = platform.system() == "Darwin"
    is_arm64 = platform.machine() in ("arm64", "aarch64")
    has_metal = _metal_available()

    if not (is_macos and is_arm64 and has_metal):
        skip = pytest.mark.skip(reason="Metal kernel tests require macOS arm64 with Metal available.")
        for item in items:
            item.add_marker(skip)
