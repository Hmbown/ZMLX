from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "KVTC requires numpy. Install with: pip install zmlx[kvtc]"
    ) from e


def to_numpy(x: Any) -> np.ndarray:
    """Convert common tensor types to a NumPy array without copying when possible.

    Supports:
    - numpy.ndarray
    - mlx.core.array (via .to_numpy())
    - torch.Tensor (via .detach().cpu().numpy())
    - objects implementing __array__
    """
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy())
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return np.asarray(x.detach().cpu().numpy())
    return np.asarray(x)


def maybe_to_mlx(x: np.ndarray, dtype: str | None = None):
    """Convert NumPy -> MLX array if MLX is available; otherwise return NumPy."""
    try:
        import mlx.core as mx
    except Exception:
        return x

    if dtype is None:
        return mx.array(x)
    return mx.array(x, dtype=getattr(mx, dtype))


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)
