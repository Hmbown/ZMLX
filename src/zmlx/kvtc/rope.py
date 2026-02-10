from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "KVTC requires numpy. Install with: pip install zmlx[kvtc]"
    ) from e


@dataclass(frozen=True)
class RotaryConfig:
    """Configuration for rotary positional embeddings (RoPE).

    Args:
        dim: Rotary dimension (must be even).
        base: Base frequency for inv_freq computation.
        traditional: If False (default), use interleaved even/odd pairing
            (standard Llama-style). If True, use half-split pairing where
            the first half and second half of the rotary dims form pairs.
        offset: Starting index within head_dim where RoPE is applied.
            Dims [0, offset) are passed through unchanged, RoPE applies to
            [offset, offset+dim), and [offset+dim, head_dim) passes through.
            This handles GLM's ``[kv_latent(512) | k_pe(64)]`` layout where
            RoPE applies only to the last 64 dims (offset=512).
    """

    dim: int
    base: float = 10000.0
    traditional: bool = False
    offset: int = 0


class RotaryEmbedding:
    """Compute RoPE cos/sin tables and apply / invert rotary embedding.

    NumPy-only implementation that supports both interleaved and traditional
    (half-split) RoPE layouts, plus an optional offset within head_dim.
    """

    def __init__(self, cfg: RotaryConfig):
        if cfg.dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {cfg.dim}")
        self.cfg = cfg
        inv_freq = 1.0 / (cfg.base ** (np.arange(0, cfg.dim, 2, dtype=np.float32) / cfg.dim))
        self.inv_freq = inv_freq  # (dim/2,)

    def cos_sin(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return cos, sin with shape (len(positions), dim/2)."""
        pos = positions.astype(np.float32).reshape(-1)
        freqs = np.outer(pos, self.inv_freq)  # (n, dim/2)
        return np.cos(freqs), np.sin(freqs)

    def apply(self, x: np.ndarray, positions: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Apply RoPE (or its inverse) to the last dimension of x.

        x: (..., head_dim)
        positions: 1D token positions.

        RoPE applies to ``x[..., offset:offset+dim]``.  Dims before offset
        and after ``offset+dim`` pass through unchanged.
        """
        head_dim = x.shape[-1]
        rot_dim = self.cfg.dim
        offset = self.cfg.offset

        if offset + rot_dim > head_dim:
            raise ValueError(
                f"offset({offset}) + rotary dim({rot_dim}) = {offset + rot_dim} "
                f"exceeds head_dim({head_dim})"
            )

        # Split into pre-offset passthrough, rotary region, post-rotary passthrough
        x_pre = x[..., :offset]
        x_rot = x[..., offset : offset + rot_dim]
        x_post = x[..., offset + rot_dim :]

        pos = positions.astype(np.int64).reshape(-1)
        cos, sin = self.cos_sin(pos)  # (n, rot_dim/2)
        n = cos.shape[0]

        # Find token axis in x_rot (a dim with length == n, preferring second-to-last)
        token_axis = None
        for ax in range(x_rot.ndim - 1):
            if x_rot.shape[ax] == n:
                token_axis = ax
        if token_axis is None:
            raise ValueError(
                f"Could not find a token axis of length {n} in x_rot shape {x_rot.shape}. "
                "Pass x with an explicit token dimension matching positions."
            )

        # Move token axis to -2 for broadcasting: (..., n, rot_dim)
        x_moved = np.moveaxis(x_rot, token_axis, -2)

        cos_b = cos.reshape((1,) * (x_moved.ndim - 2) + (n, rot_dim // 2))
        sin_b = sin.reshape((1,) * (x_moved.ndim - 2) + (n, rot_dim // 2))
        if inverse:
            sin_b = -sin_b

        half = rot_dim // 2
        if self.cfg.traditional:
            # Half-split: first half pairs with second half
            x1 = x_moved[..., :half]
            x2 = x_moved[..., half:]
        else:
            # Interleaved: even indices pair with odd indices
            x1 = x_moved[..., ::2]
            x2 = x_moved[..., 1::2]

        y1 = x1 * cos_b - x2 * sin_b
        y2 = x1 * sin_b + x2 * cos_b

        y = np.empty_like(x_moved)
        if self.cfg.traditional:
            y[..., :half] = y1
            y[..., half:] = y2
        else:
            y[..., ::2] = y1
            y[..., 1::2] = y2

        # Move token axis back
        y = np.moveaxis(y, -2, token_axis)

        # Reassemble with passthrough regions
        parts = []
        if x_pre.size > 0:
            parts.append(x_pre)
        parts.append(y)
        if x_post.size > 0:
            parts.append(x_post)

        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts, axis=-1)
