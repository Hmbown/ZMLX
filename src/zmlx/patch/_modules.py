"""Drop-in nn.Module replacements backed by ZMLX Metal kernels."""

from __future__ import annotations

import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..kernels import norms, transformer

_TG_CANDIDATES = (32, 64, 128, 256, 512, 1024)


def _autotune_threadgroup(fn: Any, x: Any, warmup: int = 2, iters: int = 5) -> int:
    """Try threadgroup sizes and return the fastest one."""
    sync = getattr(mx, "synchronize", None)
    best_tg = 256
    best_time = float("inf")
    for tg in _TG_CANDIDATES:
        # Warmup
        for _ in range(warmup):
            mx.eval(fn(x, tg))
        if callable(sync):
            sync()
        # Time
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn(x, tg))
        if callable(sync):
            sync()
        elapsed = time.perf_counter() - t0
        if elapsed < best_time:
            best_time = elapsed
            best_tg = tg
    return best_tg


class ZMLXRMSNorm(nn.Module):
    """Drop-in replacement for nn.RMSNorm using a fused Metal kernel."""

    def __init__(
        self,
        dims: int,
        eps: float = 1e-6,
        threadgroup: int | str = 256,
        compute_dtype: Any = mx.float32,
    ):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self._auto = threadgroup == "auto"
        self._resolved_tg: int | None = None if self._auto else int(threadgroup)
        self.compute_dtype = compute_dtype
        self.weight = mx.ones((dims,))

    @property
    def threadgroup(self) -> int:
        return self._resolved_tg or 256

    def __call__(self, x: Any) -> Any:
        if self._auto and self._resolved_tg is None:
            self._resolved_tg = _autotune_threadgroup(
                lambda x, tg: norms.rmsnorm(x, self.weight, eps=self.eps, threadgroup=tg),
                x,
            )
        return norms.rmsnorm(
            x,
            self.weight,
            eps=self.eps,
            threadgroup=self._resolved_tg or 256,
        )

    def __repr__(self) -> str:
        tg = f"auto->{self._resolved_tg}" if self._auto else str(self._resolved_tg)
        return f"ZMLXRMSNorm(dims={self.dims}, eps={self.eps}, tg={tg})"


class ZMLXLayerNorm(nn.Module):
    """Drop-in replacement for nn.LayerNorm using a fused Metal kernel."""

    def __init__(
        self,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        threadgroup: int | str = 256,
        compute_dtype: Any = mx.float32,
    ):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.affine = affine
        self._auto = threadgroup == "auto"
        self._resolved_tg: int | None = None if self._auto else int(threadgroup)
        self.compute_dtype = compute_dtype
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    @property
    def threadgroup(self) -> int:
        return self._resolved_tg or 256

    def __call__(self, x: Any) -> Any:
        tg = self._resolved_tg or 256
        if self._auto and self._resolved_tg is None:
            if self.affine:
                tg = _autotune_threadgroup(
                    lambda x, tg: norms.layernorm(
                        x, self.weight, self.bias, eps=self.eps, threadgroup=tg
                    ),
                    x,
                )
            else:
                tg = _autotune_threadgroup(
                    lambda x, tg: norms.layer_norm_no_weight(
                        x, eps=self.eps, threadgroup=tg
                    ),
                    x,
                )
            self._resolved_tg = tg
        if self.affine:
            return norms.layernorm(
                x,
                self.weight,
                self.bias,
                eps=self.eps,
                threadgroup=tg,
            )
        return norms.layer_norm_no_weight(
            x,
            eps=self.eps,
            threadgroup=tg,
        )

    def __repr__(self) -> str:
        tg = f"auto->{self._resolved_tg}" if self._auto else str(self._resolved_tg)
        return f"ZMLXLayerNorm(dims={self.dims}, eps={self.eps}, affine={self.affine}, tg={tg})"


class ZMLXSwiGLUActivation(nn.Module):
    """Replaces the activation portion of a SwiGLU MLP.

    Expects input of shape (..., 2*D) where the first half is the gate
    and the second half is the up projection.  Returns (..., D).
    """

    def __init__(
        self,
        compute_dtype: Any = mx.float32,
    ):
        super().__init__()
        self.compute_dtype = compute_dtype

    def __call__(self, x: Any) -> Any:
        return transformer.swiglu(x)

    def __repr__(self) -> str:
        return "ZMLXSwiGLUActivation()"


class ZMLXGeGLUActivation(nn.Module):
    """Replaces the activation portion of a GeGLU MLP."""

    def __init__(
        self,
        compute_dtype: Any = mx.float32,
    ):
        super().__init__()
        self.compute_dtype = compute_dtype

    def __call__(self, x: Any) -> Any:
        return transformer.geglu(x)

    def __repr__(self) -> str:
        return "ZMLXGeGLUActivation()"
