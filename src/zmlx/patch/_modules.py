"""Drop-in nn.Module replacements backed by ZMLX Metal kernels."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..kernels import norms, transformer


class ZMLXRMSNorm(nn.Module):
    """Drop-in replacement for nn.RMSNorm using a fused Metal kernel."""

    def __init__(
        self,
        dims: int,
        eps: float = 1e-6,
        threadgroup: int = 256,
        compute_dtype: Any = mx.float32,
    ):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.threadgroup = threadgroup
        self.compute_dtype = compute_dtype
        self.weight = mx.ones((dims,))

    def __call__(self, x: Any) -> Any:
        return norms.rmsnorm(
            x,
            self.weight,
            eps=self.eps,
            threadgroup=self.threadgroup,
            compute_dtype=self.compute_dtype,
        )

    def __repr__(self) -> str:
        return f"ZMLXRMSNorm(dims={self.dims}, eps={self.eps})"


class ZMLXLayerNorm(nn.Module):
    """Drop-in replacement for nn.LayerNorm using a fused Metal kernel."""

    def __init__(
        self,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        threadgroup: int = 256,
        compute_dtype: Any = mx.float32,
    ):
        super().__init__()
        self.dims = dims
        self.eps = eps
        self.affine = affine
        self.threadgroup = threadgroup
        self.compute_dtype = compute_dtype
        if affine:
            self.weight = mx.ones((dims,))
            self.bias = mx.zeros((dims,))

    def __call__(self, x: Any) -> Any:
        if self.affine:
            return norms.layernorm(
                x,
                self.weight,
                self.bias,
                eps=self.eps,
                threadgroup=self.threadgroup,
                compute_dtype=self.compute_dtype,
            )
        return norms.layer_norm_no_weight(
            x,
            eps=self.eps,
            threadgroup=self.threadgroup,
            compute_dtype=self.compute_dtype,
        )

    def __repr__(self) -> str:
        return f"ZMLXLayerNorm(dims={self.dims}, eps={self.eps}, affine={self.affine})"


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
        return transformer.swiglu(x, compute_dtype=self.compute_dtype)

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
        return transformer.geglu(x, compute_dtype=self.compute_dtype)

    def __repr__(self) -> str:
        return "ZMLXGeGLUActivation()"
