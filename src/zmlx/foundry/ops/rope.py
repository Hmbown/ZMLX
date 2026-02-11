from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import TOKENS_LADDER, KernelOp, maybe_to_float32, randn_np


def _rope_apply(x: np.ndarray, theta: float, offset: int = 0) -> np.ndarray:
    tokens, n_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2

    pos = np.arange(tokens, dtype=np.float32) + float(offset)
    inv_freq = theta ** (-np.arange(0, half, dtype=np.float32) / float(half))
    angles = np.outer(pos, inv_freq)
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]

    x1 = x[..., :half]
    x2 = x[..., half:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return np.concatenate([y1, y2], axis=-1)


class RoPEOp(KernelOp):
    name = "rope"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.ATTENTION,
            summary="Apply RoPE to Q/K tensors (rotary embedding)",
            inputs=["x[tokens,n_heads,head_dim]"],
            outputs=["y[tokens,n_heads,head_dim]"],
            op_params_schema={
                "theta": {"type": "float", "default": 10000.0},
                "offset": {"type": "int", "default": 0},
            },
            shape_hints={"tokens": TOKENS_LADDER, "head_dim": [64, 96, 128], "n_heads": [8, 16, 32]},
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        n_heads = int(rng.choice([8, 16, 32]))
        head_dim = int(rng.choice([64, 96, 128]))
        head_dim = head_dim + (head_dim % 2)
        return {"tokens": tokens, "n_heads": n_heads, "head_dim": head_dim}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        theta = float(rng.choice([10000.0, 50000.0, 1000.0]))
        offset = int(rng.integers(0, 2048))
        return {"theta": theta, "offset": offset}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        x = randn_np(
            (shape["tokens"], shape["n_heads"], shape["head_dim"]),
            dtype=dtype, seed=seed, scale=0.5,
        )
        return {"x": x}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        x = maybe_to_float32(inputs["x"])
        theta = float(op_params.get("theta", 10000.0))
        offset = int(op_params.get("offset", 0))
        y = _rope_apply(x, theta=theta, offset=offset)
        return {"y": y.astype(np.float32)}
