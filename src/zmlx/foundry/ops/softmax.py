from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import TOKENS_LADDER, KernelOp, maybe_to_float32, randn_np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class SoftmaxOp(KernelOp):
    name = "softmax"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.ATTENTION,
            summary="Softmax (attention-style) along last axis",
            inputs=["x[tokens,dim]"],
            outputs=["y[tokens,dim]"],
            op_params_schema={"axis": {"type": "int", "default": -1}},
            shape_hints={"tokens": TOKENS_LADDER, "dim": [32, 64, 128, 256, 512, 1024, 2048]},
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        dim = int(rng.choice([32, 64, 128, 256, 512, 1024]))
        return {"tokens": tokens, "dim": dim}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        return {"axis": -1}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        x = randn_np((shape["tokens"], shape["dim"]), dtype=dtype, seed=seed, scale=1.0)
        return {"x": x}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        x = maybe_to_float32(inputs["x"])
        axis = int(op_params.get("axis", -1))
        y = _softmax(x, axis=axis)
        return {"y": y.astype(np.float32)}
