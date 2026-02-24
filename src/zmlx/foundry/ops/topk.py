from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import TOKENS_LADDER, KernelOp, maybe_to_float32, randn_np


class TopKOp(KernelOp):
    name = "topk"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.REDUCTION,
            summary="TopK along last axis",
            inputs=["x[tokens,dim]"],
            outputs=["values[tokens,k]", "indices[tokens,k]"],
            op_params_schema={"k": {"type": "int", "enum": [2, 4, 8], "default": 4}},
            shape_hints={"tokens": TOKENS_LADDER, "dim": [8, 16, 32, 64, 128, 256]},
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        dim = int(rng.choice([8, 16, 32, 64, 128]))
        return {"tokens": tokens, "dim": dim}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        k = int(rng.choice([2, 4, 8]))
        k = min(k, shape["dim"])
        return {"k": k}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        x = randn_np((shape["tokens"], shape["dim"]), dtype=dtype, seed=seed, scale=1.0)
        return {"x": x}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        x = maybe_to_float32(inputs["x"])
        k = int(op_params.get("k", 4))
        idx = np.argsort(-x, axis=-1)[:, :k]
        vals = np.take_along_axis(x, idx, axis=-1)
        return {"values": vals.astype(np.float32), "indices": idx.astype(np.int32)}
