from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import HIDDEN_LADDER, TOKENS_LADDER, KernelOp, randint_np, randn_np


class GatherOp(KernelOp):
    name = "gather"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.DATA_MOVEMENT,
            summary="Gather (index select) along axis=0",
            inputs=["x[tokens,hidden]", "idx[m]"],
            outputs=["y[m,hidden]"],
            op_params_schema={"axis": {"type": "int", "default": 0}},
            shape_hints={
                "tokens": TOKENS_LADDER,
                "hidden": HIDDEN_LADDER,
                "m": [16, 32, 64, 128, 256, 512],
            },
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        hidden = int(rng.choice(HIDDEN_LADDER))
        m = int(rng.choice([16, 32, 64, 128, 256]))
        m = min(m, tokens) if tokens > 0 else m
        return {"tokens": tokens, "hidden": hidden, "m": m}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        x = randn_np((shape["tokens"], shape["hidden"]), dtype=dtype, seed=seed, scale=0.5)
        idx = randint_np(0, max(1, shape["tokens"]), (shape["m"],), dtype="int32", seed=seed + 1)
        return {"x": x, "idx": idx}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        x = np.asarray(inputs["x"])
        idx = np.asarray(inputs["idx"]).astype(np.int64)
        y = x[idx, :]
        return {"y": y}
