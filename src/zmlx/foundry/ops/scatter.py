from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import HIDDEN_LADDER, TOKENS_LADDER, KernelOp, randint_np, randn_np


class ScatterOp(KernelOp):
    name = "scatter"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.DATA_MOVEMENT,
            summary="Scatter-add along axis=0: y[idx] += updates",
            inputs=["updates[m,hidden]", "idx[m]", "base[tokens,hidden]"],
            outputs=["y[tokens,hidden]"],
            op_params_schema={"mode": {"type": "str", "enum": ["add"], "default": "add"}},
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
        return {"tokens": tokens, "hidden": hidden, "m": m}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        updates = randn_np((shape["m"], shape["hidden"]), dtype=dtype, seed=seed, scale=0.5)
        idx = randint_np(0, max(1, shape["tokens"]), (shape["m"],), dtype="int32", seed=seed + 1)
        base = randn_np((shape["tokens"], shape["hidden"]), dtype=dtype, seed=seed + 2, scale=0.1)
        return {"updates": updates, "idx": idx, "base": base}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        updates = np.asarray(inputs["updates"])
        idx = np.asarray(inputs["idx"]).astype(np.int64)
        base = np.asarray(inputs["base"]).copy()
        np.add.at(base, idx, updates)
        return {"y": base}
