from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import (
    HIDDEN_LADDER,
    TOKENS_LADDER,
    KernelOp,
    maybe_to_float32,
    randn_np,
)


class LayerNormOp(KernelOp):
    name = "layernorm"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.REDUCTION,
            summary="LayerNorm: (x-mean)/sqrt(var+eps)*gamma+beta",
            inputs=["x[tokens,hidden]", "gamma[hidden]", "beta[hidden]"],
            outputs=["y[tokens,hidden]"],
            op_params_schema={"eps": {"type": "float", "default": 1e-5}},
            shape_hints={"tokens": TOKENS_LADDER, "hidden": HIDDEN_LADDER},
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        hidden = int(rng.choice(HIDDEN_LADDER))
        return {"tokens": tokens, "hidden": hidden}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        eps = float(rng.choice([1e-5, 1e-6, 1e-4]))
        return {"eps": eps}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        x = randn_np((shape["tokens"], shape["hidden"]), dtype=dtype, seed=seed, scale=0.5)
        gamma = randn_np((shape["hidden"],), dtype=dtype, seed=seed + 1, scale=0.1)
        gamma = gamma + 1.0  # center around 1
        beta = randn_np((shape["hidden"],), dtype=dtype, seed=seed + 2, scale=0.1)
        return {"x": x, "gamma": gamma, "beta": beta}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        x = maybe_to_float32(inputs["x"])
        gamma = maybe_to_float32(inputs["gamma"])
        beta = maybe_to_float32(inputs["beta"])
        eps = float(op_params.get("eps", 1e-5))
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        y = (x - mean) / np.sqrt(var + eps)
        y = y * gamma + beta
        return {"y": y.astype(np.float32)}
