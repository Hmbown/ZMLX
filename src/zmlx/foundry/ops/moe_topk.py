from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import (
    MOE_K_LADDER,
    N_EXPERTS_LADDER,
    TOKENS_LADDER,
    KernelOp,
    maybe_to_float32,
    randn_np,
)


def softmax_1d(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class MoETopKOp(KernelOp):
    name = "moe_topk"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.MOE,
            summary="MoE routing: router logits -> top-k indices + weights",
            inputs=["logits[tokens,n_experts]"],
            outputs=["topk_idx[tokens,k]", "topk_w[tokens,k]"],
            op_params_schema={
                "k": {"type": "int", "enum": MOE_K_LADDER, "default": 2},
                "distribution": {"type": "str", "enum": ["uniform", "peaked", "ties"], "default": "uniform"},
                "temperature": {"type": "float", "default": 1.0},
            },
            shape_hints={"tokens": TOKENS_LADDER, "n_experts": N_EXPERTS_LADDER},
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        n_experts = int(rng.choice(N_EXPERTS_LADDER))
        return {"tokens": tokens, "n_experts": n_experts}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        k = int(rng.choice(MOE_K_LADDER))
        k = min(k, shape["n_experts"])
        distribution = str(rng.choice(["uniform", "peaked", "ties"]))
        temperature = float(rng.choice([0.7, 1.0, 1.5]))
        return {"k": k, "distribution": distribution, "temperature": temperature}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        dist = str(op_params.get("distribution", "uniform"))
        logits = randn_np((shape["tokens"], shape["n_experts"]), dtype=dtype, seed=seed, scale=1.0)
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        if dist == "peaked":
            peak_idx = rng.integers(0, shape["n_experts"], size=(shape["tokens"],))
            for t in range(shape["tokens"]):
                logits[t, peak_idx[t]] += 6.0
        elif dist == "ties":
            logits = np.round(logits * 2.0) / 2.0
            if shape["n_experts"] >= 4:
                logits[0, :4] = 1.0
        return {"logits": logits}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        logits = maybe_to_float32(inputs["logits"])
        k = int(op_params.get("k", 2))
        temperature = float(op_params.get("temperature", 1.0))
        if temperature <= 0:
            temperature = 1.0
        x = logits / temperature
        idx = np.argsort(-x, axis=-1)[:, :k]
        top = np.take_along_axis(x, idx, axis=-1)
        w = softmax_1d(top, axis=-1)
        return {"topk_idx": idx.astype(np.int32), "topk_w": w.astype(np.float32)}
