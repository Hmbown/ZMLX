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


def _pack_int4(vals: np.ndarray) -> np.ndarray:
    """Pack signed int4 values (two's complement nibbles) into uint8."""
    v = vals.astype(np.int8)
    if v.shape[-1] % 2 != 0:
        raise ValueError("int4 pack requires even last dimension")
    low = np.bitwise_and(v[..., 0::2], 0xF).astype(np.uint8)
    high = np.bitwise_and(v[..., 1::2], 0xF).astype(np.uint8)
    return low | (high << 4)


class QuantizeOp(KernelOp):
    name = "quantize"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.QUANT,
            summary="Quantize fp16/bf16 -> int8 or packed int4 (symmetric)",
            inputs=["x[tokens,hidden]"],
            outputs=["q[tokens,hidden] (int8) OR q_packed[tokens,hidden/2] (int4 packed)"],
            op_params_schema={
                "q_dtype": {"type": "str", "enum": ["int8", "int4"], "default": "int8"},
                "scale": {"type": "float", "default": 0.02},
            },
            shape_hints={"tokens": TOKENS_LADDER, "hidden": HIDDEN_LADDER},
            dtype_hints=["float16", "bfloat16"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        hidden = int(rng.choice(HIDDEN_LADDER))
        if hidden % 2 != 0:
            hidden += 1
        return {"tokens": tokens, "hidden": hidden}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        q_dtype = str(rng.choice(["int8", "int4"]))
        scale = float(rng.choice([0.01, 0.02, 0.05]))
        return {"q_dtype": q_dtype, "scale": scale}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        x = randn_np((shape["tokens"], shape["hidden"]), dtype=dtype, seed=seed, scale=0.5)
        return {"x": x}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        x = maybe_to_float32(inputs["x"])
        q_dtype = str(op_params.get("q_dtype", "int8"))
        scale = float(op_params.get("scale", 0.02))
        if scale <= 0:
            scale = 0.02

        if q_dtype == "int8":
            q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
            return {"q": q}
        if q_dtype == "int4":
            q4 = np.clip(np.round(x / scale), -8, 7).astype(np.int8)
            q_packed = _pack_int4(q4)
            return {"q_packed": q_packed}
        raise ValueError(f"Unknown q_dtype: {q_dtype}")
