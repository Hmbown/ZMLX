from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import HIDDEN_LADDER, TOKENS_LADDER, KernelOp, randint_np


def _unpack_int4(packed: np.ndarray) -> np.ndarray:
    """Unpack uint8 packed int4 (two's complement) into int8 in [-8,7]."""
    p = packed.astype(np.uint8)
    low = np.bitwise_and(p, 0xF).astype(np.int8)
    high = np.bitwise_and(p >> 4, 0xF).astype(np.int8)
    # two's complement sign extend 4-bit
    low = low - (low >= 8).astype(np.int8) * 16
    high = high - (high >= 8).astype(np.int8) * 16
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.int8)
    out[..., 0::2] = low
    out[..., 1::2] = high
    return out


class DequantizeOp(KernelOp):
    name = "dequantize"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.QUANT,
            summary="Dequantize int8 or packed int4 -> fp16/bf16",
            inputs=["q[tokens,hidden] (int8) OR q_packed[tokens,hidden/2] (int4 packed)"],
            outputs=["y[tokens,hidden]"],
            op_params_schema={
                "q_dtype": {"type": "str", "enum": ["int8", "int4"], "default": "int8"},
                "scale": {"type": "float", "default": 0.02},
                "out_dtype": {"type": "str", "enum": ["float16", "bfloat16", "float32"], "default": "float16"},
            },
            shape_hints={"tokens": TOKENS_LADDER, "hidden": HIDDEN_LADDER},
            dtype_hints=["int8", "int4"],
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
        out_dtype = str(rng.choice(["float16", "bfloat16", "float32"]))
        return {"q_dtype": q_dtype, "scale": scale, "out_dtype": out_dtype}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        q_dtype = str(op_params.get("q_dtype", "int8"))
        if q_dtype == "int8":
            q = randint_np(-128, 128, (shape["tokens"], shape["hidden"]), dtype="int8", seed=seed)
            return {"q": q}
        if q_dtype == "int4":
            packed = randint_np(
                0, 256, (shape["tokens"], shape["hidden"] // 2), dtype="int32", seed=seed
            ).astype(np.uint8)
            return {"q_packed": packed}
        raise ValueError(f"Unknown q_dtype: {q_dtype}")

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        q_dtype = str(op_params.get("q_dtype", "int8"))
        scale = float(op_params.get("scale", 0.02))
        if scale <= 0:
            scale = 0.02

        if q_dtype == "int8":
            q = np.asarray(inputs["q"]).astype(np.int8)
            y = q.astype(np.float32) * scale
            return {"y": y}
        if q_dtype == "int4":
            packed = np.asarray(inputs["q_packed"]).astype(np.uint8)
            q = _unpack_int4(packed).astype(np.int8)
            y = q.astype(np.float32) * scale
            return {"y": y}
        raise ValueError(f"Unknown q_dtype: {q_dtype}")
