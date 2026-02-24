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

# ---------------------------------------------------------------------------
# Knob space constants (from datafoundry rmsnorm_knobs)
# ---------------------------------------------------------------------------

RMSNORM_TEMPLATES = ["t0_basic", "t1_tgmem"]
TG_SIZES = [32, 64, 128, 256]
VECS = [1, 2, 4]
UNROLLS = [1, 2, 4]


class RMSNormOp(KernelOp):
    name = "rmsnorm"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.REDUCTION,
            summary="RMSNorm: y = x * w / sqrt(mean(x^2)+eps)",
            inputs=["x[tokens,hidden]", "w[hidden]"],
            outputs=["y[tokens,hidden]"],
            op_params_schema={"eps": {"type": "float", "default": 1e-5}},
            shape_hints={"tokens": TOKENS_LADDER, "hidden": HIDDEN_LADDER},
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=RMSNORM_TEMPLATES,
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def templates(self) -> list[str]:
        return RMSNORM_TEMPLATES

    def knob_space(self, template_id: str) -> dict[str, Any]:
        space: dict[str, Any] = {
            "tg_size": {"type": "int", "values": TG_SIZES},
            "unroll": {"type": "int", "values": UNROLLS},
            "fast_math": {"type": "bool"},
        }
        if template_id == "t0_basic":
            space["vec"] = {"type": "int", "values": VECS}
        return space

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        hidden = int(rng.choice(HIDDEN_LADDER))
        return {"tokens": tokens, "hidden": hidden}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        eps = float(rng.choice([1e-5, 1e-6, 1e-4]))
        return {"eps": eps}

    def generate_inputs_numpy(
        self,
        shape: dict[str, int],
        dtype: str,
        op_params: dict[str, Any],
        seed: int,
    ) -> dict[str, np.ndarray]:
        x = randn_np((shape["tokens"], shape["hidden"]), dtype=dtype, seed=seed, scale=0.5)
        w = randn_np((shape["hidden"],), dtype=dtype, seed=seed + 1, scale=0.1)
        return {"x": x, "w": w}

    def reference_numpy(
        self,
        inputs: dict[str, np.ndarray],
        op_params: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        x = maybe_to_float32(inputs["x"])
        w = maybe_to_float32(inputs["w"])
        eps = float(op_params.get("eps", 1e-5))
        ms = np.mean(x * x, axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(ms + eps)
        y = x * inv * w
        return {"y": y.astype(np.float32)}

    def validate_knobs(
        self, template_id: str, knobs: dict[str, Any], shape: dict[str, int], dtype: str
    ) -> tuple[bool, str]:
        tg = int(knobs.get("tg_size", 0))
        if tg not in TG_SIZES:
            return False, "invalid_tg_size"
        if tg & (tg - 1) != 0:
            return False, "tg_size_not_pow2"
        unroll = int(knobs.get("unroll", 1))
        if unroll not in UNROLLS:
            return False, "invalid_unroll"
        vec = int(knobs.get("vec", 1))
        if template_id == "t0_basic" and vec not in VECS:
            return False, "invalid_vec"
        if template_id != "t0_basic" and vec != 1:
            return False, "vec_not_supported"
        return True, ""

    def bytes_and_flops(self, shape: dict[str, int], dtype: str) -> tuple[int, int]:
        tokens = int(shape.get("tokens", shape.get("batch", 1) * shape.get("seq", 1)))
        h = int(shape["hidden"])
        bytes_per_elem = 4 if dtype == "float32" else 2
        # x read + y write
        bytes_rw = tokens * h * bytes_per_elem * 2
        # weight read (amortized across tokens)
        bytes_w = h * bytes_per_elem
        # flops: square + sum (2), rsqrt (~1), mul (2) per element
        flops = tokens * h * 4
        return int(bytes_rw + bytes_w), int(flops)
