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
# Knob space constants (from datafoundry swiglu_knobs)
# ---------------------------------------------------------------------------

SWIGLU_TEMPLATES = ["t0_basic", "t1_unrolled"]
TG_SIZES = [32, 64, 128, 256]
VECS = [1, 2, 4]
UNROLLS = [1, 2, 4]


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class SwiGLUOp(KernelOp):
    name = "swiglu"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.FUSED_POINTWISE,
            summary="SwiGLU gate: y = silu(a) * b where x=[a|b]",
            inputs=["x[tokens,2*hidden]"],
            outputs=["y[tokens,hidden]"],
            op_params_schema={},
            shape_hints={"tokens": TOKENS_LADDER, "hidden": HIDDEN_LADDER},
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=SWIGLU_TEMPLATES,
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def templates(self) -> list[str]:
        return SWIGLU_TEMPLATES

    def knob_space(self, template_id: str) -> dict[str, Any]:
        space: dict[str, Any] = {
            "tg_size": {"type": "int", "values": TG_SIZES},
            "unroll": {"type": "int", "values": UNROLLS},
            "fast_math": {"type": "bool"},
        }
        if template_id == "t1_unrolled":
            space["vec"] = {"type": "int", "values": VECS}
        return space

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        hidden = int(rng.choice(HIDDEN_LADDER))
        return {"tokens": tokens, "hidden": hidden}

    def generate_inputs_numpy(
        self,
        shape: dict[str, int],
        dtype: str,
        op_params: dict[str, Any],
        seed: int,
    ) -> dict[str, np.ndarray]:
        x = randn_np((shape["tokens"], shape["hidden"] * 2), dtype=dtype, seed=seed, scale=0.5)
        return {"x": x}

    def reference_numpy(
        self,
        inputs: dict[str, np.ndarray],
        op_params: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        x = maybe_to_float32(inputs["x"])
        h2 = x.shape[-1]
        h = h2 // 2
        x0 = x[..., :h]
        x1 = x[..., h:2 * h]
        sig = 1.0 / (1.0 + np.exp(-x0))
        y = (x0 * sig) * x1
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
        if template_id == "t1_unrolled":
            if vec not in VECS:
                return False, "invalid_vec"
        else:
            if vec != 1:
                return False, "vec_not_supported"
        return True, ""

    def bytes_and_flops(self, shape: dict[str, int], dtype: str) -> tuple[int, int]:
        tokens = int(shape.get("tokens", shape.get("batch", 1) * shape.get("seq", 1)))
        h = int(shape["hidden"])
        bytes_per_elem = 4 if dtype == "float32" else 2
        # reads 2H inputs, writes H output
        bytes_rw = tokens * (2 * h * bytes_per_elem + h * bytes_per_elem)
        # flops per output elem: sigmoid approx + muls
        flops = tokens * h * 12
        return int(bytes_rw), int(flops)
