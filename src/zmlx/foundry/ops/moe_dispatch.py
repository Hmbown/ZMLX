from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import (
    HIDDEN_LADDER,
    MOE_K_LADDER,
    N_EXPERTS_LADDER,
    TOKENS_LADDER,
    KernelOp,
    randn_np,
)
from .moe_pack import pack_assignments
from .moe_topk import softmax_1d


class MoEDispatchOp(KernelOp):
    name = "moe_dispatch"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.MOE,
            summary="MoE dispatch gather: build expert_x by gathering x using packed_token_ids",
            inputs=["x[tokens,hidden]", "packed_token_ids[tokens*k]", "expert_offsets[n_experts+1]"],
            outputs=["expert_x[tokens*k,hidden]"],
            op_params_schema={
                "n_experts": {"type": "int", "enum": N_EXPERTS_LADDER, "default": 16},
                "k": {"type": "int", "enum": MOE_K_LADDER, "default": 2},
                "distribution": {"type": "str", "enum": ["uniform", "peaked", "ties"], "default": "uniform"},
            },
            shape_hints={
                "tokens": TOKENS_LADDER,
                "hidden": HIDDEN_LADDER,
                "n_experts": N_EXPERTS_LADDER,
                "k": MOE_K_LADDER,
            },
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        hidden = int(rng.choice(HIDDEN_LADDER))
        n_experts = int(rng.choice(N_EXPERTS_LADDER))
        k = int(rng.choice(MOE_K_LADDER))
        k = min(k, n_experts)
        return {"tokens": tokens, "hidden": hidden, "n_experts": n_experts, "k": k}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        return {
            "n_experts": int(shape["n_experts"]),
            "k": int(shape["k"]),
            "distribution": str(rng.choice(["uniform", "peaked", "ties"])),
        }

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        x = randn_np((shape["tokens"], shape["hidden"]), dtype=dtype, seed=seed, scale=0.5)

        # Build packed assignments from synthetic router logits
        dist = str(op_params.get("distribution", "uniform"))
        logits = randn_np(
            (shape["tokens"], shape["n_experts"]), dtype=dtype, seed=seed + 1, scale=1.0
        )
        rng = np.random.default_rng(int(seed + 1) & 0xFFFFFFFF)
        if dist == "peaked":
            peak_idx = rng.integers(0, shape["n_experts"], size=(shape["tokens"],))
            for t in range(shape["tokens"]):
                logits[t, peak_idx[t]] += 6.0
        elif dist == "ties":
            logits = np.round(logits * 2.0) / 2.0
            if shape["n_experts"] >= 4:
                logits[0, :4] = 1.0

        topk_idx = np.argsort(-logits, axis=-1)[:, : shape["k"]].astype(np.int32)
        top = np.take_along_axis(logits, topk_idx, axis=-1)
        topk_w = softmax_1d(top, axis=-1).astype(np.float32)

        _, expert_offsets, packed_token_ids, _ = pack_assignments(
            topk_idx, topk_w, shape["n_experts"]
        )
        return {
            "x": x,
            "packed_token_ids": packed_token_ids,
            "expert_offsets": expert_offsets,
        }

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        x = np.asarray(inputs["x"])
        ids = np.asarray(inputs["packed_token_ids"]).astype(np.int64)
        expert_x = x[ids, :]
        return {"expert_x": expert_x}
