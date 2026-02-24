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

MOE_COMBINE_TEMPLATES = ["t0_basic", "t1_k8_unrolled", "t2_row_tile"]
TG_SIZES = [32, 64, 128, 256]
UNROLLS = [1, 2, 4, 8]


class MoECombineOp(KernelOp):
    name = "moe_combine"
    extra_shape_dims = {"n_experts": N_EXPERTS_LADDER, "k": MOE_K_LADDER}

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.MOE,
            summary="MoE combine scatter: accumulate expert_y back to tokens (weighted)",
            inputs=[
                "expert_y[tokens*k,hidden]",
                "packed_token_ids[tokens*k]",
                "packed_weights[tokens*k]",
            ],
            outputs=["y[tokens,hidden]"],
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
            templates=MOE_COMBINE_TEMPLATES,
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def templates(self) -> list[str]:
        return MOE_COMBINE_TEMPLATES

    def knob_space(self, template_id: str) -> dict[str, Any]:
        _ = template_id
        return {
            "tg_size": {"type": "int", "values": TG_SIZES},
            "unroll": {"type": "int", "values": UNROLLS},
            "fast_math": {"type": "bool"},
        }

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

        # Router -> topk -> pack
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

        _, _, packed_token_ids, packed_weights = pack_assignments(
            topk_idx, topk_w, shape["n_experts"]
        )

        # expert_y = x[packed_token_ids] for a strong roundtrip test
        expert_y = x[packed_token_ids.astype(np.int64), :]

        return {
            "expert_y": expert_y,
            "packed_token_ids": packed_token_ids,
            "packed_weights": packed_weights,
        }

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        expert_y = np.asarray(inputs["expert_y"]).astype(np.float32)
        ids = np.asarray(inputs["packed_token_ids"]).astype(np.int64)
        w = np.asarray(inputs["packed_weights"]).astype(np.float32)
        tokens = int(ids.max()) + 1 if ids.size else 0
        hidden = expert_y.shape[-1]
        y = np.zeros((tokens, hidden), dtype=np.float32)
        np.add.at(y, ids, expert_y * w[:, None])
        return {"y": y}

    def validate_knobs(
        self, template_id: str, knobs: dict[str, Any], shape: dict[str, int], dtype: str
    ) -> tuple[bool, str]:
        _ = template_id, shape, dtype
        tg = int(knobs.get("tg_size", 0))
        if tg not in TG_SIZES:
            return False, "invalid_tg_size"
        if tg & (tg - 1) != 0:
            return False, "tg_size_not_pow2"
        unroll = int(knobs.get("unroll", 1))
        if unroll not in UNROLLS:
            return False, "invalid_unroll"
        return True, ""

    def bytes_and_flops(self, shape: dict[str, int], dtype: str) -> tuple[int, int]:
        tokens = int(shape.get("tokens", shape.get("batch", 1) * shape.get("seq", 1)))
        hidden = int(shape["hidden"])
        k = int(shape.get("k", 2))
        pairs = tokens * k
        bytes_per_elem = 4 if dtype == "float32" else 2

        bytes_expert_y = pairs * hidden * bytes_per_elem
        bytes_ids = pairs * 4
        bytes_weights = pairs * 4
        bytes_y = tokens * hidden * bytes_per_elem

        # One multiply and one add per (assignment, hidden) contribution.
        flops = pairs * hidden * 2
        return int(bytes_expert_y + bytes_ids + bytes_weights + bytes_y), int(flops)
