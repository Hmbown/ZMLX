from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import (
    MOE_K_LADDER,
    N_EXPERTS_LADDER,
    TOKENS_LADDER,
    KernelOp,
    randn_np,
)
from .moe_topk import softmax_1d


def pack_assignments(
    topk_idx: np.ndarray, topk_w: np.ndarray, n_experts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack token->expert assignments into expert-grouped layout.

    Returns (expert_counts, expert_offsets, packed_token_ids, packed_weights).
    """
    tokens, k = topk_idx.shape
    pairs = []
    for t in range(tokens):
        for j in range(k):
            e = int(topk_idx[t, j])
            pairs.append((e, t, float(topk_w[t, j])))
    pairs.sort(key=lambda x: x[0])

    expert_counts = np.zeros((n_experts,), dtype=np.int32)
    for e, _, _ in pairs:
        expert_counts[e] += 1

    expert_offsets = np.zeros((n_experts + 1,), dtype=np.int32)
    expert_offsets[1:] = np.cumsum(expert_counts)

    packed_token_ids = np.zeros((tokens * k,), dtype=np.int32)
    packed_weights = np.zeros((tokens * k,), dtype=np.float32)
    for i, (_, t, w) in enumerate(pairs):
        packed_token_ids[i] = t
        packed_weights[i] = w

    return expert_counts, expert_offsets, packed_token_ids, packed_weights


class MoEPackOp(KernelOp):
    name = "moe_pack"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.MOE,
            summary="MoE: build expert_counts/offsets + packed token ids/weights grouped by expert",
            inputs=["topk_idx[tokens,k]", "topk_w[tokens,k]"],
            outputs=[
                "expert_counts[n_experts]",
                "expert_offsets[n_experts+1]",
                "packed_token_ids[tokens*k]",
                "packed_weights[tokens*k]",
            ],
            op_params_schema={
                "n_experts": {"type": "int", "enum": N_EXPERTS_LADDER, "default": 16},
                "k": {"type": "int", "enum": MOE_K_LADDER, "default": 2},
                "distribution": {"type": "str", "enum": ["uniform", "peaked", "ties"], "default": "uniform"},
            },
            shape_hints={"tokens": TOKENS_LADDER, "n_experts": N_EXPERTS_LADDER, "k": MOE_K_LADDER},
            dtype_hints=["float16", "float32"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        n_experts = int(rng.choice(N_EXPERTS_LADDER))
        k = int(rng.choice(MOE_K_LADDER))
        k = min(k, n_experts)
        return {"tokens": tokens, "n_experts": n_experts, "k": k}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        return {
            "n_experts": int(shape["n_experts"]),
            "k": int(shape["k"]),
            "distribution": str(rng.choice(["uniform", "peaked", "ties"])),
        }

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
        idx = np.argsort(-logits, axis=-1)[:, : shape["k"]]
        top = np.take_along_axis(logits, idx, axis=-1)
        w = softmax_1d(top, axis=-1).astype(np.float32)
        return {"topk_idx": idx.astype(np.int32), "topk_w": w}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        topk_idx = np.asarray(inputs["topk_idx"]).astype(np.int32)
        topk_w = np.asarray(inputs["topk_w"]).astype(np.float32)
        n_experts = int(op_params.get("n_experts", int(topk_idx.max()) + 1))
        expert_counts, expert_offsets, packed_token_ids, packed_weights = pack_assignments(
            topk_idx, topk_w, n_experts
        )
        return {
            "expert_counts": expert_counts,
            "expert_offsets": expert_offsets,
            "packed_token_ids": packed_token_ids,
            "packed_weights": packed_weights,
        }
