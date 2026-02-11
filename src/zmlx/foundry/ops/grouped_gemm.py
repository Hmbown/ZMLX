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


class GroupedGemmOp(KernelOp):
    """Grouped GEMM hook for MoE.

    Not a Metal kernel yet. Provides a stable interface that can be swapped
    for a future specialized grouped/batched matmul kernel.
    """

    name = "grouped_gemm"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.GEMM,
            summary="Grouped GEMM hook for MoE (reference uses NumPy matmul per expert)",
            inputs=[
                "expert_x[tokens*k,hidden]",
                "expert_offsets[n_experts+1]",
                "(implicit) W_expert[n_experts,hidden,out_hidden] (generated from seed)",
            ],
            outputs=["expert_y[tokens*k,out_hidden]"],
            op_params_schema={
                "n_experts": {"type": "int", "enum": N_EXPERTS_LADDER, "default": 16},
                "k": {"type": "int", "enum": MOE_K_LADDER, "default": 2},
                "out_hidden": {"type": "int", "default": 512},
            },
            shape_hints={
                "tokens": TOKENS_LADDER,
                "hidden": HIDDEN_LADDER,
                "out_hidden": [512, 1024, 2048],
                "n_experts": N_EXPERTS_LADDER,
                "k": MOE_K_LADDER,
            },
            dtype_hints=["float16", "bfloat16", "float32"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice([16, 32, 64, 128, 256]))
        hidden = int(rng.choice([512, 768, 1024]))
        n_experts = int(rng.choice(N_EXPERTS_LADDER))
        k = int(rng.choice(MOE_K_LADDER))
        out_hidden = int(rng.choice([512, 1024]))
        return {
            "tokens": tokens,
            "hidden": hidden,
            "n_experts": n_experts,
            "k": min(k, n_experts),
            "out_hidden": out_hidden,
        }

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        return {
            "n_experts": int(shape["n_experts"]),
            "k": int(shape["k"]),
            "out_hidden": int(shape["out_hidden"]),
        }

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        # Build synthetic expert_x + expert_offsets from router assignments
        x = randn_np((shape["tokens"], shape["hidden"]), dtype=dtype, seed=seed, scale=0.5)
        logits = randn_np(
            (shape["tokens"], shape["n_experts"]), dtype=dtype, seed=seed + 1, scale=1.0
        )
        topk_idx = np.argsort(-logits, axis=-1)[:, : shape["k"]].astype(np.int32)
        top = np.take_along_axis(logits, topk_idx, axis=-1)
        topk_w = softmax_1d(top, axis=-1).astype(np.float32)
        _, expert_offsets, packed_token_ids, _ = pack_assignments(
            topk_idx, topk_w, shape["n_experts"]
        )
        expert_x = x[packed_token_ids.astype(np.int64), :]
        return {"expert_x": expert_x, "expert_offsets": expert_offsets}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        expert_x = np.asarray(inputs["expert_x"]).astype(np.float32)
        offsets = np.asarray(inputs["expert_offsets"]).astype(np.int32)
        n_experts = int(op_params.get("n_experts", len(offsets) - 1))
        out_hidden = int(op_params.get("out_hidden", expert_x.shape[-1]))

        # Per-expert weight matrices from a fixed seed (hook; real system provides W_expert)
        W = []
        base_rng = np.random.default_rng(12345)
        for _ in range(n_experts):
            W.append(
                base_rng.standard_normal(size=(expert_x.shape[-1], out_hidden)).astype(np.float32)
                * 0.02
            )

        y = np.zeros((expert_x.shape[0], out_hidden), dtype=np.float32)
        for e in range(n_experts):
            s = int(offsets[e])
            t = int(offsets[e + 1])
            if t <= s:
                continue
            y[s:t, :] = expert_x[s:t, :] @ W[e]
        return {"expert_y": y}
