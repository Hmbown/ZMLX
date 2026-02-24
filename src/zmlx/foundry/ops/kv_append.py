from __future__ import annotations

from typing import Any

import numpy as np

from ..taxonomy import KernelClass, OpSpec
from .base import TOKENS_LADDER, KernelOp, randn_np


class KVAppendOp(KernelOp):
    name = "kv_append"

    def spec(self) -> OpSpec:
        return OpSpec(
            name=self.name,
            kernel_class=KernelClass.KV_CACHE,
            summary="KV cache append: write k_new/v_new into cache at [start_pos:start_pos+tokens]",
            inputs=[
                "k_cache[max_seq,n_kv_heads,head_dim]",
                "v_cache[max_seq,n_kv_heads,head_dim]",
                "k_new[tokens,n_kv_heads,head_dim]",
                "v_new[tokens,n_kv_heads,head_dim]",
            ],
            outputs=["k_cache_out", "v_cache_out"],
            op_params_schema={
                "max_seq": {"type": "int", "default": 2048},
                "start_pos": {"type": "int", "default": 0},
            },
            shape_hints={"tokens": TOKENS_LADDER, "n_kv_heads": [4, 8, 16], "head_dim": [64, 128]},
            dtype_hints=["float16", "bfloat16"],
            templates=["ref"],
        )

    def supported_dtypes(self) -> list[str]:
        return ["float16", "bfloat16", "float32"]

    def sample_shape(self, rng: np.random.Generator) -> dict[str, int]:
        tokens = int(rng.choice(TOKENS_LADDER))
        n_kv_heads = int(rng.choice([4, 8, 16]))
        head_dim = int(rng.choice([64, 128]))
        max_seq = int(rng.choice([512, 1024, 2048, 4096]))
        if tokens > max_seq:
            max_seq = tokens
        return {"tokens": tokens, "n_kv_heads": n_kv_heads, "head_dim": head_dim, "max_seq": max_seq}

    def sample_op_params(self, shape: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
        max_seq = int(shape["max_seq"])
        tokens = int(shape["tokens"])
        start_pos = int(rng.integers(0, max(1, max_seq - tokens + 1)))
        return {"max_seq": max_seq, "start_pos": start_pos}

    def generate_inputs_numpy(
        self, shape: dict[str, int], dtype: str, op_params: dict[str, Any], seed: int
    ) -> dict[str, np.ndarray]:
        max_seq = int(op_params.get("max_seq", shape["max_seq"]))
        dims = (shape["n_kv_heads"], shape["head_dim"])
        k_cache = randn_np((max_seq, *dims), dtype=dtype, seed=seed, scale=0.1)
        v_cache = randn_np((max_seq, *dims), dtype=dtype, seed=seed + 1, scale=0.1)
        k_new = randn_np((shape["tokens"], *dims), dtype=dtype, seed=seed + 2, scale=0.5)
        v_new = randn_np((shape["tokens"], *dims), dtype=dtype, seed=seed + 3, scale=0.5)
        return {"k_cache": k_cache, "v_cache": v_cache, "k_new": k_new, "v_new": v_new}

    def reference_numpy(
        self, inputs: dict[str, np.ndarray], op_params: dict[str, Any]
    ) -> dict[str, np.ndarray]:
        k_cache = np.asarray(inputs["k_cache"]).copy()
        v_cache = np.asarray(inputs["v_cache"]).copy()
        k_new = np.asarray(inputs["k_new"])
        v_new = np.asarray(inputs["v_new"])
        start_pos = int(op_params.get("start_pos", 0))
        end = start_pos + k_new.shape[0]
        k_cache[start_pos:end, :, :] = k_new
        v_cache[start_pos:end, :, :] = v_new
        return {"k_cache_out": k_cache, "v_cache_out": v_cache}
