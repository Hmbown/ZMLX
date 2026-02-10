"""GLM-4.7-Flash SKV integration (opt-in, experimental).

Pattern name: ``skv_mla``

This pattern stores MLA latent KV in compressed SKV form and keeps RoPE
uncompressed. It implements:
  - Strategy A fallback: materialize dense keys/values and run SDPA
  - Strategy B decode path: score in compressed rank-space + RoPE score add

RoPE slice is intentionally left uncompressed.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...kvtc.skv_mla import (
    SKVMLALatentCacheRuntime,
    skv_dequantize_rank_chunk,
    skv_glm_compressed_attention_scores,
)
from .._registry import register
from .._types import PatchConfig


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}={raw!r}; expected integer.") from exc


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return str(raw)


class _GLM47SKVMLAPattern:
    @property
    def name(self) -> str:
        return "skv_mla"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False
        if module.__class__.__name__ != "Glm4MoeLiteAttention":
            return False
        if os.environ.get("ZMLX_SKV_MLA_ENABLE", "0") not in {"1", "true", "TRUE"}:
            return False
        required = (
            "num_heads",
            "q_head_dim",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "kv_lora_rank",
            "scale",
            "embed_q",
            "unembed_out",
            "o_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
            "rope",
        )
        return all(hasattr(module, attr) for attr in required)

    def apply(self, module: Any, config: PatchConfig) -> Any:
        from mlx_lm.models.base import scaled_dot_product_attention

        original_call = module.__call__.__func__ if hasattr(module.__call__, "__func__") else module.__call__

        rank = _env_int("ZMLX_SKV_MLA_RANK", 128)
        bits = _env_int("ZMLX_SKV_MLA_BITS", 4)
        group_size = _env_int("ZMLX_SKV_MLA_GROUP_SIZE", 32)
        warmup_tokens = _env_int("ZMLX_SKV_MLA_WARMUP_TOKENS", max(rank, 64))
        strategy = _env_str("ZMLX_SKV_MLA_STRATEGY", "B").strip().upper()

        def patched_call(
            self_mod: Any,
            x: mx.array,
            mask: mx.array | None = None,
            cache: Any | None = None,
        ) -> Any:
            # Keep quantized cache path untouched.
            if cache is not None and hasattr(cache, "bits"):
                return original_call(self_mod, x, mask=mask, cache=cache)
            if cache is None:
                return original_call(self_mod, x, mask=mask, cache=cache)

            B, L, _ = x.shape
            if int(B) != 1:
                return original_call(self_mod, x, mask=mask, cache=cache)

            if self_mod.q_lora_rank is None:
                q = self_mod.q_proj(x)
            else:
                q = self_mod.q_b_proj(self_mod.q_a_layernorm(self_mod.q_a_proj(x)))

            q = q.reshape(B, L, self_mod.num_heads, self_mod.q_head_dim).transpose(0, 2, 1, 3)
            q_nope, q_pe = mx.split(q, [self_mod.qk_nope_head_dim], axis=-1)
            compressed_kv = self_mod.kv_a_proj_with_mqa(x)
            compressed_kv, k_pe = mx.split(compressed_kv, [self_mod.kv_lora_rank], axis=-1)
            k_pe = k_pe.reshape(B, L, 1, self_mod.qk_rope_head_dim).transpose(0, 2, 1, 3)
            kv_latent = self_mod.kv_a_layernorm(compressed_kv)

            offset = int(cache.offset)
            q_pe = self_mod.rope(q_pe, offset)
            k_pe = self_mod.rope(k_pe, offset)

            kv_latent = mx.expand_dims(kv_latent, axis=1)
            q_nope = self_mod.embed_q(q_nope)
            queries = mx.concatenate([q_nope, q_pe], axis=-1)

            if not hasattr(self_mod, "_zmlx_skv_mla_runtime"):
                self_mod._zmlx_skv_mla_runtime = SKVMLALatentCacheRuntime(
                    kv_lora_rank=int(self_mod.kv_lora_rank),
                    rope_dim=int(self_mod.qk_rope_head_dim),
                    rank=min(rank, int(self_mod.kv_lora_rank) - 1),
                    bits=bits,
                    group_size=group_size,
                    warmup_tokens=warmup_tokens,
                )
            runtime = self_mod._zmlx_skv_mla_runtime
            if offset == 0 and int(runtime.offset) > 0:
                runtime.reset()

            keys_step = mx.concatenate([kv_latent, k_pe], axis=-1)
            runtime.ingest(keys_step)
            cache.offset = int(runtime.offset)
            # Keep mlx-lm cache contract valid without storing dense MLA keys.
            # stream_generate() evaluates c.state every step; KVCache.state
            # requires non-None keys/values with seq axis >= offset.
            cache.keys = mx.zeros((1, 1, int(runtime.offset), 0), dtype=keys_step.dtype)
            cache.values = mx.zeros((1, 1, int(runtime.offset), 0), dtype=keys_step.dtype)
            self_mod._zmlx_skv_mla_last_state = (
                runtime.compressed_chunks[-1] if runtime.compressed_chunks else None
            )

            # Strategy B decode: compressed score path in no-PE latent space
            # plus explicit RoPE score add. Prefill stays on dense SDPA path.
            use_strategy_b = (
                strategy == "B"
                and runtime.ready()
                and int(L) == 1
            )
            if use_strategy_b:
                num_heads = int(self_mod.num_heads)
                basis = runtime.basis
                chunks = runtime.compressed_chunks
                score_parts = [
                    skv_glm_compressed_attention_scores(
                        q_nope,
                        chunk,
                        basis,
                        num_heads=num_heads,
                        scale=float(self_mod.scale),
                    )
                    for chunk in chunks
                ]
                score_nope = (
                    score_parts[0]
                    if len(score_parts) == 1
                    else mx.concatenate(score_parts, axis=-1)
                )
                q_pe_shd = mx.transpose(q_pe, axes=(2, 1, 0, 3)).reshape(
                    int(L), num_heads, int(self_mod.qk_rope_head_dim)
                )
                k_rope_shd = runtime.rope_shd()
                k_rope_heads = mx.repeat(k_rope_shd, repeats=num_heads, axis=1)
                score_rope = (
                    mx.einsum("qhd,khd->hqk", q_pe_shd, k_rope_heads) * float(self_mod.scale)
                )

                scores = score_nope + score_rope
                probs = mx.softmax(scores.astype(mx.float32), axis=-1)

                # Value aggregation in rank-space:
                #   z_out = sum_i p_i * z_i, then latent_out = z_out @ basis^T.
                q_len = int(L)
                rank_dim = int(basis.shape[-1])
                z_out = mx.zeros((num_heads, q_len, rank_dim), dtype=mx.float32)
                start = 0
                for chunk in chunks:
                    k_len = int(chunk["shape"][0])
                    end = start + k_len
                    probs_chunk = probs[..., start:end]
                    z_chunk = skv_dequantize_rank_chunk(chunk, compute_dtype=mx.float32)
                    z_heads = mx.repeat(z_chunk, repeats=num_heads, axis=1)
                    z_out = z_out + mx.einsum("hqk,khr->hqr", probs_chunk, z_heads)
                    start = end

                latent_out = mx.einsum("hqr,dr->hqd", z_out, basis[0].astype(mx.float32))
                output = latent_out.astype(q_nope.dtype)[None, ...]
            else:
                keys, values = runtime.materialize_keys_values()
                output = scaled_dot_product_attention(
                    queries, keys, values, cache=None, scale=self_mod.scale, mask=mask
                )

            output = self_mod.unembed_out(output)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self_mod.o_proj(output)

        module._zmlx_skv_mla_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_GLM47SKVMLAPattern())
