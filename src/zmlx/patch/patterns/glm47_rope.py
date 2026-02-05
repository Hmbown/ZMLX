"""GLM-4.7-Flash RoPE optimization (decode-only).

Fuses:
  - RoPE(q_pe) + concat([q_nope, q_pe])  -> queries
  - RoPE(k_pe) + concat([kv_latent, k_pe]) -> keys (current step)

into a single Metal kernel for decode (T=1). This reduces dispatch overhead in
each GLM-4.7-Flash attention layer.

This pattern is opt-in (not part of default presets) because it is model-
specific and only targets `glm4_moe_lite` attention.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...kernels.rope import _rope_cos_sin, rope_concat_qk_decode_pos
from .._registry import register
from .._types import PatchConfig


def _is_vanilla_rope(mod: Any) -> bool:
    cls = mod.__class__
    return cls.__name__ == "RoPE" and "mlx.nn.layers.positional_encoding" in (cls.__module__ or "")


class _GLM47RopePattern:
    @property
    def name(self) -> str:
        return "glm47_rope"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False

        # Specifically target GLM-4.7-Flash's attention implementation.
        if module.__class__.__name__ != "Glm4MoeLiteAttention":
            return False

        rope = getattr(module, "rope", None)
        if rope is None or not _is_vanilla_rope(rope):
            return False
        if not getattr(rope, "traditional", False):
            return False

        dims = getattr(rope, "dims", None)
        if dims is None or int(dims) <= 0 or int(dims) % 2 != 0:
            return False

        # Required attributes for the optimized decode path.
        required = (
            "num_heads",
            "q_head_dim",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "kv_lora_rank",
            "max_position_embeddings",
            "scale",
            "embed_q",
            "unembed_out",
            "o_proj",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
        )
        return all(hasattr(module, attr) for attr in required)

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = module.__call__.__func__ if hasattr(module.__call__, "__func__") else module.__call__

        from mlx_lm.models.base import scaled_dot_product_attention

        def _next_pow2(n: int) -> int:
            if n <= 1:
                return 1
            return 1 << (n - 1).bit_length()

        def patched_call(
            self_mod: Any,
            x: mx.array,
            mask: mx.array | None = None,
            cache: Any | None = None,
        ) -> Any:
            # Only optimize decode steps (T=1). Fall back to upstream for prefill.
            try:
                B, L, _ = x.shape
            except Exception:
                return original_call(self_mod, x, mask=mask, cache=cache)

            if L != 1:
                return original_call(self_mod, x, mask=mask, cache=cache)

            rope = getattr(self_mod, "rope", None)
            if rope is None or not _is_vanilla_rope(rope) or not getattr(rope, "traditional", False):
                return original_call(self_mod, x, mask=mask, cache=cache)

            max_pos = int(getattr(self_mod, "max_position_embeddings", 0) or 0)
            if max_pos <= 0:
                return original_call(self_mod, x, mask=mask, cache=cache)

            # Offset can be int, scalar array, or (B,) vector. For safety, only
            # use the fused kernel when offsets are in-bounds for the precomputed table.
            offset = cache.offset if cache is not None else 0
            if isinstance(offset, mx.array):
                # Vector offsets can be negative (BatchKVCache); still supported,
                # but avoid an extra reduction here and just fall back.
                if int(offset.ndim) != 0:
                    return original_call(self_mod, x, mask=mask, cache=cache)
                try:
                    off_val = int(offset.item())
                except Exception:
                    return original_call(self_mod, x, mask=mask, cache=cache)
                if abs(off_val) >= max_pos:
                    return original_call(self_mod, x, mask=mask, cache=cache)
            else:
                try:
                    off_val = int(offset)
                except Exception:
                    return original_call(self_mod, x, mask=mask, cache=cache)
                if abs(off_val) >= max_pos:
                    return original_call(self_mod, x, mask=mask, cache=cache)

            # --- Upstream attention math (mirrors mlx_lm.models.glm4_moe_lite) ---
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

            kv_latent = mx.expand_dims(kv_latent, axis=1)
            q_nope = self_mod.embed_q(q_nope)

            # Precompute just enough cos/sin rows to cover the current offset.
            # GLM's `max_position_embeddings` is ~202k, which is expensive to
            # materialize eagerly; grow the table in powers of two as needed.
            needed = abs(int(off_val)) + 1
            table_len = min(max_pos, _next_pow2(needed))
            tables = _rope_cos_sin(table_len, int(rope.dims), float(rope.base), float(rope.scale))
            pos = abs(int(off_val))
            cos = tables.cos[pos]
            sin = tables.sin[pos]
            if int(off_val) < 0:
                sin = -sin
            queries, keys_step = rope_concat_qk_decode_pos(
                q_nope,
                q_pe,
                kv_latent,
                k_pe,
                cos,
                sin,
            )

            if cache is not None:
                if hasattr(cache, "bits"):
                    # QuantizedKVCache: cache values with rope dims included.
                    keys, values = cache.update_and_fetch(keys_step, keys_step)
                else:
                    keys, _ = cache.update_and_fetch(keys_step, mx.zeros((B, 1, L, 0)))
                    values = keys[..., : -self_mod.qk_rope_head_dim]
            else:
                keys = keys_step
                values = keys[..., : -self_mod.qk_rope_head_dim]

            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self_mod.scale, mask=mask
            )

            if cache is not None and hasattr(cache, "bits"):
                output = output[..., : self_mod.kv_lora_rank]

            output = self_mod.unembed_out(output)
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self_mod.o_proj(output)

        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_GLM47RopePattern())
