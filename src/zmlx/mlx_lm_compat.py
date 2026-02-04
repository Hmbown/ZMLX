"""Compatibility patches for upstream mlx-lm model implementations.

These are *surgical* monkey-patches applied only when needed for optional
features (e.g. quantized KV cache). They should preserve behavior for the
default (non-quantized) path.
"""

from __future__ import annotations

import types
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def _is_glm4_moe_lite_attention(mod: Any) -> bool:
    cls = mod.__class__
    name = getattr(cls, "__name__", "")
    mod_path = getattr(cls, "__module__", "") or ""
    if name == "Glm4MoeLiteAttention" and "glm4_moe_lite" in mod_path:
        return True
    return False


def _patch_glm4_moe_lite_attention_quant_kv(attn: nn.Module) -> bool:
    """Patch GLM-4 MoE Lite attention to support QuantizedKVCache.

    Upstream GLM-4 MoE Lite stores only `keys` in the cache and derives `values`
    as a slice of `keys`. QuantizedKVCache returns a nested quantized structure
    (tuple of packed weights + scales + biases), so slicing fails.

    This patch keeps the original path for non-quantized caches, and uses a
    separate `values` cache when `cache.bits` is present.
    """
    if getattr(attn, "_zmlx_glm4_quant_kv_patched", False):
        return False

    original_call = attn.__call__

    def patched_call(
        self_mod: Any,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> Any:
        if cache is None or not hasattr(cache, "bits"):
            return self_mod._zmlx_glm4_quant_kv_original_call(x, mask=mask, cache=cache)

        from mlx_lm.models.base import scaled_dot_product_attention

        B, L, _ = x.shape

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

        offset = cache.offset if cache is not None else 0
        q_pe = self_mod.rope(q_pe, offset)
        k_pe = self_mod.rope(k_pe, offset)

        kv_latent = mx.expand_dims(kv_latent, axis=1)
        q_nope = self_mod.embed_q(q_nope)

        keys = mx.concatenate([kv_latent, k_pe], axis=-1)

        # mlx-lm's quantized SDPA currently assumes key/value dim == query dim.
        # GLM uses keys dim (kv + rope) but values dim (kv only). Work around
        # this by caching values with rope dims included, then slicing the
        # attention output back to kv dims before unembedding.
        keys, values = cache.update_and_fetch(keys, keys)

        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self_mod.scale, mask=mask
        )

        output = output[..., : self_mod.kv_lora_rank]
        output = self_mod.unembed_out(output)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self_mod.o_proj(output)

    attn._zmlx_glm4_quant_kv_original_call = original_call  # type: ignore[attr-defined]
    attn._zmlx_glm4_quant_kv_patched = True  # type: ignore[attr-defined]

    attn.__class__ = type(
        attn.__class__.__name__,
        (attn.__class__,),
        {"__call__": patched_call},
    )
    return True


def _walk_children(module: Any) -> list[Any]:
    children: dict[str, Any] = {}
    if hasattr(module, "children") and callable(module.children):
        children = dict(module.children())
    out: list[Any] = []
    for child in children.values():
        if isinstance(child, list):
            for item in child:
                if isinstance(item, nn.Module):
                    out.append(item)
        elif isinstance(child, nn.Module):
            out.append(child)
    return out


def apply_kv_quantization_fixes(model: nn.Module, *, kv_bits: int | None, verbose: bool = False) -> int:
    """Apply model-side fixes required for quantized KV cache generation."""
    if kv_bits is None:
        return 0

    patched = 0
    stack: list[Any] = [model]
    while stack:
        mod = stack.pop()
        if isinstance(mod, nn.Module) and _is_glm4_moe_lite_attention(mod):
            if _patch_glm4_moe_lite_attention_quant_kv(mod):
                patched += 1
        stack.extend(_walk_children(mod))

    if verbose and patched:
        print(f"[zmlx.compat] Applied GLM-4 quantized KV fix to {patched} attention modules.")
    return patched


def make_prompt_cache_for_kv_quantization(model: nn.Module) -> list[Any]:
    """Create a prompt cache that is safe to convert to QuantizedKVCache.

    mlx-lm's default kv_bits implementation converts KVCache -> QuantizedKVCache
    after a model call. Some models store only keys (and pass values with
    v_head_dim==0), which breaks quantized attention. We wrap KVCache.to_quantized
    so that, when values are empty, it derives quantized values from keys.

    This preserves the default (unquantized) path until mlx-lm flips caches to
    quantized, and does not change behavior for caches that already store values.
    """
    from mlx_lm.models import cache as mlx_cache

    prompt_cache: list[Any] = mlx_cache.make_prompt_cache(model)

    for c in prompt_cache:
        if not hasattr(c, "to_quantized"):
            continue
        if getattr(c, "_zmlx_to_quantized_wrapped", False):
            continue

        c._zmlx_original_to_quantized = c.to_quantized  # type: ignore[attr-defined]

        def _wrapped_to_quantized(self_mod: Any, group_size: int = 64, bits: int = 4):
            quant_cache = self_mod._zmlx_original_to_quantized(  # type: ignore[attr-defined]
                group_size=group_size,
                bits=bits,
            )
            try:
                values = getattr(self_mod, "values", None)
                keys = getattr(self_mod, "keys", None)
                if values is None or keys is None:
                    return quant_cache
                if hasattr(values, "shape") and int(values.shape[-1]) == 0:
                    quant_cache.values = mx.quantize(
                        keys, group_size=group_size, bits=bits
                    )
            except Exception:
                pass
            return quant_cache

        c.to_quantized = types.MethodType(_wrapped_to_quantized, c)
        c._zmlx_to_quantized_wrapped = True  # type: ignore[attr-defined]

    return prompt_cache
