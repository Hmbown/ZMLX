"""DeltaNet pattern: fuse decode operations in GatedDeltaNet layers.

Structural match: module has separate input projections (in_proj_qkv,
in_proj_z, in_proj_b, in_proj_a), depthwise conv1d, and gated RMSNorm output
(norm). This matches the ``qwen3_5.GatedDeltaNet`` layout used in
Qwen3.5-35B-A3B and similar models.

Decode-only fusions (S=1):
  - fused_conv1d_silu: replaces conv1d + silu with 1 Metal kernel
  - fused_input_proj: replaces 4 matmuls with 1 (dense weights only)

Note: gated_rmsnorm_silu is available but NOT used by default because its
threadgroup reduction ordering differs from mx.fast.rms_norm, causing ~2e-3
max diff in bfloat16 that accumulates across 30 layers and breaks fidelity.
The original self.norm(out, z) is used instead (calls mx.fast.rms_norm).

Prefill (S>1) falls through to the original forward pass unchanged.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...kernels.deltanet import (
    fused_conv1d_silu,
    fused_input_proj,
)
from .._registry import register
from .._types import PatchConfig


class _DeltaNetPattern:
    @property
    def name(self) -> str:
        return "deltanet"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False
        # Structural check: must have all the separate-projection attributes
        # that characterise qwen3_5.GatedDeltaNet.
        required = (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "conv1d",
            "norm",
            "out_proj",
            "A_log",
            "dt_bias",
            "num_v_heads",
            "num_k_heads",
            "head_k_dim",
            "head_v_dim",
            "key_dim",
            "value_dim",
            "conv_dim",
            "conv_kernel_size",
        )
        return all(hasattr(module, attr) for attr in required)

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = (
            module.__call__.__func__
            if hasattr(module.__call__, "__func__")
            else module.__call__
        )

        # Lazily import gated_delta_update at patch time — mlx_lm must be
        # present if the module matched, so this is safe.
        from mlx_lm.models.gated_delta import gated_delta_update

        def patched_call(
            self_mod: Any,
            inputs: mx.array,
            mask: Any | None = None,
            cache: Any | None = None,
        ) -> mx.array:
            B, S, _ = inputs.shape

            if S > 1:
                # Prefill: use original forward pass
                return original_call(self_mod, inputs, mask, cache)

            # =================================================================
            # Decode path (S=1): fused kernels
            # =================================================================

            # --- 1. Input projections ---
            # fused_input_proj works with dense weights only; quantized models
            # fall back to separate calls.
            use_fused_proj = (
                not isinstance(self_mod.in_proj_qkv, nn.QuantizedLinear)
                and hasattr(self_mod.in_proj_qkv, "weight")
            )

            if use_fused_proj:
                qkv, z_flat, b, a = fused_input_proj(
                    inputs,
                    self_mod.in_proj_qkv.weight,
                    self_mod.in_proj_z.weight,
                    self_mod.in_proj_b.weight,
                    self_mod.in_proj_a.weight,
                )
            else:
                qkv = self_mod.in_proj_qkv(inputs)
                z_flat = self_mod.in_proj_z(inputs)
                b = self_mod.in_proj_b(inputs)
                a = self_mod.in_proj_a(inputs)

            z = z_flat.reshape(B, S, self_mod.num_v_heads, self_mod.head_v_dim)

            # --- 2. Conv state ---
            if cache is not None and cache[0] is not None:
                conv_state = cache[0]
            else:
                conv_state = mx.zeros(
                    (B, self_mod.conv_kernel_size - 1, self_mod.conv_dim),
                    dtype=inputs.dtype,
                )

            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)

            # --- 3. Fused conv1d + SiLU ---
            conv_out, new_conv_state = fused_conv1d_silu(
                conv_input, self_mod.conv1d.weight
            )
            if cache is not None:
                cache[0] = new_conv_state

            # --- 4. Split q, k, v ---
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(
                        conv_out,
                        [self_mod.key_dim, 2 * self_mod.key_dim],
                        -1,
                    ),
                    [self_mod.num_k_heads, self_mod.num_k_heads, self_mod.num_v_heads],
                    [self_mod.head_k_dim, self_mod.head_k_dim, self_mod.head_v_dim],
                )
            ]

            # --- 5. QK norm + gated delta update (existing Metal kernel) ---
            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

            out, state = gated_delta_update(
                q, k, v, a, b,
                self_mod.A_log,
                self_mod.dt_bias,
                state,
                mask,
                use_kernel=True,
            )

            if cache is not None:
                cache[1] = state

            # --- 6. Gated RMSNorm + SiLU (use original for fidelity) ---
            # gated_rmsnorm_silu has different reduction ordering vs
            # mx.fast.rms_norm, causing ~2e-3 max diff that accumulates.
            out = self_mod.norm(out, z)

            return self_mod.out_proj(out.reshape(B, S, -1))

        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_DeltaNetPattern())
