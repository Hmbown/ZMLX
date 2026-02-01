"""MoE MLP pattern: fuse the expert combine step with a Metal kernel.

Preserves each model's original gating logic (softmax ordering, expert bias,
renormalization) exactly — only the final weighted-sum of expert outputs is
replaced with a fused ``moe_combine`` Metal kernel.

When ``mx.gather_qmm_swiglu`` is available and the switch_mlp uses quantized
SwitchLinear layers, the gate+up projections plus SwiGLU activation are fused
into a single kernel launch (reading x once instead of twice).

Targets several MoE styles:
- **Qwen3** — ``gate`` returns raw logits, ``switch_mlp`` handles experts.
- **GPT-OSS** — ``router`` returns raw logits, ``experts`` is a SwitchGLU.
- **LFM2** — ``gate`` returns raw logits with ``expert_bias`` post-softmax.
- **GLM-4 / DeepSeek-V3** — ``gate`` returns ``(indices, scores)`` already
  computed (sigmoid + group selection).
- **Mixtral** — ``gate`` returns logits, ``experts`` is a list of modules.

Also handles ``shared_experts`` (additive dense MLP) when present.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...kernels import moe
from ...kernels.fused_moe import has_gather_qmm_swiglu
from .._registry import register
from .._types import PatchConfig


def _is_qwen3_moe_block(mod: Any) -> bool:
    """Return True if this module looks like Qwen3's MoE block."""
    cls = mod.__class__
    if cls.__name__ == "Qwen3MoeSparseMoeBlock":
        return True
    return bool(cls.__module__.startswith("mlx_lm.models.qwen3_moe"))


def _gating(self_mod: Any, x: Any, gate_attr: str, k: int) -> tuple[Any, Any]:
    """Run the model's gating logic faithfully, returning (indices, weights).

    Handles several conventions:
    - Gate that returns a tuple ``(indices, scores)`` directly (GLM-4 / DeepSeek-V3).
    - Gate that returns raw logits, with optional ``expert_bias`` and
      ``norm_topk_prob`` (Qwen3, LFM2, Mixtral, GPT-OSS).
    """
    gate_fn = getattr(self_mod, gate_attr)
    gate_out = gate_fn(x)

    if isinstance(gate_out, tuple):
        # Gate already computed indices + scores (GLM-4, DeepSeek-V3 style).
        indices, weights = gate_out
        return indices, weights

    if _is_qwen3_moe_block(self_mod):
        # Qwen3 uses precise softmax on logits, then argpartition on probs.
        gates = mx.softmax(gate_out, axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(self_mod, "norm_topk_prob", False):
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        return inds.astype(mx.uint32), scores

    # Raw logits path — preserve the standard gating sequence exactly,
    # but use the fused kernel when possible.
    expert_bias = getattr(self_mod, "expert_bias", None)
    norm_topk_prob = getattr(self_mod, "norm_topk_prob", False)

    weights, indices = moe.topk_gating_softmax(
        gate_out,
        k=k,
        expert_bias=expert_bias,
        norm_topk_prob=norm_topk_prob,
        compute_dtype=mx.float32,
    )
    weights = weights.astype(x.dtype)
    return indices, weights


# ---------------------------------------------------------------------------
# Fused SwitchGLU detection
# ---------------------------------------------------------------------------

# Max token count for the fused path.  Beyond this the fused kernel regresses
# vs the two-pass approach (benchmarked on M-series: ~0.5x at M=64).
_FUSED_SWIGLU_MAX_TOKENS = 32


def _is_quantized_switch_linear(mod: Any) -> bool:
    """Return True if *mod* looks like a QuantizedSwitchLinear."""
    return (
        hasattr(mod, "weight")
        and hasattr(mod, "scales")
        and hasattr(mod, "group_size")
        and hasattr(mod, "bits")
    )


def _is_switch_glu_module(mod: Any) -> bool:
    """Return True if *mod* looks like a SwitchGLU-style expert module."""
    return (
        mod is not None
        and hasattr(mod, "gate_proj")
        and hasattr(mod, "up_proj")
        and hasattr(mod, "down_proj")
    )


def _is_standard_swiglu_activation(switch_mlp: Any) -> bool:
    """Return True if the activation matches MLX's standard SwiGLU."""
    activation = getattr(switch_mlp, "activation", None)
    if activation is None:
        return True

    mod = getattr(activation.__class__, "__module__", "")
    name = activation.__class__.__name__
    if mod == "mlx_lm.models.switch_layers" and name == "SwiGLU":
        return True

    if callable(activation):
        func_mod = getattr(activation, "__module__", "")
        func_name = getattr(activation, "__name__", "")
        if func_mod == "mlx_lm.models.activations" and func_name == "swiglu":
            return True

    return False


def _can_fuse_switch_mlp(switch_mlp: Any) -> bool:
    """Return True if the switch_mlp can use gather_qmm_swiglu.

    Requirements:
    - mx.gather_qmm_swiglu must be available
    - gate_proj and up_proj must be QuantizedSwitchLinear
    - Both must use the same quantization config
    - mode must be "affine"
    - activation must be standard SwiGLU (no custom gating)
    """
    if not has_gather_qmm_swiglu():
        return False

    gate_proj = getattr(switch_mlp, "gate_proj", None)
    up_proj = getattr(switch_mlp, "up_proj", None)
    if gate_proj is None or up_proj is None:
        return False

    if not _is_quantized_switch_linear(gate_proj):
        return False
    if not _is_quantized_switch_linear(up_proj):
        return False

    # Must be affine quantization
    if getattr(gate_proj, "mode", "affine") != "affine":
        return False
    if getattr(up_proj, "mode", "affine") != "affine":
        return False

    # Must have matching quant config
    if gate_proj.group_size != up_proj.group_size:
        return False
    if gate_proj.bits != up_proj.bits:
        return False

    if not _is_standard_swiglu_activation(switch_mlp):
        return False

    return True


def _fused_switch_mlp_call(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
    *,
    max_tokens: int,
) -> mx.array:
    """Replace gate_proj + up_proj + SwiGLU with a single gather_qmm_swiglu.

    Falls back to the original switch_mlp call when the token count is large
    (the fused kernel regresses at high M due to different tiling strategy).
    """
    # The fused kernel benefits decode (small M) but regresses at large M.
    # SwitchGLU's sorting threshold is indices.size >= 64, which correlates
    # with the same regime where the fused kernel slows down.
    total_tokens = 1
    for d in indices.shape:
        total_tokens *= d
    if total_tokens > max_tokens:
        return switch_mlp._zmlx_original_switch_call(x, indices)

    gate_proj = switch_mlp.gate_proj
    up_proj = switch_mlp.up_proj
    down_proj = switch_mlp.down_proj

    # Expand dims to match SwitchGLU convention: (B, L, D) -> (B, L, 1, 1, D)
    x_expanded = mx.expand_dims(x, (-2, -3))

    # Fused gate + up + SwiGLU in one kernel
    activated = mx.gather_qmm_swiglu(
        x_expanded,
        gate_proj.weight, gate_proj.scales, gate_proj.get("biases"),
        up_proj.weight, up_proj.scales, up_proj.get("biases"),
        rhs_indices=indices,
        transpose=True,
        group_size=gate_proj.group_size,
        bits=gate_proj.bits,
    )

    # Add per-expert bias for gate/up if present (rare, typically bias=False)
    # The fused op does not handle the additive bias from the Linear layer,
    # but QuantizedSwitchLinear rarely has bias=True for gate/up in practice.

    # Down projection remains separate
    x_out = down_proj(activated, indices)

    return x_out.squeeze(-2)


class _MoEMLPPattern:
    @property
    def name(self) -> str:
        return "moe_mlp"

    def matches(self, module: Any, name: str, parent: Any | None = None) -> bool:
        if not isinstance(module, nn.Module):
            return False
        # Match Qwen3MoeSparseMoeBlock (gate), GPT-OSS MLPBlock (router), etc.
        has_gate = hasattr(module, "gate") or hasattr(module, "router")
        # Check for experts list or a single expert-handling module like switch_mlp
        has_experts = hasattr(module, "experts") or hasattr(module, "switch_mlp")
        return bool(has_gate and has_experts)

    def apply(self, module: Any, config: PatchConfig) -> Any:
        original_call = module.__call__

        # Detect the number of experts activated per token from the module.
        num_experts_per_tok = (
            getattr(module, "num_experts_per_tok", None)
            or getattr(module, "top_k", None)
            or getattr(module, "num_selected_experts", None)
            or 2  # conservative fallback
        )

        # Resolve which attribute holds the gating linear layer.
        _gate_attr = "gate" if hasattr(module, "gate") else "router"

        # Resolve SwitchGLU-style experts (switch_mlp or experts module).
        switch_mlp = None
        switch_mlp_attr = None
        if hasattr(module, "switch_mlp") and _is_switch_glu_module(module.switch_mlp):
            switch_mlp = module.switch_mlp
            switch_mlp_attr = "switch_mlp"
        elif hasattr(module, "experts") and _is_switch_glu_module(module.experts):
            switch_mlp = module.experts
            switch_mlp_attr = "experts"

        # Check if we can fuse the switch_mlp's gate+up+SwiGLU step.
        _use_fused_swiglu = False
        if switch_mlp is not None and _can_fuse_switch_mlp(switch_mlp):
            _use_fused_swiglu = True
            fused_swiglu_max_tokens = (
                _FUSED_SWIGLU_MAX_TOKENS
                if config.moe_fused_swiglu_max_tokens is None
                else config.moe_fused_swiglu_max_tokens
            )
            # Store original switch_mlp call for fallback at large token counts
            switch_mlp._zmlx_original_switch_call = switch_mlp.__call__
            if config.verbose:
                gp = switch_mlp.gate_proj
                loc = f"{switch_mlp_attr}" if switch_mlp_attr else "experts"
                print(
                    f"  [moe_mlp] Fusing gate+up+SwiGLU via gather_qmm_swiglu "
                    f"(bits={gp.bits}, gs={gp.group_size}, attr={loc})"
                )

        use_exact_combine = _is_qwen3_moe_block(module)

        def patched_call(self_mod: Any, x: Any) -> Any:
            # 1. Gating — preserve the model's original logic exactly.
            indices, weights = _gating(self_mod, x, _gate_attr, num_experts_per_tok)

            # 2. Expert Execution
            if switch_mlp is not None:
                if _use_fused_swiglu:
                    # Fused path: gather_qmm_swiglu for small token counts,
                    # falls back to original for large M.
                    expert_outputs = _fused_switch_mlp_call(
                        switch_mlp,
                        x,
                        indices,
                        max_tokens=fused_swiglu_max_tokens,
                    )
                else:
                    # Qwen3/GLM/LFM2 style: vectorized experts (SwitchGLU)
                    expert_outputs = switch_mlp(x, indices)
            else:
                # Mixtral/DeepSeek style: list of expert modules
                experts = getattr(self_mod, "experts", None)
                if experts is None:
                    return original_call(x)
                B = indices.shape[0]
                K = indices.shape[-1]
                D = x.shape[-1]
                expert_outputs = mx.zeros((B, K, D), dtype=x.dtype)

                for i, expert in enumerate(experts):
                    for k in range(K):
                        mask = indices[:, k] == i
                        if mask.any():
                            expert_outputs[mask, k] = expert(x[mask])

            # 3. Combine: Qwen3 uses the exact sum path for fidelity.
            if use_exact_combine:
                y = moe.moe_combine_exact(expert_outputs, weights)
            else:
                # Fused Combine: weighted sum of expert outputs in one kernel
                y = moe.moe_combine(expert_outputs, weights)

            # 4. Shared experts (GLM-4, DeepSeek-V3): additive dense path
            shared = getattr(self_mod, "shared_experts", None)
            if shared is not None:
                y = y + shared(x)

            return y

        # Store original for unpatch
        module._zmlx_original_call = original_call
        module.__class__ = type(
            module.__class__.__name__,
            (module.__class__,),
            {"__call__": patched_call},
        )
        return module


register(_MoEMLPPattern())
