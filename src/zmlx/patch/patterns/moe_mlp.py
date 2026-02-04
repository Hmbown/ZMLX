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

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...kernels import moe, transformer
from ...kernels.fused_moe import has_gather_qmm_swiglu
from .._registry import register
from .._types import PatchConfig


def _is_qwen3_moe_block(mod: Any) -> bool:
    """Return True if this module looks like Qwen3's MoE block."""
    cls = mod.__class__
    if cls.__name__ == "Qwen3MoeSparseMoeBlock":
        return True
    return bool(cls.__module__.startswith("mlx_lm.models.qwen3_moe"))


def _is_gpt_oss_block(mod: Any) -> bool:
    """Return True if this module looks like a GPT-OSS MoE block."""
    cls = mod.__class__
    mod_path = getattr(cls, "__module__", "") or ""
    return "gpt_oss" in mod_path


def _is_lfm2_moe_block(mod: Any) -> bool:
    """Return True if this module looks like an LFM2 MoE block."""
    cls = mod.__class__
    mod_path = getattr(cls, "__module__", "") or ""
    combined = f"{mod_path} {cls.__name__}".lower()
    return "lfm2" in combined or "lfm" in combined


def _is_glm_moe_block(mod: Any) -> bool:
    """Return True if this module looks like a GLM MoE block."""
    cls = mod.__class__
    mod_path = (getattr(cls, "__module__", "") or "").lower()
    name = cls.__name__.lower()
    return "glm" in mod_path or "glm" in name


def _glm_allow_fused_swiglu() -> bool:
    """Return True if GLM should use fused SwiGLU.

    Default: enabled when ``mx.gather_qmm_swiglu`` is available.
    Set ``ZMLX_GLM_FUSED_SWIGLU=0`` to disable.
    """
    raw = os.environ.get("ZMLX_GLM_FUSED_SWIGLU")
    if raw is None:
        return True
    try:
        return int(raw) != 0
    except ValueError:
        return True


def _gating(
    self_mod: Any,
    x: Any,
    gate_attr: str,
    k: int,
    *,
    is_qwen3: bool = False,
    is_gpt_oss: bool = False,
) -> tuple[Any, Any]:
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

    if is_qwen3:
        # Qwen3 uses precise softmax on logits, then argpartition on probs.
        gates = mx.softmax(gate_out, axis=-1, precise=True)  # type: ignore[call-arg]
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if getattr(self_mod, "norm_topk_prob", False):
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        return inds.astype(mx.uint32), scores

    if is_gpt_oss:
        # GPT-OSS does top-k on raw logits first, then softmax on only K values.
        # This produces different probabilities than full-softmax-then-top-k.
        inds = mx.argpartition(gate_out, kth=-k, axis=-1)[..., -k:]
        top_k_values = mx.take_along_axis(gate_out, inds, axis=-1)
        scores = mx.softmax(top_k_values, axis=-1, precise=True)  # type: ignore[call-arg]
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
_FUSED_SWIGLU_MAX_TOKENS = 1
_MOE_STREAMS_ENV = "ZMLX_MOE_STREAMS"
_MOE_STREAMS_REDUCE_ENV = "ZMLX_MOE_STREAMS_REDUCE"


def _get_moe_stream_pool() -> list[mx.Stream] | None:
    """Return a deterministic GPU stream pool for MoE dispatch."""
    raw = os.environ.get(_MOE_STREAMS_ENV)
    if raw is None:
        return None
    try:
        count = int(raw)
    except ValueError:
        return None
    if count <= 1:
        return None
    gpu = mx.Device(mx.gpu)
    if not mx.is_available(gpu):
        return None
    streams = [mx.default_stream(gpu)]
    for _ in range(count - 1):
        streams.append(mx.new_stream(gpu))
    return streams


def _get_moe_stream_reduce() -> str:
    """Return the reduction mode for stream outputs."""
    raw = os.environ.get(_MOE_STREAMS_REDUCE_ENV, "").strip().lower()
    if raw in {"tree", "stack"}:
        return raw
    return "serial"


def _reduce_stream_outputs(stream_outputs: list[mx.array], mode: str) -> mx.array:
    """Combine per-stream outputs into a single expert output tensor."""
    if not stream_outputs:
        raise ValueError("stream_outputs cannot be empty")

    if mode == "stack":
        # Experimental: stack + sum reduces Python overhead at cost of temp memory.
        stacked = mx.stack(stream_outputs, axis=0)
        return mx.sum(stacked, axis=0)

    if mode == "tree":
        # Experimental: pairwise tree reduction to shorten dependency chain.
        outputs = stream_outputs
        while len(outputs) > 1:
            next_round: list[mx.array] = []
            it = iter(outputs)
            for left in it:
                right = next(it, None)
                if right is None:
                    next_round.append(left)
                else:
                    next_round.append(left + right)
            outputs = next_round
        return outputs[0]

    # Default: serial accumulation (most deterministic vs baseline)
    out = stream_outputs[0]
    for extra in stream_outputs[1:]:
        out = out + extra
    return out


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


def _is_lora_like(mod: Any) -> bool:
    """Return True if *mod* looks like a LoRA/DoRA wrapper."""
    cls = mod.__class__
    name = cls.__name__.lower()
    mod_path = (getattr(cls, "__module__", "") or "").lower()
    return "lora" in name or "dora" in name or "lora" in mod_path or "dora" in mod_path


def _flatten_for_fused_combine(
    act: Any,
    gate: Any,
    indices: Any,
) -> tuple[Any, Any, Any, tuple[int, ...]] | None:
    """Normalize MoE tensors to 3D for gather_qmm_combine helpers."""
    if act.ndim != indices.ndim + 1:
        return None
    if gate.shape != indices.shape:
        return None
    if act.shape[:-1] != indices.shape:
        return None
    k = indices.shape[-1]
    act_flat = act.reshape(-1, k, act.shape[-1])
    gate_flat = gate.reshape(-1, k)
    indices_flat = indices.reshape(-1, k)
    return act_flat, gate_flat, indices_flat, gate.shape[:-1]


def _prepare_downproj_weights(weights: Any, d_in: int) -> Any | None:
    """Normalize SwitchLinear weights to (E, D_in, D_out) layout."""
    if weights.ndim != 3:
        return None
    if weights.shape[1] == d_in:
        return weights
    if weights.shape[2] == d_in:
        return mx.swapaxes(weights, 1, 2)
    return None


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


def _should_fuse_swiglu_tokens(total_tokens: int, max_tokens: int) -> bool:
    """Return True when fused SwiGLU should be used for the token count."""
    return total_tokens <= max_tokens


def _fused_switch_mlp_call(
    switch_mlp: Any,
    x: mx.array,
    indices: mx.array,
    *,
    max_tokens: int,
) -> Any:
    """Replace gate_proj + up_proj + SwiGLU with a single gather_qmm_swiglu.

    Falls back to the original switch_mlp call when the token count is large
    (the fused kernel regresses at high M due to different tiling strategy).
    """
    # The fused kernel benefits decode (small M) but regresses at large M.
    # SwitchGLU's sorting threshold is indices.size >= 64, which correlates
    # with the same regime where the fused kernel slows down.
    # indices: (..., K) where K is experts-per-token; count tokens excluding K.
    k = int(indices.shape[-1]) if indices.ndim else 1
    total_tokens = int(indices.size) // max(1, k)
    if not _should_fuse_swiglu_tokens(total_tokens, max_tokens):
        return switch_mlp._zmlx_original_switch_call(x, indices)

    gate_proj = switch_mlp.gate_proj
    up_proj = switch_mlp.up_proj
    down_proj = switch_mlp.down_proj

    # Expand dims to match SwitchGLU convention: (B, L, D) -> (B, L, 1, 1, D)
    x_expanded = mx.expand_dims(x, (-2, -3))

    # Fused gate + up + SwiGLU in one kernel
    activated = mx.gather_qmm_swiglu(  # type: ignore[attr-defined]
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


def _try_fused_downproj_combine(
    switch_mlp: Any,
    x: Any,
    indices: Any,
    gate: Any,
    *,
    require_fp32: bool,
) -> Any | None:
    """Attempt fused down-projection + combine for SwitchGLU modules."""
    if not _is_standard_swiglu_activation(switch_mlp):
        return None

    gate_proj = getattr(switch_mlp, "gate_proj", None)
    up_proj = getattr(switch_mlp, "up_proj", None)
    down_proj = getattr(switch_mlp, "down_proj", None)
    if gate_proj is None or up_proj is None or down_proj is None:
        return None

    if _is_lora_like(down_proj):
        return None
    if getattr(down_proj, "bias", None) is not None:
        return None

    is_quantized = _is_quantized_switch_linear(down_proj)
    if is_quantized:
        if not hasattr(mx, "gather_qmm"):
            return None
        if getattr(down_proj, "mode", "affine") != "affine":
            return None

    if require_fp32 and gate.dtype != mx.float32:
        return None

    try:
        gate_act = gate_proj(x, indices)
        up_act = up_proj(x, indices)
        activated = transformer.swiglu2(gate_act, up_act)
    except Exception:
        return None
    flat = _flatten_for_fused_combine(activated, gate, indices)
    if flat is None:
        return None
    act_flat, gate_flat, indices_flat, out_shape = flat

    if require_fp32 and (act_flat.dtype != mx.float32 or gate_flat.dtype != mx.float32):
        return None
    if indices_flat.dtype != mx.uint32:
        indices_flat = indices_flat.astype(mx.uint32)

    if is_quantized:
        biases = down_proj.get("biases") if hasattr(down_proj, "get") else None
        out_flat = moe.gather_qmm_combine_quantized(
            act_flat,
            down_proj.weight,
            down_proj.scales,
            biases,
            gate_flat,
            indices_flat,
            group_size=down_proj.group_size,
            bits=down_proj.bits,
        )
    else:
        weights = getattr(down_proj, "weight", None)
        if weights is None:
            return None
        weights = _prepare_downproj_weights(weights, act_flat.shape[-1])
        if weights is None:
            return None
        out_flat = moe.gather_qmm_combine(
            act_flat,
            weights,
            gate_flat,
            indices_flat,
        )

    return out_flat.reshape((*out_shape, out_flat.shape[-1]))


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

        is_qwen3 = _is_qwen3_moe_block(module)
        is_lfm2 = _is_lfm2_moe_block(module)
        is_gpt_oss = _is_gpt_oss_block(module)
        is_glm = _is_glm_moe_block(module)
        use_exact_combine = is_qwen3
        if is_glm and _use_fused_swiglu and not _glm_allow_fused_swiglu():
            if config.verbose:
                print(
                    "  [moe_mlp] GLM detected: fused SwiGLU disabled via "
                    "ZMLX_GLM_FUSED_SWIGLU=0"
                )
            _use_fused_swiglu = False
        moe_stream_pool = _get_moe_stream_pool()
        moe_stream_reduce = None
        if moe_stream_pool is not None:
            moe_stream_reduce = _get_moe_stream_reduce()
        if moe_stream_pool is not None and config.verbose:
            print(
                f"  [moe_mlp] MoE stream pool enabled "
                f"({len(moe_stream_pool)} streams, reduce={moe_stream_reduce})"
            )

        def patched_call(self_mod: Any, x: Any) -> Any:
            # 1. Gating — preserve the model's original logic exactly.
            indices, gate_weights = _gating(
                self_mod,
                x,
                _gate_attr,
                num_experts_per_tok,
                is_qwen3=is_qwen3,
                is_gpt_oss=is_gpt_oss,
            )

            # 2. Expert Execution
            expert_outputs = None
            if switch_mlp is not None:
                fused_out = None
                if is_qwen3 or is_lfm2:
                    fused_out = _try_fused_downproj_combine(
                        switch_mlp,
                        x,
                        indices,
                        gate_weights,
                        require_fp32=use_exact_combine,
                    )

                if fused_out is not None:
                    y = fused_out
                else:
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
                if moe_stream_pool is None:
                    expert_outputs = mx.zeros((B, K, D), dtype=x.dtype)
                    for i, expert in enumerate(experts):
                        for k in range(K):
                            mask = indices[:, k] == i
                            if mask.any():
                                expert_outputs[mask, k] = expert(x[mask])
                else:
                    stream_count = len(moe_stream_pool)
                    stream_outputs: list[mx.array] = []
                    for stream in moe_stream_pool:
                        with mx.stream(stream):
                            stream_outputs.append(mx.zeros((B, K, D), dtype=x.dtype))
                    for i, expert in enumerate(experts):
                        stream_index = i % stream_count
                        stream = moe_stream_pool[stream_index]
                        with mx.stream(stream):
                            for k in range(K):
                                mask = indices[:, k] == i
                                # Avoid host sync from mask.any() so streams can overlap.
                                stream_outputs[stream_index][mask, k] = expert(x[mask])
                    expert_outputs = _reduce_stream_outputs(
                        stream_outputs,
                        moe_stream_reduce or "serial",
                    )

            # 3. Combine
            if expert_outputs is not None:
                if is_glm:
                    y = moe.moe_combine_no_fma(expert_outputs, gate_weights)
                elif is_gpt_oss:
                    # GPT-OSS: match MLX's dtype promotion and sum order.
                    if gate_weights.dtype == mx.float32:
                        y = moe.moe_combine_fp32(expert_outputs, gate_weights)
                    else:
                        y = moe.moe_combine_exact(expert_outputs, gate_weights)
                elif use_exact_combine:
                    # Qwen3: match MLX dtype promotion if weights are float32.
                    if gate_weights.dtype == mx.float32:
                        y = moe.moe_combine_fp32(expert_outputs, gate_weights)
                    else:
                        y = moe.moe_combine_exact(expert_outputs, gate_weights)
                else:
                    # Fused Combine: weighted sum of expert outputs in one kernel
                    y = moe.moe_combine(expert_outputs, gate_weights)

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
