"""Profile DeltaNet decode hotspots on Qwen3.5.

SHA-1411: Instruments the DeltaNet forward pass to measure per-op time
distribution, Metal dispatch counts, and identify fusion targets.

Usage:
    python benchmarks/profile_deltanet_decode.py \
        /Volumes/VIXinSSD/models/Qwen3.5-35B-A3B-MLX-4bit \
        --decode-tokens 50 --warmup 3
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Timing infrastructure
# ---------------------------------------------------------------------------
@dataclass
class OpTiming:
    name: str
    total_ms: float = 0.0
    calls: int = 0
    samples: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return self.total_ms / max(self.calls, 1)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0


class TimingCollector:
    def __init__(self):
        self.ops: dict[str, OpTiming] = {}
        self.layer_timings: dict[str, list[float]] = defaultdict(list)

    def record(self, name: str, elapsed_ms: float):
        if name not in self.ops:
            self.ops[name] = OpTiming(name=name)
        op = self.ops[name]
        op.total_ms += elapsed_ms
        op.calls += 1
        op.samples.append(elapsed_ms)

    def record_layer(self, layer_type: str, elapsed_ms: float):
        self.layer_timings[layer_type].append(elapsed_ms)

    def report(self) -> dict:
        ops = sorted(self.ops.values(), key=lambda o: o.total_ms, reverse=True)
        total = sum(o.total_ms for o in ops) or 1.0
        return {
            "ops": [
                {
                    "name": o.name,
                    "total_ms": round(o.total_ms, 3),
                    "calls": o.calls,
                    "mean_ms": round(o.mean_ms, 4),
                    "median_ms": round(o.median_ms, 4),
                    "pct": round(100 * o.total_ms / total, 1),
                }
                for o in ops
            ],
            "layer_timings": {
                k: {
                    "total_ms": round(sum(v), 3),
                    "calls": len(v),
                    "mean_ms": round(sum(v) / len(v), 4) if v else 0,
                    "median_ms": round(statistics.median(v), 4) if v else 0,
                }
                for k, v in self.layer_timings.items()
            },
        }


def _timed(collector, name, fn, *args, **kwargs):
    """Execute fn, synchronize GPU, and record timing."""
    mx.synchronize()
    t0 = time.perf_counter_ns()
    result = fn(*args, **kwargs)
    # Force evaluation
    if isinstance(result, tuple):
        mx.eval(*[r for r in result if isinstance(r, mx.array)])
    elif isinstance(result, list):
        mx.eval(*[r for r in result if isinstance(r, mx.array)])
    elif isinstance(result, mx.array):
        mx.eval(result)
    mx.synchronize()
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    collector.record(name, elapsed_ms)
    return result


# ---------------------------------------------------------------------------
# Profiled DeltaNet forward (qwen3_5.GatedDeltaNet variant)
# ---------------------------------------------------------------------------
def profiled_deltanet_forward(mod, inputs, mask, cache, layer_idx, collector):
    """Manually execute GatedDeltaNet ops with per-op timing.

    Handles qwen3_5.GatedDeltaNet which has separate projections:
      in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
    """
    from mlx_lm.models.gated_delta import gated_delta_update

    t0_layer = time.perf_counter_ns()
    B, S, _ = inputs.shape
    tag = f"deltanet_L{layer_idx}"

    # Op 1: Input projection qkv
    qkv = _timed(collector, f"{tag}/in_proj_qkv", mod.in_proj_qkv, inputs)

    # Op 2: Input projection z
    def _proj_z():
        return mod.in_proj_z(inputs).reshape(B, S, mod.num_v_heads, mod.head_v_dim)
    z = _timed(collector, f"{tag}/in_proj_z", _proj_z)

    # Op 3: Input projection b
    b = _timed(collector, f"{tag}/in_proj_b", mod.in_proj_b, inputs)

    # Op 4: Input projection a
    a = _timed(collector, f"{tag}/in_proj_a", mod.in_proj_a, inputs)

    # Op 5: Conv state setup + concat
    def _conv_setup():
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, mod.conv_kernel_size - 1, mod.conv_dim), dtype=inputs.dtype
            )
        qkv_m = mx.where(mask[..., None], qkv, 0) if mask is not None else qkv
        return mx.concatenate([conv_state, qkv_m], axis=1)
    conv_input = _timed(collector, f"{tag}/conv_state_setup", _conv_setup)

    # Op 6: Conv cache update
    def _conv_cache():
        if cache is not None:
            cache[0] = conv_input[:, -(mod.conv_kernel_size - 1):]
            return cache[0]
        return conv_input
    _timed(collector, f"{tag}/conv_cache_update", _conv_cache)

    # Op 7: Conv1d
    conv_out_raw = _timed(collector, f"{tag}/conv1d", mod.conv1d, conv_input)

    # Op 8: SiLU activation
    conv_out = _timed(collector, f"{tag}/silu", nn.silu, conv_out_raw)

    # Op 9: Split + reshape q,k,v from conv output
    def _split_reshape():
        return [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [mod.key_dim, 2 * mod.key_dim], -1),
                [mod.num_k_heads, mod.num_k_heads, mod.num_v_heads],
                [mod.head_k_dim, mod.head_k_dim, mod.head_v_dim],
            )
        ]
    q, k, v = _timed(collector, f"{tag}/split_reshape", _split_reshape)

    # Op 10: RMS norm q, k + scaling
    def _rms_norm_qk():
        inv_scale = k.shape[-1] ** -0.5
        q_n = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k_n = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        return q_n, k_n
    q2, k2 = _timed(collector, f"{tag}/rms_norm_qk", _rms_norm_qk)

    # Op 11: Gated delta update (core recurrence)
    state = cache[1] if cache else None
    out, state = _timed(
        collector, f"{tag}/gated_delta_update",
        gated_delta_update,
        q2, k2, v, a, b,
        mod.A_log, mod.dt_bias,
        state, mask,
        use_kernel=not mod.training,
    )

    # State cache update (no sync needed - just pointer assignment)
    if cache is not None:
        cache[1] = state

    # Op 12: Gated RMSNorm (norm + silu gate)
    out_normed = _timed(collector, f"{tag}/gated_rmsnorm", mod.norm, out, z)

    # Op 13: Output projection
    result = _timed(collector, f"{tag}/out_proj",
                     mod.out_proj, out_normed.reshape(B, S, -1))

    mx.synchronize()
    elapsed_ms = (time.perf_counter_ns() - t0_layer) / 1e6
    collector.record_layer("deltanet", elapsed_ms)
    return result


# ---------------------------------------------------------------------------
# Profiled full model forward pass
# ---------------------------------------------------------------------------
def profiled_model_forward(model, inputs, cache, collector):
    """Run a full forward pass with per-layer profiling."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    # Navigate to inner text model: model.language_model.model
    inner = model.language_model.model

    hidden_states = inner.embed_tokens(inputs)

    ssm_idx = getattr(inner, "ssm_idx", 0)
    fa_idx = getattr(inner, "fa_idx", 3)

    fa_mask = create_attention_mask(hidden_states, cache[fa_idx] if cache else None)
    ssm_mask = create_ssm_mask(hidden_states, cache[ssm_idx] if cache else None)

    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        is_linear = getattr(layer, "is_linear", False)
        cur_mask = ssm_mask if is_linear else fa_mask

        if is_linear:
            # Input layernorm
            normed = _timed(collector, f"layer_L{i}/input_layernorm",
                             layer.input_layernorm, hidden_states)

            # DeltaNet with per-op profiling
            r = profiled_deltanet_forward(
                layer.linear_attn, normed, cur_mask, c, i, collector
            )

            # Residual add
            h = _timed(collector, f"layer_L{i}/residual_add_1",
                        lambda: hidden_states + r)

            # Post-attention layernorm
            normed2 = _timed(collector, f"layer_L{i}/post_attn_layernorm",
                              layer.post_attention_layernorm, h)

            # MoE
            mx.synchronize()
            t0_moe = time.perf_counter_ns()
            mlp_out = layer.mlp(normed2)
            mx.eval(mlp_out)
            mx.synchronize()
            moe_ms = (time.perf_counter_ns() - t0_moe) / 1e6
            collector.record(f"moe_L{i}/total", moe_ms)
            collector.record_layer("moe", moe_ms)

            # Residual add
            hidden_states = _timed(
                collector, f"layer_L{i}/residual_add_2",
                lambda: h + mlp_out
            )

        else:
            # Attention + MoE layer - time as a whole
            mx.synchronize()
            t0_attn = time.perf_counter_ns()
            normed = layer.input_layernorm(hidden_states)
            r = layer.self_attn(normed, cur_mask, c)
            h = hidden_states + r
            mx.eval(h)
            mx.synchronize()
            attn_ms = (time.perf_counter_ns() - t0_attn) / 1e6
            collector.record(f"attention_L{i}/total", attn_ms)
            collector.record_layer("attention", attn_ms)

            # MoE for attention layer
            mx.synchronize()
            t0_moe = time.perf_counter_ns()
            normed2 = layer.post_attention_layernorm(h)
            mlp_out = layer.mlp(normed2)
            hidden_states = h + mlp_out
            mx.eval(hidden_states)
            mx.synchronize()
            moe_ms = (time.perf_counter_ns() - t0_moe) / 1e6
            collector.record(f"moe_L{i}/total", moe_ms)
            collector.record_layer("moe", moe_ms)

    hidden_states = inner.norm(hidden_states)

    # LM head
    if hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head(hidden_states)
    else:
        return inner.embed_tokens.as_linear(hidden_states)


# ---------------------------------------------------------------------------
# Profiled generation loop
# ---------------------------------------------------------------------------
def profiled_generate(model, tokenizer, prompt, max_tokens, collector):
    """Manual generation loop with profiled forward passes."""
    tokens = tokenizer.encode(prompt)

    cache = model.make_cache()

    # Prefill (not profiled in detail)
    print(f"  Prefill ({len(tokens)} tokens)...")
    input_ids = mx.array([tokens])
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    next_token = int(logits[:, -1, :].argmax(axis=-1).item())
    generated = [next_token]

    # Decode with profiling
    print(f"  Decode ({max_tokens} tokens)...")
    for step in range(max_tokens):
        input_ids = mx.array([[next_token]])
        logits = profiled_model_forward(model, input_ids, cache, collector)
        mx.eval(logits)
        next_token = int(logits[:, -1, :].argmax(axis=-1).item())
        generated.append(next_token)

        if (step + 1) % 10 == 0:
            print(f"    {step + 1}/{max_tokens} tokens...")

        if next_token == tokenizer.eos_token_id:
            print(f"    EOS at step {step + 1}")
            break

    text = tokenizer.decode(generated)
    return text, generated


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_deltanet_ops(report: dict) -> dict:
    categories = defaultdict(lambda: {"total_ms": 0.0, "calls": 0, "samples": []})

    for op in report["ops"]:
        name = op["name"]
        if not name.startswith("deltanet_L"):
            continue
        cat = name.split("/", 1)[1] if "/" in name else name
        categories[cat]["total_ms"] += op["total_ms"]
        categories[cat]["calls"] += op["calls"]
        categories[cat]["samples"].extend([op["mean_ms"]])

    total = sum(c["total_ms"] for c in categories.values()) or 1.0
    result = []
    for cat, data in sorted(categories.items(), key=lambda x: -x[1]["total_ms"]):
        result.append({
            "op": cat,
            "total_ms": round(data["total_ms"], 3),
            "calls": data["calls"],
            "mean_ms": round(data["total_ms"] / max(data["calls"], 1), 4),
            "pct_of_deltanet": round(100 * data["total_ms"] / total, 1),
        })
    return {"deltanet_op_breakdown": result, "total_deltanet_ms": round(total, 3)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Profile DeltaNet decode hotspots")
    parser.add_argument("model_path", help="Path to Qwen3.5 MLX model")
    parser.add_argument("--decode-tokens", type=int, default=50,
                        help="Number of decode tokens to profile")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup decode iterations (not profiled)")
    parser.add_argument("--prompt", type=str,
                        default="Explain the key differences between TCP and UDP protocols.",
                        help="Prompt to use for generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to file")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  DeltaNet Decode Profiler (SHA-1411)")
    print(f"  Model: {args.model_path}")
    print(f"  Tokens: {args.decode_tokens} (+ {args.warmup} warmup)")
    print(f"{'='*70}\n")

    import mlx_lm
    print("Loading model...")
    model, tokenizer = mlx_lm.load(args.model_path)[:2]

    layers = model.layers
    dn_count = sum(1 for l in layers if getattr(l, "is_linear", False))
    attn_count = sum(1 for l in layers if not getattr(l, "is_linear", True))
    moe_count = sum(1 for l in layers if hasattr(l.mlp, "switch_mlp"))
    print(f"  Layers: {dn_count} DeltaNet, {attn_count} Attention, {moe_count} MoE")

    # Check GatedDeltaNet variant
    dn_layer = next(l for l in layers if getattr(l, "is_linear", False))
    dn_mod = dn_layer.linear_attn
    has_separate_projs = hasattr(dn_mod, "in_proj_qkv")
    print(f"  GatedDeltaNet variant: {'separate projs (qwen3_5)' if has_separate_projs else 'combined (qwen3_next)'}")
    print(f"  Conv dim: {dn_mod.conv_dim}, Key dim: {dn_mod.key_dim}, "
          f"Value dim: {dn_mod.value_dim}")
    print(f"  Heads: Hk={dn_mod.num_k_heads}, Hv={dn_mod.num_v_heads}, "
          f"Dk={dn_mod.head_k_dim}, Dv={dn_mod.head_v_dim}")

    # Warmup
    print(f"\nWarming up ({args.warmup} generations of 5 tokens)...")
    for i in range(args.warmup):
        mlx_lm.generate(model, tokenizer, prompt="Hello", max_tokens=5)
        mx.synchronize()
        print(f"  warmup {i+1}/{args.warmup} done")
    gc.collect()
    mx.clear_cache()

    # Profile decode
    collector = TimingCollector()
    print(f"\nProfiling {args.decode_tokens} decode tokens...")
    text, gen_tokens = profiled_generate(
        model, tokenizer, args.prompt, args.decode_tokens, collector
    )
    print(f"\n  Generated {len(gen_tokens)} tokens")

    # Report
    report = collector.report()
    agg = aggregate_deltanet_ops(report)

    print(f"\n{'='*70}")
    print(f"  PROFILING RESULTS")
    print(f"{'='*70}\n")

    # 1. Layer type comparison
    print("== Layer Type Time Distribution ==\n")
    lt = report["layer_timings"]
    total_layer_ms = sum(d["total_ms"] for d in lt.values()) or 1.0
    for ltype in ["deltanet", "attention", "moe"]:
        if ltype in lt:
            d = lt[ltype]
            pct = 100 * d["total_ms"] / total_layer_ms
            print(f"  {ltype:12s}: {d['total_ms']:8.1f} ms total "
                  f"({d['calls']:4d} calls, {d['mean_ms']:6.3f} ms/call, "
                  f"{pct:5.1f}%)")
    print()

    # 2. DeltaNet sub-op breakdown
    print("== DeltaNet Sub-Operation Breakdown ==\n")
    print(f"  Total DeltaNet time: {agg['total_deltanet_ms']:.1f} ms\n")
    print(f"  {'Operation':<25s} {'Total (ms)':>10s} {'Calls':>6s} "
          f"{'Mean (ms)':>10s} {'% of DN':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*6} {'-'*10} {'-'*8}")
    for op in agg["deltanet_op_breakdown"]:
        print(f"  {op['op']:<25s} {op['total_ms']:10.1f} {op['calls']:6d} "
              f"{op['mean_ms']:10.4f} {op['pct_of_deltanet']:7.1f}%")

    # 3. Fusion target ranking
    print(f"\n== Fusion Target Ranking ==\n")
    fusible = [
        ("conv1d + silu", ["conv1d", "silu"]),
        ("gated_rmsnorm (norm+silu gate)", ["gated_rmsnorm"]),
        ("input projections (qkv+z+b+a)", ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]),
        ("conv state setup + cache", ["conv_state_setup", "conv_cache_update"]),
        ("qk rms_norm", ["rms_norm_qk"]),
        ("split + reshape", ["split_reshape"]),
    ]
    op_lookup = {op["op"]: op for op in agg["deltanet_op_breakdown"]}
    targets = []
    for label, ops in fusible:
        total_ms = sum(op_lookup.get(o, {}).get("total_ms", 0) for o in ops)
        targets.append((label, total_ms))
    targets.sort(key=lambda x: -x[1])

    dn_total = agg["total_deltanet_ms"] or 1.0
    for label, ms in targets:
        pct = 100 * ms / dn_total
        est_savings_ms = ms * 0.3
        print(f"  {label:<40s}: {ms:7.1f} ms ({pct:5.1f}% of DN) "
              f"~{est_savings_ms:.1f} ms saveable")

    # 4. Gated delta kernel assessment
    print(f"\n== Gated Delta Kernel Assessment ==\n")
    gd = op_lookup.get("gated_delta_update", {})
    if gd:
        gd_pct = gd.get("pct_of_deltanet", 0)
        print(f"  gated_delta_update: {gd.get('total_ms', 0):.1f} ms "
              f"({gd_pct:.1f}% of DeltaNet time)")
        print(f"  Mean per call: {gd.get('mean_ms', 0):.4f} ms")
        if gd_pct > 30:
            print(f"  -> DOMINANT: Worth optimizing the kernel itself (SHA-1415)")
        elif gd_pct > 15:
            print(f"  -> SIGNIFICANT: Kernel optimization may help")
        else:
            print(f"  -> MINOR: Focus on other fusions first")

    # 5. Overall opportunity
    print(f"\n== Overall Opportunity ==\n")
    dn_pct = 100 * lt.get("deltanet", {}).get("total_ms", 0) / total_layer_ms
    moe_pct = 100 * lt.get("moe", {}).get("total_ms", 0) / total_layer_ms
    attn_pct = 100 * lt.get("attention", {}).get("total_ms", 0) / total_layer_ms
    print(f"  DeltaNet is {dn_pct:.1f}% of total layer time ({dn_count} layers)")
    print(f"  MoE is {moe_pct:.1f}% of total layer time ({moe_count} layers)")
    print(f"  Attention is {attn_pct:.1f}% of total layer time ({attn_count} layers)")

    if dn_pct > 0:
        savings_pct = dn_pct * 0.30
        overall_speedup = 100 / (100 - savings_pct)
        print(f"\n  If we save ~30% of DeltaNet time via fusions:")
        print(f"    {savings_pct:.1f}% of total time saved -> "
              f"{overall_speedup:.1%} overall speedup")

    n_decode = max(len(gen_tokens) - 1, 1)
    total_decode_ms = sum(d["total_ms"] for d in lt.values())
    per_token_ms = total_decode_ms / n_decode
    est_tps = 1000 / per_token_ms if per_token_ms > 0 else 0
    print(f"\n  Per-token decode (profiled): {per_token_ms:.1f} ms ({est_tps:.1f} tok/s)")
    print(f"  Note: profiling overhead inflates this by ~2-5x vs unprofiled")

    # Save JSON
    full_report = {
        "model": args.model_path,
        "decode_tokens": args.decode_tokens,
        "warmup_iters": args.warmup,
        "layer_counts": {
            "deltanet": dn_count,
            "attention": attn_count,
            "moe": moe_count,
        },
        "dims": {
            "conv_dim": dn_mod.conv_dim,
            "key_dim": dn_mod.key_dim,
            "value_dim": dn_mod.value_dim,
            "num_k_heads": dn_mod.num_k_heads,
            "num_v_heads": dn_mod.num_v_heads,
            "head_k_dim": dn_mod.head_k_dim,
            "head_v_dim": dn_mod.head_v_dim,
            "conv_kernel_size": dn_mod.conv_kernel_size,
        },
        "layer_timings": report["layer_timings"],
        "deltanet_breakdown": agg,
        "all_ops": report["ops"],
        "generated_text_preview": text[:200] if text else "",
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(full_report, f, indent=2)
        print(f"\n  Report saved to: {args.output}")

    return full_report


if __name__ == "__main__":
    main()
