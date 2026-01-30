"""zmlx.patch — Module-level patching for MLX models.

Usage::

    import zmlx
    model = zmlx.patch.patch(model)                       # safe default (inference)
    model = zmlx.patch.patch(model, mode="training")      # training preset
    model = zmlx.patch.patch(model, patterns=ALL_PATTERNS)  # explicit full set
    model = zmlx.patch.smart_patch(model, sample)         # auto-benchmark

Presets::

    FUSED_ACTIVATIONS — SwiGLU/GeGLU/MoE fusions only (safe default, no regressions).
        MoE models get 1.3–1.6x speedup; dense models neutral.
    TRAINING_RECOMMENDED — activations + norms + fused residual (best for training).
    ALL_PATTERNS — all 7 patterns including norms and softmax.
        WARNING: causes 3–5% decode regression on ALL tested models (dense and MoE).
        Norm/softmax kernels are slower than MLX built-ins for inference.

Note on inference: LLM decode is memory-bandwidth-bound — 95%+ of time is spent
reading quantized weights through matmuls.  MLX's built-in ``mx.fast.rms_norm``,
``mx.fast.rope``, and ``mx.fast.scaled_dot_product_attention`` are already highly
optimized.  Custom norm/softmax kernels add dispatch overhead that exceeds any
memory savings.  The fused activation patterns (SwiGLU/GeGLU/MoE) are the only
patterns that help inference — MoE routing fusion is the killer feature (1.3–1.6x
on MoE models) because MLX lacks a fast path for multi-step expert dispatch.

Use ``smart_patch`` to automatically benchmark and keep only beneficial patterns.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import mlx.nn as nn

from ._registry import _ensure_loaded, get_all_patterns, get_pattern, list_patterns
from ._traversal import apply_patterns
from ._types import PatchConfig, PatchResult

# ---------------------------------------------------------------------------
# Presets — curated pattern lists for common scenarios
# ---------------------------------------------------------------------------

#: Fused activation patterns only.  Safe for inference: these replace multi-op
#: activation sequences (split → silu → mul) with a single fused Metal kernel.
#: Neutral-to-positive on throughput, no regressions observed.
FUSED_ACTIVATIONS: list[str] = ["swiglu_mlp", "geglu_mlp", "moe_mlp"]

#: Recommended for training workloads.  Includes fused activations plus norm
#: replacements and residual-norm fusion, which benefit training (weight
#: gradients, fused loss).  Not recommended for inference — MLX's built-in
#: ``mx.fast.rms_norm`` is faster than custom norm kernels.
TRAINING_RECOMMENDED: list[str] = [
    "swiglu_mlp",
    "geglu_mlp",
    "rmsnorm",
    "layernorm",
    "residual_norm",
]

#: All 7 patterns.  **WARNING**: causes 3–5% decode regression on ALL tested
#: models (dense 8B/32B and MoE 30B).  Norm and softmax kernels are slower
#: than MLX's built-in ``mx.fast.rms_norm`` / ``mx.softmax`` at inference.
#: Only use this preset if you have benchmarked it on your specific workload.
ALL_PATTERNS: list[str] = [
    "swiglu_mlp",
    "geglu_mlp",
    "moe_mlp",
    "rmsnorm",
    "layernorm",
    "softmax",
    "residual_norm",
]


def patch(
    model: nn.Module,
    *,
    mode: str | None = None,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
    compute_dtype: str = "float32",
    threadgroup: int | str = 256,
    verbose: bool = False,
) -> nn.Module:
    """Patch an MLX model to use fused ZMLX Metal kernels.

    Walks the module tree and replaces matching submodules with
    ZMLX-backed drop-in replacements.

    Args:
        model: The nn.Module to patch (modified in place and returned).
        mode: Shorthand for common workloads.  ``"inference"`` (default)
            selects :data:`FUSED_ACTIVATIONS` — safe, never regresses, and
            gives 1.3–1.6x on MoE models.  ``"training"`` selects
            :data:`TRAINING_RECOMMENDED` — adds norm fusions that benefit
            backward passes.  Ignored if ``patterns`` is provided explicitly.
        patterns: Explicit list of pattern names.  Overrides ``mode`` when
            both are given.  ``None`` falls through to ``mode`` selection.
        exclude: Pattern names to skip.
        compute_dtype: Compute dtype name (e.g. "float32", "float16").
        threadgroup: Default threadgroup size for fused kernels, or ``"auto"``
            to autotune on first invocation.
        verbose: Print each replacement as it happens.

    Returns:
        The same model, modified in place, with a ``_zmlx_patch_result``
        attribute containing a :class:`PatchResult`.

    Examples::

        patch(model)                    # inference (safe default)
        patch(model, mode="training")   # training preset
        patch(model, patterns=["swiglu_mlp", "moe_mlp"])  # explicit

    .. versionadded:: 0.6.0
        ``mode`` parameter for workload-aware preset selection.
    .. versionchanged:: 0.5.0
        Default changed from all patterns to ``FUSED_ACTIVATIONS`` to avoid
        decode regressions on inference workloads.
    """
    _ensure_loaded()

    _MODES = {
        "inference": FUSED_ACTIVATIONS,
        "training": TRAINING_RECOMMENDED,
    }

    config = PatchConfig(
        compute_dtype=compute_dtype,
        threadgroup=threadgroup,
        verbose=verbose,
    )

    # Resolve patterns: explicit patterns > mode > default (inference)
    if patterns is not None:
        selected = [get_pattern(name) for name in patterns]
    elif mode is not None:
        if mode not in _MODES:
            raise ValueError(
                f"Unknown mode {mode!r}. Expected one of: {', '.join(_MODES)}"
            )
        selected = [get_pattern(name) for name in _MODES[mode]]
    else:
        selected = [get_pattern(name) for name in FUSED_ACTIVATIONS]

    if exclude:
        exclude_set = set(exclude)
        selected = [p for p in selected if p.name not in exclude_set]

    if verbose:
        print(f"[zmlx.patch] Applying {len(selected)} patterns: {[p.name for p in selected]}")

    result = apply_patterns(model, selected, config)

    if verbose:
        print(result.summary())

    model._zmlx_patch_result = result  # type: ignore[attr-defined]
    return model


def unpatch(model: nn.Module) -> nn.Module:
    """Remove ZMLX patches from a model.

    Note: Only modules that stored their original call can be unpatched.
    Module replacements (like RMSNorm -> ZMLXRMSNorm) cannot be automatically
    reversed — reload the model instead.
    """
    children: dict[str, Any] = {}
    if hasattr(model, "children") and callable(model.children):
        children = dict(model.children())

    for _name, child in children.items():
        if hasattr(child, "_zmlx_original_call") and child._zmlx_original_call is not None:
            child.__call__ = child._zmlx_original_call
            del child._zmlx_original_call
        if hasattr(child, "_zmlx_original_softmax"):
            child.softmax = child._zmlx_original_softmax
            del child._zmlx_original_softmax
        if isinstance(child, nn.Module):
            unpatch(child)

    if hasattr(model, "_zmlx_patch_result"):
        del model._zmlx_patch_result
    return model


# ---------------------------------------------------------------------------
# Smart patching — auto-benchmark each pattern
# ---------------------------------------------------------------------------

# Patterns that wrap __call__ and can be cleanly reverted via unpatch().
# Module-replacement patterns (rmsnorm, layernorm) swap the nn.Module and
# cannot be cheaply reverted — we only test those if the user explicitly
# includes them.
_REVERTIBLE_PATTERNS = {"swiglu_mlp", "geglu_mlp", "residual_norm", "softmax", "moe_mlp"}


def _time_forward(
    model: nn.Module,
    sample: Any,
    warmup: int,
    iters: int,
    forward_fn: Callable[..., Any] | None,
) -> float:
    """Time a model forward pass, return median ms."""
    import mlx.core as mx

    def _run() -> None:
        if forward_fn is not None:
            out = forward_fn(model, sample)
        else:
            out = model(sample)
        if isinstance(out, (list, tuple)):
            mx.eval(*out)
        else:
            mx.eval(out)

    # Warmup
    for _ in range(warmup):
        _run()
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()

    # Timed runs
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _run()
        if callable(sync):
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]


def smart_patch(
    model: nn.Module,
    sample: Any,
    *,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
    threshold: float = 0.995,
    warmup: int = 3,
    iters: int = 5,
    compute_dtype: str = "float32",
    threadgroup: int | str = 256,
    verbose: bool = True,
    forward_fn: Callable[..., Any] | None = None,
) -> nn.Module:
    """Intelligently patch a model, keeping only patterns that help.

    Applies each candidate pattern, benchmarks the model's forward pass,
    and reverts any pattern that makes things slower. Only patterns whose
    call-wrapping can be cleanly reverted are tested individually; module-
    replacement patterns (rmsnorm, layernorm) are skipped unless explicitly
    listed in ``patterns``.

    Args:
        model: The model to patch (modified in place).
        sample: A sample input for benchmarking (e.g. tokenized prompt IDs).
        patterns: Candidate patterns. ``None`` defaults to revertible patterns
            (swiglu_mlp, geglu_mlp, residual_norm, softmax).
        exclude: Patterns to skip.
        threshold: Keep a pattern if post-patch speed >= ``threshold`` * baseline.
            Default 0.995 (keep if within 0.5% of baseline).
        warmup: Forward-pass warmup iterations per benchmark.
        iters: Timed forward-pass iterations per benchmark.
        compute_dtype: Metal compute dtype.
        threadgroup: Threadgroup size or ``"auto"``.
        verbose: Print benchmark results as each pattern is tested.
        forward_fn: Custom forward function ``(model, sample) -> output``.
            If None, calls ``model(sample)`` directly.

    Returns:
        The patched model with a ``_zmlx_patch_result`` attribute that
        includes per-pattern speedup data in ``result.benchmarks``.
    """
    _ensure_loaded()

    config = PatchConfig(
        compute_dtype=compute_dtype,
        threadgroup=threadgroup,
        verbose=False,  # We handle printing ourselves
    )

    # Resolve candidate patterns
    if patterns is not None:
        candidates = [get_pattern(name) for name in patterns]
    else:
        # Default: only test cleanly revertible patterns
        all_patterns = get_all_patterns()
        candidates = [p for name, p in all_patterns.items() if name in _REVERTIBLE_PATTERNS]

    if exclude:
        exclude_set = set(exclude)
        candidates = [p for p in candidates if p.name not in exclude_set]

    if verbose:
        print(f"[smart_patch] Benchmarking {len(candidates)} patterns: "
              f"{[p.name for p in candidates]}")

    # Baseline
    baseline_ms = _time_forward(model, sample, warmup, iters, forward_fn)
    if verbose:
        print(f"[smart_patch] Baseline: {baseline_ms:.2f} ms")

    result = PatchResult()
    kept: list[str] = []
    reverted: list[str] = []
    current_ms = baseline_ms

    for pattern in candidates:
        # Apply this single pattern
        single_result = apply_patterns(model, [pattern], config)

        if single_result.patched_count == 0:
            if verbose:
                print(f"  {pattern.name}: no matches (skipped)")
            continue

        # Benchmark with this pattern applied
        patched_ms = _time_forward(model, sample, warmup, iters, forward_fn)
        speedup = current_ms / patched_ms if patched_ms > 0 else 1.0

        if speedup >= threshold:
            # Keep it
            kept.append(pattern.name)
            result.patched_count += single_result.patched_count
            for name, count in single_result.pattern_counts.items():
                result.pattern_counts[name] = (
                    result.pattern_counts.get(name, 0) + count
                )
            result.benchmarks[pattern.name] = round(speedup, 4)
            current_ms = patched_ms
            if verbose:
                print(f"  {pattern.name}: {speedup:.3f}x -> KEEP "
                      f"({single_result.patched_count} modules)")
        else:
            # Revert it
            unpatch(model)
            # Re-apply all previously kept patterns
            for kept_name in kept:
                apply_patterns(model, [get_pattern(kept_name)], config)
            reverted.append(pattern.name)
            result.benchmarks[pattern.name] = round(speedup, 4)
            if verbose:
                print(f"  {pattern.name}: {speedup:.3f}x -> REVERT "
                      f"(below {threshold:.3f} threshold)")

    if verbose:
        total_speedup = baseline_ms / current_ms if current_ms > 0 else 1.0
        print(f"[smart_patch] Result: {baseline_ms:.2f} -> {current_ms:.2f} ms "
              f"({total_speedup:.3f}x)")
        print(f"  Kept: {kept or '(none)'}")
        if reverted:
            print(f"  Reverted: {reverted}")

    model._zmlx_patch_result = result  # type: ignore[attr-defined]
    return model


__all__ = [
    "patch",
    "unpatch",
    "smart_patch",
    "list_patterns",
    "ALL_PATTERNS",
    "FUSED_ACTIVATIONS",
    "TRAINING_RECOMMENDED",
    "PatchConfig",
    "PatchResult",
]
