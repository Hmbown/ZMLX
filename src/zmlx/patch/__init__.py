"""zmlx.patch — Module-level patching for MLX models.

Usage::

    import zmlx
    model = zmlx.patch.patch(model)                       # inference default
    model = zmlx.patch.patch(model, mode="training")      # training preset
    model = zmlx.patch.patch(model, patterns=ALL_PATTERNS)  # explicit full set
    model = zmlx.patch.smart_patch(model, sample)         # auto-benchmark

Presets::

    FUSED_ACTIVATIONS — SwiGLU/GeGLU/MoE fusions only (default for inference).
        Best results on MoE when fused SwiGLU is available; on stock MLX use
        ``smart_patch`` to avoid regressions.
    TRAINING_RECOMMENDED — activations + norms + fused residual (best for training).
    ALL_PATTERNS — all 7 patterns including norms and softmax.
        WARNING: can regress on inference. Benchmark before enabling.

Note on inference: LLM decode is bandwidth-bound and MLX's built-in
``mx.fast.rms_norm``, ``mx.fast.rope``, and
``mx.fast.scaled_dot_product_attention`` are highly optimized. Custom
norm/softmax kernels often add dispatch overhead. Fused activations
(SwiGLU/GeGLU/MoE) are the most likely to help, with MoE gains strongest
when fused SwiGLU is available. On stock MLX, gating+combine can be neutral
to negative — prefer ``smart_patch`` for MoE models.

Use ``smart_patch`` to automatically benchmark and keep only beneficial patterns.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import mlx.nn as nn

from ._registry import _ensure_loaded, get_all_patterns, get_pattern, list_patterns
from ._traversal import apply_patterns
from ._types import PatchConfig, PatchResult

# ---------------------------------------------------------------------------
# Model-aware safety: auto-exclude patterns with known fidelity issues
# ---------------------------------------------------------------------------

def _model_family(model: nn.Module) -> str:
    """Best-effort model family detection from class/module names."""
    candidates: list[str] = []
    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue
        candidates.append(type(obj).__module__ or "")
        candidates.append(type(obj).__name__)

    combined = " ".join(candidates).lower()
    if "lfm" in combined:
        return "lfm"
    if "qwen" in combined:
        return "qwen"
    if "gpt_oss" in combined or "gptoss" in combined:
        return "gpt_oss"
    if "nemotron" in combined:
        return "nemotron"
    if "llama" in combined:
        return "llama"
    if "mixtral" in combined:
        return "mixtral"
    if "deepseek" in combined:
        return "deepseek"
    if "glm" in combined:
        return "glm"
    return "unknown"


# Patterns known to break token fidelity per model family.
# Validated with `python -m zmlx.validate` (200-token greedy).
_FIDELITY_EXCLUDES: dict[str, set[str]] = {
    "qwen": {"swiglu_mlp", "residual_norm"},
    "gpt_oss": {"residual_norm"},
    "mixtral": {"moe_mlp"},
    # GLM: moe_mlp/swiglu_mlp are fidelity-safe (validated) as of 2026-02-04.
    "glm": set(),
}

# Patterns known to regress performance per model family.
_PERF_EXCLUDES: dict[str, set[str]] = {
    "qwen": {"moe_mlp"},
    "glm": {"moe_mlp", "swiglu_mlp"},
}


def _apply_family_excludes(
    selected: list[Any],
    *,
    family: str,
    allow_perf_risk: bool,
) -> list[Any]:
    excludes = set(_FIDELITY_EXCLUDES.get(family, set()))
    if not allow_perf_risk:
        excludes |= _PERF_EXCLUDES.get(family, set())
    if not excludes:
        return selected
    return [p for p in selected if p.name not in excludes]

# ---------------------------------------------------------------------------
# Presets — curated pattern lists for common scenarios
# ---------------------------------------------------------------------------

#: Fused activation patterns only. Default for inference: these replace multi-op
#: activation sequences (split → silu → mul) with a single fused Metal kernel.
#: On stock MLX, MoE gating+combine can regress; use smart_patch to validate.
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

#: All 7 patterns.  **WARNING**: can regress on inference. Norm and softmax
#: kernels are often slower than MLX's built-in ``mx.fast.rms_norm`` /
#: ``mx.softmax``. Only use this preset if you have benchmarked it on your
#: specific workload.
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
    profile: str | None = None,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
    compute_dtype: str = "float32",
    threadgroup: int | str = 256,
    moe_fused_swiglu_max_tokens: int | None = None,
    verbose: bool = False,
) -> nn.Module:
    """Patch an MLX model to use fused ZMLX Metal kernels.

    Walks the module tree and replaces matching submodules with
    ZMLX-backed drop-in replacements.

    Args:
        model: The nn.Module to patch (modified in place and returned).
        mode: Shorthand for common workloads.  ``"inference"`` (default)
            selects :data:`FUSED_ACTIVATIONS` — best results on MoE when
            fused SwiGLU is available; on stock MLX use ``smart_patch`` to
            avoid regressions.  ``"training"`` selects
            :data:`TRAINING_RECOMMENDED` — adds norm fusions that benefit
            backward passes.  Ignored if ``patterns`` is provided explicitly.
        profile: Optional profile name that selects a curated pattern set.
            Overrides ``mode`` when provided (unless ``patterns`` is set).
        patterns: Explicit list of pattern names.  Overrides ``mode`` when
            both are given.  ``None`` falls through to ``mode`` selection.
        exclude: Pattern names to skip.
        compute_dtype: Compute dtype name (e.g. "float32", "float16").
        threadgroup: Default threadgroup size for fused kernels, or ``"auto"``
            to autotune on first invocation.
        moe_fused_swiglu_max_tokens: Override the max token count for the fused
            SwiGLU MoE path. ``None`` uses the built-in default threshold.
        verbose: Print each replacement as it happens.

    Returns:
        The same model, modified in place, with a ``_zmlx_patch_result``
        attribute containing a :class:`PatchResult`.

    Examples::

        patch(model)                    # inference (safe default)
        patch(model, mode="training")   # training preset
        patch(model, patterns=["swiglu_mlp", "moe_mlp"])  # explicit
        patch(model, profile="qwen3")   # Qwen3 minimal (moe only)

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
    _PROFILES = {
        "qwen3": ["moe_mlp"],
        "qwen3-minimal": ["moe_mlp"],
    }

    config = PatchConfig(
        compute_dtype=compute_dtype,
        threadgroup=threadgroup,
        moe_fused_swiglu_max_tokens=moe_fused_swiglu_max_tokens,
        verbose=verbose,
    )

    # Resolve patterns: explicit patterns > profile > mode > default (inference)
    explicit = patterns is not None
    profiled = profile is not None
    if explicit:
        assert patterns is not None  # mypy narrowing
        selected = [get_pattern(name) for name in patterns]
    elif profile is not None:
        if profile not in _PROFILES:
            raise ValueError(
                f"Unknown profile {profile!r}. Expected one of: "
                f"{', '.join(sorted(_PROFILES))}"
            )
        selected = [get_pattern(name) for name in _PROFILES[profile]]
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

    # Model-aware safety: auto-exclude patterns with known fidelity issues.
    # Only when using default pattern selection (no explicit patterns=).
    # Pass patterns=[...] explicitly to override.
    family = _model_family(model)
    fidelity_risks = set(_FIDELITY_EXCLUDES.get(family, set()))
    perf_risks = set(_PERF_EXCLUDES.get(family, set()))
    # Qwen3: moe_mlp is only a perf risk when fused SwiGLU is unavailable.
    # When `mx.gather_qmm_swiglu` exists, moe_mlp is typically decode-positive.
    if family == "qwen" and "moe_mlp" in perf_risks:
        try:
            from ..kernels.fused_moe import has_gather_qmm_swiglu

            if has_gather_qmm_swiglu():
                perf_risks.discard("moe_mlp")
        except Exception:
            # If detection fails, keep the conservative perf exclude.
            pass
    # GLM: moe_mlp is the primary perf win (+7%) but requires gather_qmm_swiglu.
    # swiglu_mlp on shared_experts always regresses (Python dispatch overhead
    # exceeds any activation-fusion benefit for these dense layers).
    if family == "glm":
        try:
            from ..kernels.fused_moe import has_gather_qmm_swiglu

            has_fused = has_gather_qmm_swiglu()
        except Exception:
            has_fused = False

        if has_fused:
            perf_risks.discard("moe_mlp")
    allow_perf_risk = os.environ.get("ZMLX_PATCH_ALLOW_PERF_RISK", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    auto_excludes = set(fidelity_risks)
    if not allow_perf_risk:
        auto_excludes |= perf_risks

    if not explicit and not profiled and auto_excludes:
        before_names = {p.name for p in selected}
        selected = [p for p in selected if p.name not in auto_excludes]
        removed = before_names - {p.name for p in selected}
        if removed:
            note = "known fidelity issues"
            if perf_risks and not allow_perf_risk:
                note = "known fidelity/perf issues"
            print(
                f"[zmlx.patch] {family} model detected — excluded {sorted(removed)} "
                f"({note}). Override with patterns=[...] or set "
                f"ZMLX_PATCH_ALLOW_PERF_RISK=1."
            )
    elif (explicit or profiled) and fidelity_risks:
        requested_risky = {p.name for p in selected} & fidelity_risks
        if requested_risky:
            print(
                f"[zmlx.patch] WARNING: {sorted(requested_risky)} may break token "
                f"fidelity on {family} models. "
                f"Run `python -m zmlx.validate` to verify."
            )

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
# NOTE: residual_norm excluded — slower than baseline (0.94-0.99x) on every
# model tested and breaks fidelity on Qwen/GPT-OSS.  Only use explicitly.
_REVERTIBLE_PATTERNS = {"swiglu_mlp", "geglu_mlp", "softmax", "moe_mlp"}


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
