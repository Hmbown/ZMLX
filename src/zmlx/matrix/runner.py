"""Test runner: wraps validate.py to produce MatrixEntry results."""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone

from .models import ModelInfo, load_catalog
from .schema import MatrixEntry
from .storage import append as ledger_append

_DEFAULT_PROMPT = (
    "Explain the key differences between TCP and UDP protocols, "
    "including their use cases, reliability guarantees, and "
    "performance characteristics. Be thorough and precise."
)


def _get_environment() -> dict:
    """Collect environment metadata."""
    import mlx.core as mx

    import zmlx

    macos_ver = platform.mac_ver()[0] or platform.platform()
    mlx_ver = getattr(mx, "__version__", "unknown")
    zmlx_ver = zmlx.__version__

    # Git commit
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"

    # Custom MLX detection
    custom_mlx = hasattr(mx, "gather_qmm_swiglu")

    # Hardware
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        chip = platform.processor() or "unknown"

    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        mem_gb = mem_bytes // (1024 ** 3)
    except Exception:
        mem_gb = 0

    hardware = f"{chip} {mem_gb}GB" if mem_gb else chip

    return {
        "macos_version": macos_ver,
        "mlx_version": mlx_ver,
        "zmlx_version": zmlx_ver,
        "zmlx_commit": commit,
        "python_version": platform.python_version(),
        "custom_mlx": custom_mlx,
        "hardware": hardware,
    }


def _available_memory_gb() -> float:
    """Return total system memory in GB."""
    try:
        mem_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
        return mem_bytes / (1024 ** 3)
    except Exception:
        return 0.0


def run_one(
    model_id: str,
    *,
    patterns: list[str] | None = None,
    profile: str | None = None,
    runs: int = 3,
    max_tokens: int = 200,
    prompt: str = _DEFAULT_PROMPT,
    notes: str = "",
) -> MatrixEntry:
    """Run a full baseline + patched validation and return a MatrixEntry.

    This loads the model twice (baseline then patched), generates tokens,
    compares fidelity, and collects throughput metrics.
    """
    from zmlx.validate import _bench_config, _compare_tokens, _RunMetrics

    env = _get_environment()

    # Infer model metadata
    from .models import _infer_architecture, _infer_family
    family = _infer_family(model_id)
    architecture = _infer_architecture(model_id, family)

    if patterns is not None and profile is not None:
        raise ValueError("run_one: provide either patterns or profile, not both")

    # Baseline (no patches)
    print(f"\n[matrix] Running: {model_id}")
    if profile is not None:
        patterns_label = f"profile={profile}"
    else:
        patterns_label = patterns if patterns is not None else "(default)"
    print(f"[matrix] Patterns: {patterns_label}")

    baseline = _bench_config(
        model_path=model_id,
        label="Baseline (unpatched)",
        patterns=[],
        profile=None,
        prompt=prompt,
        max_tokens=max_tokens,
        runs=runs,
        gen_kwargs=None,
    )

    # Patched
    patched = _bench_config(
        model_path=model_id,
        label="ZMLX Patched",
        patterns=patterns,
        profile=profile,
        prompt=prompt,
        max_tokens=max_tokens,
        runs=runs,
        gen_kwargs=None,
    )

    # Fidelity comparison (first run)
    b_run = baseline.runs[0] if baseline.runs else _RunMetrics()
    p_run = patched.runs[0] if patched.runs else _RunMetrics()
    match_count, total, _ = _compare_tokens(b_run, p_run)
    fidelity_pass = total > 0 and match_count == total

    # Compute speedups
    baseline_gen_tps = baseline.median_gen_tps
    patched_gen_tps = patched.median_gen_tps
    decode_speedup = (patched_gen_tps / baseline_gen_tps) if baseline_gen_tps > 0 else 0.0

    baseline_prompt_tps = baseline.median_prompt_tps
    patched_prompt_tps = patched.median_prompt_tps
    prefill_change = (patched_prompt_tps / baseline_prompt_tps - 1.0) if baseline_prompt_tps > 0 else 0.0

    # Actual patterns applied
    applied = sorted(patched.pattern_counts.keys()) if patched.pattern_counts else []

    entry = MatrixEntry(
        model_id=model_id,
        model_family=family,
        architecture=architecture,
        patterns_applied=applied,
        hardware=env["hardware"],
        notes=notes,
        macos_version=env["macos_version"],
        mlx_version=env["mlx_version"],
        zmlx_version=env["zmlx_version"],
        zmlx_commit=env["zmlx_commit"],
        python_version=env["python_version"],
        custom_mlx=env["custom_mlx"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        fidelity="PASS" if fidelity_pass else "FAIL",
        fidelity_detail=f"{match_count}/{total} tokens identical",
        modules_patched=patched.patched_count,
        decode_tps_baseline=round(baseline_gen_tps, 2),
        decode_tps_patched=round(patched_gen_tps, 2),
        decode_speedup=round(decode_speedup, 4),
        decode_runs_baseline=[round(r.gen_tps, 2) for r in baseline.runs],
        decode_runs_patched=[round(r.gen_tps, 2) for r in patched.runs],
        prefill_tps_baseline=round(baseline_prompt_tps, 2),
        prefill_tps_patched=round(patched_prompt_tps, 2),
        prefill_change=round(prefill_change, 4),
        peak_mem_baseline_gb=round(baseline.peak_mem_gb, 2),
        peak_mem_patched_gb=round(patched.peak_mem_gb, 2),
        max_tokens=max_tokens,
        gen_tokens=p_run.gen_tokens if patched.runs else 0,
        runs=runs,
    )

    return entry


def run_model(
    model_info: ModelInfo,
    *,
    runs: int = 3,
    max_tokens: int = 200,
    prompt: str = _DEFAULT_PROMPT,
    ledger_path: str | None = None,
) -> MatrixEntry | None:
    """Run validation for a single ModelInfo, skipping if it won't fit in RAM.

    Returns a MatrixEntry on success, or None if skipped.
    """
    available = _available_memory_gb()
    # Need ~2GB headroom beyond model storage
    if available > 0 and model_info.storage_gb > (available - 2.0):
        print(f"[matrix] SKIP {model_info.display_name}: "
              f"{model_info.storage_gb:.0f}GB model > {available:.0f}GB available RAM")
        return None

    patterns = model_info.expected_patterns if model_info.expected_patterns else None
    entry = run_one(
        model_info.model_id,
        patterns=patterns,
        runs=runs,
        max_tokens=max_tokens,
        prompt=prompt,
    )

    ledger_append(entry, ledger_path)

    return entry


def run_all(
    catalog: list[ModelInfo] | None = None,
    *,
    runs: int = 3,
    max_tokens: int = 200,
    prompt: str = _DEFAULT_PROMPT,
    family_filter: str | None = None,
    ledger_path: str | None = None,
) -> list[MatrixEntry]:
    """Run validation across all models that fit in RAM.

    Appends each result to the JSONL ledger as it completes.
    """
    models: list[ModelInfo] = catalog if catalog is not None else load_catalog()

    if family_filter:
        models = [m for m in models if m.family == family_filter]

    results: list[MatrixEntry] = []
    total = len(models)
    for i, model_info in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{total}] {model_info.display_name}")
        print(f"{'='*60}")

        entry = run_model(
            model_info,
            runs=runs,
            max_tokens=max_tokens,
            prompt=prompt,
            ledger_path=ledger_path,
        )
        if entry is not None:
            results.append(entry)

    print(f"\n[matrix] Completed: {len(results)}/{total} models")
    return results
