"""Model catalog: ModelInfo dataclass and loader from Exo TOML cards."""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Known MoE model families (by model-id substring matching).
_MOE_FAMILIES = {"deepseek", "glm-4.7", "qwen3-30b-a3b", "qwen3-235b-a22b",
                 "qwen3-coder-480b", "qwen3-next-80b-a3b",
                 "kimi-k2", "minimax", "gpt-oss", "lfm2"}

# Hardware tiers: name -> usable RAM in GB (rough: need ~2GB headroom).
_HARDWARE_TIERS: dict[str, float] = {
    "M1 Pro 16GB": 14.0,
    "M1 Pro 32GB": 30.0,
    "M1 Max 64GB": 60.0,
    "M2 Ultra 192GB": 185.0,
    "M4 Max 36GB": 33.0,
    "M4 Max 64GB": 60.0,
    "M4 Max 128GB": 120.0,
    "M4 Ultra 192GB": 185.0,
}


@dataclass
class ModelInfo:
    """Metadata about a model in the test matrix."""

    model_id: str
    display_name: str
    family: str
    architecture: str  # "moe" or "dense"
    total_params: str  # e.g. "671B", "8B", "30B-A3B"
    quant: str  # e.g. "4bit", "8bit", "bf16"
    n_layers: int = 0
    hidden_size: int = 0
    storage_gb: float = 0.0
    supports_tensor_parallel: bool = False
    source: str = "exo"  # "exo" or "zmlx"
    zmlx_family: str = ""
    expected_patterns: list[str] = field(default_factory=list)
    excluded_patterns: dict[str, str] = field(default_factory=dict)
    fits_on: list[str] = field(default_factory=list)
    notes: str = ""


def _infer_family(model_id: str) -> str:
    """Infer the model family from the HuggingFace model ID."""
    name = model_id.lower()
    if "glm" in name:
        return "glm"
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    if "deepseek" in name:
        return "deepseek"
    if "kimi" in name:
        return "kimi"
    if "minimax" in name:
        return "minimax"
    if "gpt-oss" in name or "gpt_oss" in name:
        return "gpt_oss"
    if "lfm" in name:
        return "lfm"
    if "nemotron" in name:
        return "nemotron"
    if "mixtral" in name:
        return "mixtral"
    return "unknown"


def _infer_zmlx_family(family: str) -> str:
    """Map catalog family to the key _model_family() returns at runtime."""
    mapping = {
        "glm": "glm",
        "qwen": "qwen",
        "llama": "llama",
        "deepseek": "deepseek",
        "kimi": "deepseek",  # Kimi uses DeepSeek architecture
        "minimax": "deepseek",  # MiniMax M2 uses DeepSeek architecture
        "gpt_oss": "gpt_oss",
        "lfm": "lfm",
        "nemotron": "nemotron",
        "mixtral": "mixtral",
    }
    return mapping.get(family, "unknown")


def _infer_architecture(model_id: str, family: str) -> str:
    """Infer whether a model is MoE or dense."""
    name = model_id.lower()
    for pattern in _MOE_FAMILIES:
        if pattern in name:
            return "moe"
    if family in {"deepseek", "kimi", "minimax", "gpt_oss"}:
        return "moe"
    return "dense"


def _infer_quant(model_id: str) -> str:
    """Infer quantization from model ID suffix."""
    name = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
    name_lower = name.lower()
    if "mxfp4-q8" in name_lower:
        return "MXFP4-Q8"
    if "bf16" in name_lower:
        return "bf16"
    if "fp16" in name_lower:
        return "fp16"
    m = re.search(r"(\d)bit", name_lower)
    if m:
        return f"{m.group(1)}bit"
    # Kimi-K2.5 / Kimi-K2-Thinking without explicit quant suffix
    if name_lower.endswith((".5", "-thinking")):
        return "4bit"
    return "unknown"


def _infer_total_params(model_id: str) -> str:
    """Extract human-readable param count from model name."""
    name = model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id
    # Patterns like "30B-A3B", "235B-A22B", "480B-A35B", "80B-A3B"
    m = re.search(r"(\d+B-A\d+B)", name, re.IGNORECASE)
    if m:
        return m.group(1)
    # Patterns like "8B", "70B", "120b", "1B"
    m = re.search(r"(\d+\.?\d*)[Bb]", name)
    if m:
        return f"{m.group(1)}B"
    # DeepSeek V3.1 -> known 671B
    if "deepseek-v3" in name.lower():
        return "671B"
    # Kimi K2 -> known MoE
    if "kimi-k2.5" in name.lower():
        return "671B"
    if "kimi-k2" in name.lower():
        return "1T"
    # GLM-4.7 -> known ~8B active
    if "glm-4.7-flash" in name.lower():
        return "60B-A4B"
    if "glm-4.7" in name.lower():
        return "400B-A30B"
    if "glm-4.5-air" in name.lower():
        return "9B"
    if "minimax-m2" in name.lower():
        return "456B"
    return ""


def _display_name(model_id: str) -> str:
    """Short display name from full HF path."""
    return model_id.rsplit("/", 1)[-1] if "/" in model_id else model_id


def _compute_expected_patterns(zmlx_family: str, architecture: str) -> tuple[list[str], dict[str, str]]:
    """Compute expected patterns and exclusion reasons for a model."""
    from zmlx.patch import _FIDELITY_EXCLUDES, _PERF_EXCLUDES, FUSED_ACTIVATIONS

    all_default = list(FUSED_ACTIVATIONS)  # ["swiglu_mlp", "geglu_mlp", "moe_mlp"]
    fidelity_ex = _FIDELITY_EXCLUDES.get(zmlx_family, set())
    perf_ex = _PERF_EXCLUDES.get(zmlx_family, set())

    expected = []
    excluded: dict[str, str] = {}
    for p in all_default:
        if p in fidelity_ex:
            excluded[p] = "fidelity"
        elif p in perf_ex:
            excluded[p] = "perf"
        else:
            expected.append(p)

    # geglu_mlp only matches GeGLU architectures (none of the current models)
    if "geglu_mlp" in expected and architecture != "dense_geglu":
        expected.remove("geglu_mlp")

    # moe_mlp only applies to MoE models
    if "moe_mlp" in expected and architecture != "moe":
        expected.remove("moe_mlp")
    if "moe_mlp" in excluded and architecture != "moe":
        del excluded["moe_mlp"]

    return expected, excluded


def _compute_fits_on(storage_gb: float) -> list[str]:
    """Return hardware tiers that can fit this model."""
    return sorted(name for name, ram in _HARDWARE_TIERS.items() if ram >= storage_gb)


def _load_exo_tomls(exo_cards_dir: str | None = None) -> list[ModelInfo]:
    """Load model info from Exo TOML inference model cards."""
    if exo_cards_dir is None:
        # Default: look relative to ZMLX repo root
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        exo_cards_dir = str(repo_root / "exo" / "resources" / "inference_model_cards")

    toml_files = sorted(glob.glob(os.path.join(exo_cards_dir, "*.toml")))
    if not toml_files:
        return []

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    models: list[ModelInfo] = []
    for path in toml_files:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        model_id = data.get("model_id", "")
        if not model_id:
            continue

        n_layers = data.get("n_layers", 0)
        hidden_size = data.get("hidden_size", 0)
        storage_bytes = data.get("storage_size", {}).get("in_bytes", 0)
        storage_gb = storage_bytes / (1024 ** 3)
        supports_tensor = data.get("supports_tensor", False)

        family = _infer_family(model_id)
        zmlx_family = _infer_zmlx_family(family)
        architecture = _infer_architecture(model_id, family)

        try:
            expected, excluded = _compute_expected_patterns(zmlx_family, architecture)
        except Exception:
            expected, excluded = [], {}

        models.append(ModelInfo(
            model_id=model_id,
            display_name=_display_name(model_id),
            family=family,
            architecture=architecture,
            total_params=_infer_total_params(model_id),
            quant=_infer_quant(model_id),
            n_layers=n_layers,
            hidden_size=hidden_size,
            storage_gb=round(storage_gb, 1),
            supports_tensor_parallel=supports_tensor,
            source="exo",
            zmlx_family=zmlx_family,
            expected_patterns=expected,
            excluded_patterns=excluded,
            fits_on=_compute_fits_on(storage_gb),
        ))

    return models


def _manual_models() -> list[ModelInfo]:
    """Models not in the Exo catalog (e.g. LFM2 variants)."""
    extras: list[ModelInfo] = []
    for model_id, quant, storage_gb in [
        ("mlx-community/LFM2-8B-A1B-4bit", "4bit", 5.0),
        ("mlx-community/LFM2-8B-A1B-8bit", "8bit", 9.0),
    ]:
        family = "lfm"
        zmlx_family = "lfm"
        try:
            expected, excluded = _compute_expected_patterns(zmlx_family, "moe")
        except Exception:
            expected, excluded = [], {}

        extras.append(ModelInfo(
            model_id=model_id,
            display_name=_display_name(model_id),
            family=family,
            architecture="moe",
            total_params="8B-A1B",
            quant=quant,
            n_layers=24,
            hidden_size=2048,
            storage_gb=storage_gb,
            supports_tensor_parallel=False,
            source="zmlx",
            zmlx_family=zmlx_family,
            expected_patterns=expected,
            excluded_patterns=excluded,
            fits_on=_compute_fits_on(storage_gb),
            notes="LFM2 — primary ZMLX benchmark model",
        ))
    return extras


def load_catalog(exo_cards_dir: str | None = None) -> list[ModelInfo]:
    """Load the full model catalog: Exo TOMLs + manual additions.

    Returns models sorted by family, then storage size.
    """
    models = _load_exo_tomls(exo_cards_dir)

    # Add manual models, avoiding duplicates
    existing_ids = {m.model_id for m in models}
    for m in _manual_models():
        if m.model_id not in existing_ids:
            models.append(m)

    models.sort(key=lambda m: (m.family, m.storage_gb))
    return models


def print_catalog(models: list[ModelInfo] | None = None) -> None:
    """Print a formatted table of the model catalog."""
    if models is None:
        models = load_catalog()

    hdr = (
        f"{'Family':<10} {'Model':<42} {'Arch':<5} {'Quant':<9} "
        f"{'Size':>6} {'Layers':>6} {'Patterns':<24} {'Notes'}"
    )
    sep = "-" * len(hdr)
    print(f"\nZMLX Model Catalog ({len(models)} models)\n")
    print(hdr)
    print(sep)

    for m in models:
        patterns_str = ", ".join(m.expected_patterns) if m.expected_patterns else "(none)"
        if m.excluded_patterns:
            excl = ", ".join(f"{k}({v})" for k, v in m.excluded_patterns.items())
            patterns_str += f" | excl: {excl}"
        size_str = f"{m.storage_gb:.0f}GB" if m.storage_gb >= 1 else f"{m.storage_gb:.1f}GB"
        print(
            f"{m.family:<10} {m.display_name:<42} {m.architecture:<5} {m.quant:<9} "
            f"{size_str:>6} {m.n_layers:>6} {patterns_str:<24} {m.notes}"
        )
    print()
