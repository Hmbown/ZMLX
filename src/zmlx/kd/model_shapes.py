"""Model-aware shape derivation for kernel discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SUPPORTED_OPS = ("rmsnorm_residual", "swiglu", "rope")


def parse_decode_rows(raw: str | None) -> tuple[int, ...]:
    """Parse a comma-separated decode-rows string into a stable tuple."""
    if raw is None:
        return (1, 2, 4)
    out: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("decode rows must be positive integers")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError("decode rows cannot be empty")
    return tuple(out)


def load_hf_config(
    model_id: str,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Load Hugging Face ``config.json`` for a model."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - exercised only when dependency missing.
        raise RuntimeError(
            "huggingface_hub is required for --model-id shape derivation"
        ) from exc

    kwargs: dict[str, Any] = {
        "repo_id": model_id,
        "filename": "config.json",
        "local_files_only": bool(local_files_only),
    }
    if revision:
        kwargs["revision"] = revision
    path = hf_hub_download(**kwargs)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _first_int(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in config:
            parsed = _as_int(config.get(key))
            if parsed is not None and parsed > 0:
                return parsed
    return None


def _swiglu_dim(config: dict[str, Any]) -> int | None:
    model_type = str(config.get("model_type", "")).lower()
    moe_intermediate = _first_int(config, "moe_intermediate_size")
    n_shared = _as_int(config.get("n_shared_experts", 1)) or 1

    # GLM-4.7 shared_experts path uses this shape at decode.
    if "glm4_moe_lite" in model_type and moe_intermediate is not None:
        return moe_intermediate * max(1, n_shared)

    if "shared_expert_intermediate_size" in config:
        shared = _first_int(config, "shared_expert_intermediate_size")
        if shared is not None:
            return shared

    if moe_intermediate is not None and n_shared > 1:
        return moe_intermediate * n_shared

    return _first_int(
        config,
        "intermediate_size",
        "ffn_hidden_size",
        "feed_forward_proj_size",
    )


def derive_shape_suite_from_config(
    op_name: str,
    config: dict[str, Any],
    *,
    decode_rows: tuple[int, ...] = (1, 2, 4),
) -> list[dict[str, int]]:
    """Derive op-specific shape signatures from model config."""
    if op_name not in _SUPPORTED_OPS:
        raise ValueError(
            f"Unsupported op {op_name!r} for model-derived shapes. "
            f"Expected one of: {', '.join(_SUPPORTED_OPS)}"
        )

    rows = tuple(int(v) for v in decode_rows)
    if not rows:
        raise ValueError("decode_rows cannot be empty")

    if op_name == "rmsnorm_residual":
        hidden = _first_int(config, "hidden_size", "n_embd", "d_model")
        if hidden is None:
            raise ValueError("Could not infer hidden size for rmsnorm_residual")
        return [{"rows": r, "D": hidden} for r in rows]

    if op_name == "swiglu":
        dim = _swiglu_dim(config)
        if dim is None:
            raise ValueError("Could not infer SwiGLU width for swiglu op")
        return [{"rows": r, "D": dim} for r in rows]

    heads = _first_int(config, "num_attention_heads", "n_head")
    d_nope = _first_int(config, "qk_nope_head_dim")
    d_rope = _first_int(config, "qk_rope_head_dim")
    if heads is None or d_nope is None or d_rope is None:
        raise ValueError(
            "Could not infer GLM decode rope dims (need num_attention_heads, "
            "qk_nope_head_dim, qk_rope_head_dim)"
        )
    return [{"B": r, "H_Q": heads, "D_NOPE": d_nope, "D_ROPE": d_rope} for r in rows]

