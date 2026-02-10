"""Runtime registry for discovered/pinned kernels."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..metal import kernel as metal_kernel
from ..msl import DEFAULT_HEADER
from .env import runtime_fingerprint
from .ops import dtype_name

_ENV_ENABLE = "ZMLX_USE_DISCOVERED_KERNELS"
_ENV_PINNED_PATH = "ZMLX_DISCOVERED_KERNELS_PATH"


_payload_cache: dict[str, Any] = {"path": None, "mtime": None, "payload": {"entries": []}}
_kernel_cache: dict[tuple[str, str], Any] = {}


def enabled() -> bool:
    raw = os.environ.get(_ENV_ENABLE, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def default_pinned_path() -> Path:
    env = os.environ.get(_ENV_PINNED_PATH)
    if env:
        return Path(env)
    return Path("configs/discovered_kernels.json")


def _load_payload(path: Path) -> dict[str, Any]:
    global _payload_cache

    if not path.exists():
        return {"entries": []}

    mtime = path.stat().st_mtime
    cached_path = _payload_cache.get("path")
    cached_mtime = _payload_cache.get("mtime")
    if cached_path == str(path) and cached_mtime == mtime:
        result: dict[str, Any] = _payload_cache.get("payload", {"entries": []})
        return result

    data = json.loads(path.read_text(encoding="utf-8"))
    if "entries" not in data and "ops" in data:
        entries: list[dict[str, Any]] = []
        for op_name, op_entry in data.get("ops", {}).items():
            for shape_sig in op_entry.get("shape_signatures", [{}]):
                entries.append(
                    {
                        "key": {
                            "op_name": op_name,
                            "mlx_version": "unknown",
                            "device_arch": "unknown",
                            "device_name": "unknown",
                            "dtype": (op_entry.get("supported_dtypes") or ["float16"])[0],
                            "shape_signature": shape_sig,
                        },
                        **{k: v for k, v in op_entry.items() if k != "shape_signatures"},
                    }
                )
        data = {"schema_version": "2", "entries": entries, "runtime": {}}

    _payload_cache = {
        "path": str(path),
        "mtime": mtime,
        "payload": data,
    }
    return data  # type: ignore[no-any-return]


def _shape_match(allowed: dict[str, Any], actual: dict[str, Any]) -> bool:
    def _canon(sig: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in sorted(sig):
            value = sig[key]
            try:
                out[str(key)] = int(value)
            except Exception:
                out[str(key)] = str(value)
        return out

    return _canon(allowed) == _canon(actual)


def _shape_allowed(entry: dict[str, Any], shape_signature: dict[str, Any]) -> bool:
    key = entry.get("key", {})
    allowed = key.get("shape_signature")
    if not isinstance(allowed, dict):
        return True
    if not allowed:
        return True
    if not isinstance(shape_signature, dict):
        return False
    return _shape_match(allowed, shape_signature)


def _runtime_match_score(entry_key: dict[str, Any], runtime: dict[str, Any]) -> int | None:
    score = 0
    for field in ("mlx_version", "device_arch", "device_name"):
        expected = str(entry_key.get(field, "unknown"))
        actual = str(runtime.get(field, "unknown"))
        if expected == actual:
            score += 2
            continue
        if expected == "unknown":
            score += 1
            continue
        return None
    return score


def get_kernel(
    op: str,
    dtype: str,
    shape_signature: dict[str, Any],
    *,
    pinned_path: str | Path | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return best pinned kernel metadata for runtime+op+dtype+shape."""
    if not enabled():
        return None

    path = Path(pinned_path) if pinned_path is not None else default_pinned_path()
    payload = _load_payload(path)
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return None

    rt = runtime or runtime_fingerprint()

    def _match(entry: dict[str, Any]) -> int | None:
        key = entry.get("key", {})
        if str(key.get("op_name", "")) != op:
            return None
        if str(key.get("dtype", "")) != dtype:
            return None
        runtime_score = _runtime_match_score(key, rt)
        if runtime_score is None:
            return None
        if not _shape_allowed(entry, shape_signature):
            return None
        return runtime_score

    ranked: list[tuple[int, int, float, str, dict[str, Any]]] = []
    for entry in entries:
        score = _match(entry)
        if score is None:
            continue
        key = entry.get("key", {})
        shape_sig = key.get("shape_signature", {})
        shape_score = len(shape_sig) if isinstance(shape_sig, dict) else 0
        latency = float(entry.get("metrics", {}).get("latency_us", float("inf")))
        candidate_id = str(entry.get("candidate_id", ""))
        ranked.append((score, shape_score, latency, candidate_id, entry))

    matched = [entry for _, _, _, _, entry in ranked]
    if not matched:
        return None

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return ranked[0][4]


def _compute_launch(
    launch_params: dict[str, Any],
    shape_signature: dict[str, Any],
    inputs: list[Any],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    kind = str(launch_params.get("launch_kind", ""))
    tg = int(launch_params.get("threadgroup_x", 256))

    if kind == "rmsnorm_rows_tg":
        rows = int(shape_signature["rows"])
        grid_x = rows * tg
        return (max(1, grid_x), 1, 1), (max(1, min(tg, grid_x)), 1, 1)

    if kind == "rmsnorm_residual_rows_tg":
        rows = int(shape_signature["rows"])
        grid_x = rows * tg
        return (max(1, grid_x), 1, 1), (max(1, min(tg, grid_x)), 1, 1)

    if kind == "swiglu_flat":
        n = int(shape_signature.get("N", int(inputs[0].size)))
        vec = int(launch_params.get("vec_width", 1))
        unroll = int(launch_params.get("unroll", 1))
        per_thread = max(1, vec * unroll)
        grid_x = (n + per_thread - 1) // per_thread
        return (max(1, grid_x), 1, 1), (max(1, min(tg, grid_x)), 1, 1)

    if kind == "rope_decode_pos":
        b = int(shape_signature["B"])
        hq = int(shape_signature["H_Q"])
        d_out = int(shape_signature["D_OUT"])
        grid_x = b * (hq + 1) * d_out
        return (max(1, grid_x), 1, 1), (max(1, min(tg, grid_x)), 1, 1)

    if "grid_x" in launch_params:
        grid_x = int(launch_params["grid_x"])
        return (max(1, grid_x), 1, 1), (max(1, min(tg, grid_x)), 1, 1)

    # Fallback elementwise shape.
    n = int(inputs[0].size)
    return (max(1, n), 1, 1), (max(1, min(tg, n)), 1, 1)


def _compile_kernel(entry: dict[str, Any], dtype: str) -> Any:
    key = (str(entry.get("candidate_id", "")), dtype)
    if key in _kernel_cache:
        return _kernel_cache[key]

    k = metal_kernel(
        name=str(entry["func_name"]),
        input_names=[spec["name"] for spec in entry["inputs_spec"]],
        output_names=[spec["name"] for spec in entry["outputs_spec"]],
        source=str(entry["metal_source"]),
        header=DEFAULT_HEADER,
        cache=True,
    )
    _kernel_cache[key] = k
    return k


def launch_kernel(
    *,
    entry: dict[str, Any],
    inputs: list[Any],
    output_shapes: list[tuple[int, ...]],
    output_dtypes: list[Any],
    shape_signature: dict[str, Any],
) -> list[Any] | None:
    """Launch a pinned kernel entry with computed grid/threadgroup."""
    import mlx.core as mx

    dtype = dtype_name(output_dtypes[0]) if output_dtypes else "float16"

    try:
        k = _compile_kernel(entry, dtype)
        grid, threadgroup = _compute_launch(entry.get("launch_params", {}), shape_signature, inputs)
        outputs = k(
            *inputs,
            template=[("T", output_dtypes[0])],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=output_shapes,
            output_dtypes=output_dtypes,
        )
        mx.eval(*outputs)
        sync = getattr(mx, "synchronize", None)
        if callable(sync):
            sync()
        return list(outputs)
    except Exception:
        return None
