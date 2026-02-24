"""Kernel-generation SFT export.

Builds chat-style JSONL datasets for ``mlx_lm.lora`` from:
- Foundry session logs under ``sessions/`` (attempt records + rendered kernels)
- KD discovery runs under ``runs/*/run.ndjson``
- Discover session trees under ``discover_sessions/*session.json``
- In-repo Metal templates (foundry + kd)
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


def export_kernel_sft_jsonl(
    out_dir: str,
    *,
    sessions_root: str = "sessions",
    runs_root: str = "runs",
    discover_root: str = "discover_sessions",
    train_fraction: float = 0.96,
    valid_fraction: float = 0.03,
    seed: int = 42,
    max_examples: int | None = None,
    include_failed_kd: bool = False,
) -> dict[str, Any]:
    """Export a chat-format SFT dataset for kernel generation.

    Output files:
    - ``train.jsonl``
    - ``valid.jsonl``
    - ``test.jsonl``
    - ``manifest.json``
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1)")
    if not (0.0 <= valid_fraction < 1.0):
        raise ValueError("valid_fraction must be in [0, 1)")
    if train_fraction + valid_fraction >= 1.0:
        raise ValueError("train_fraction + valid_fraction must be < 1")

    examples: list[dict[str, Any]] = []
    source_counts = {
        "foundry_attempts": 0,
        "kd_runs": 0,
        "discover_sessions": 0,
        "templates": 0,
    }
    dedup: set[str] = set()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _append(example: dict[str, Any], source_key: str) -> None:
        key = _example_key(example)
        if key in dedup:
            return
        dedup.add(key)
        examples.append(example)
        source_counts[source_key] += 1

    for ex in _iter_foundry_examples(Path(sessions_root)):
        _append(ex, "foundry_attempts")

    for ex in _iter_kd_examples(Path(runs_root), include_failed=include_failed_kd):
        _append(ex, "kd_runs")

    for ex in _iter_discover_examples(Path(discover_root)):
        _append(ex, "discover_sessions")

    for ex in _iter_template_examples():
        _append(ex, "templates")

    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples is not None:
        examples = examples[: max(0, int(max_examples))]

    train_set, valid_set, test_set = _split_examples(
        examples,
        train_fraction=train_fraction,
        valid_fraction=valid_fraction,
    )

    train_path = out_path / "train.jsonl"
    valid_path = out_path / "valid.jsonl"
    test_path = out_path / "test.jsonl"

    _write_jsonl(train_path, train_set)
    _write_jsonl(valid_path, valid_set)
    _write_jsonl(test_path, test_set)

    manifest = {
        "schema_version": "1.0",
        "seed": seed,
        "counts": {
            "total": len(examples),
            "train": len(train_set),
            "valid": len(valid_set),
            "test": len(test_set),
        },
        "source_counts": source_counts,
        "paths": {
            "train": str(train_path),
            "valid": str(valid_path),
            "test": str(test_path),
        },
    }
    (out_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _iter_foundry_examples(sessions_root: Path) -> list[dict[str, Any]]:
    if not sessions_root.exists():
        return []

    examples: list[dict[str, Any]] = []
    for log_path in sorted(sessions_root.rglob("*.ndjson")):
        if not _is_foundry_attempt_log(log_path):
            continue
        session_dir = _attempt_session_dir(log_path)
        for rec in _iter_jsonl(log_path):
            if not _foundry_success(rec):
                continue
            source_rel = str(rec.get("kernel", {}).get("source_metal", "")).strip()
            if not source_rel:
                continue
            source = _read_source_from_attempt(session_dir, log_path.parent, source_rel)
            if not source:
                continue

            prompt = _build_foundry_prompt(rec)
            meta = {
                "source_type": "foundry_attempt",
                "session": session_dir.name,
                "op": rec.get("op"),
                "dtype": rec.get("dtype"),
                "shape": rec.get("shape", {}),
                "template_id": rec.get("kernel", {}).get("template_id"),
                "knobs": rec.get("kernel", {}).get("knobs", {}),
                "latency_ms_p50": rec.get("bench", {}).get("latency_ms", {}).get("p50"),
            }
            examples.append(_chat_example(prompt, source, meta))
    return examples


def _iter_kd_examples(runs_root: Path, *, include_failed: bool) -> list[dict[str, Any]]:
    if not runs_root.exists():
        return []

    examples: list[dict[str, Any]] = []
    for run_path in sorted(runs_root.rglob("run.ndjson")):
        for rec in _iter_jsonl(run_path):
            source = str(rec.get("metal_source", "")).strip()
            if not source:
                continue

            status = rec.get("status")
            if not include_failed and status != "benchmarked":
                continue
            if not include_failed and rec.get("metrics", {}).get("failure"):
                continue

            prompt = _build_kd_prompt(rec)
            meta = {
                "source_type": "kd_run",
                "run_dir": run_path.parent.name,
                "op": rec.get("op_name"),
                "status": status,
                "candidate_id": rec.get("candidate_id"),
                "speedup_vs_ref": rec.get("metrics", {}).get("speedup_vs_ref"),
            }
            examples.append(_chat_example(prompt, source, meta))
    return examples


def _iter_discover_examples(discover_root: Path) -> list[dict[str, Any]]:
    if not discover_root.exists():
        return []

    examples: list[dict[str, Any]] = []
    for session_path in sorted(discover_root.rglob("*session.json")):
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        metadata = payload.get("metadata", {})
        target_name = str(metadata.get("target_name", "unknown"))

        added = 0
        root = payload.get("tree_data", {}).get("root")
        if isinstance(root, dict):
            for node in _iter_tree_nodes(root):
                eval_result = node.get("eval_result", {})
                if not (eval_result.get("compiled") and eval_result.get("correct")):
                    continue
                source = str(node.get("candidate", {}).get("spec", {}).get("source", "")).strip()
                if not source:
                    continue

                prompt = _build_discover_prompt(target_name, node, metadata)
                meta = {
                    "source_type": "discover_session",
                    "session_file": str(session_path),
                    "target": target_name,
                    "node_id": node.get("node_id"),
                    "reward": eval_result.get("reward"),
                    "speedup": eval_result.get("speedup"),
                }
                examples.append(_chat_example(prompt, source, meta))
                added += 1

        if added == 0:
            best_source = str(metadata.get("best_source", "")).strip()
            if best_source:
                prompt = (
                    "Write an optimized Metal kernel body for target "
                    f"`{target_name}`. Return only source code."
                )
                meta = {
                    "source_type": "discover_session",
                    "session_file": str(session_path),
                    "target": target_name,
                    "best_speedup": metadata.get("best_speedup"),
                }
                examples.append(_chat_example(prompt, best_source, meta))

    return examples


def _iter_template_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    here = Path(__file__).resolve()
    zmlx_root = here.parents[2]
    foundry_templates = here.parents[1] / "templates"
    kd_templates = zmlx_root / "kd" / "templates"

    for tpl_path in sorted(foundry_templates.rglob("*.metal")):
        source = tpl_path.read_text(encoding="utf-8")
        op = tpl_path.parent.name
        prompt = (
            "Write a parameterized Metal template for op "
            f"`{op}` using ``{{PLACEHOLDER}}`` markers where needed."
        )
        meta = {
            "source_type": "template",
            "family": "foundry",
            "op": op,
            "template_file": str(tpl_path),
        }
        examples.append(_chat_example(prompt, source, meta))

    for tpl_path in sorted(kd_templates.rglob("*.tmpl")):
        source = tpl_path.read_text(encoding="utf-8")
        op = tpl_path.stem.split(".")[0]
        prompt = (
            "Write a Metal kernel body template for discovery op "
            f"`{op}` using explicit compile-time constants."
        )
        meta = {
            "source_type": "template",
            "family": "kd",
            "op": op,
            "template_file": str(tpl_path),
        }
        examples.append(_chat_example(prompt, source, meta))

    return examples


def _iter_tree_nodes(root: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        for child in node.get("children", []):
            if isinstance(child, dict):
                stack.append(child)
    return out


def _chat_example(prompt: str, completion: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": completion.rstrip()},
        ],
        "metadata": metadata,
    }


def _split_examples(
    examples: list[dict[str, Any]],
    *,
    train_fraction: float,
    valid_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(examples)
    if n == 0:
        return [], [], []

    n_train = int(n * train_fraction)
    n_valid = int(n * valid_fraction)

    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_valid = max(1, min(n_valid, n - n_train - 1))
    else:
        n_train = max(1, n - 1)
        n_valid = 0

    n_test = n - n_train - n_valid
    if n_test < 0:
        n_test = 0
        n_valid = max(0, n - n_train)

    train = examples[:n_train]
    valid = examples[n_train : n_train + n_valid]
    test = examples[n_train + n_valid :]
    return train, valid, test


def _example_key(example: dict[str, Any]) -> str:
    payload = json.dumps(example, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            f.write("\n")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _is_foundry_attempt_log(path: Path) -> bool:
    if path.name == "attempts.ndjson":
        return True
    if path.name.startswith("attempts.worker"):
        return True
    return path.parent.name == "attempts" and path.suffix == ".ndjson"


def _attempt_session_dir(log_path: Path) -> Path:
    if log_path.parent.name == "attempts":
        return log_path.parent.parent
    return log_path.parent


def _foundry_success(rec: dict[str, Any]) -> bool:
    return bool(
        rec.get("build", {}).get("ok")
        and rec.get("correctness", {}).get("ok")
        and rec.get("bench", {}).get("ok")
    )


def _read_source_from_attempt(session_dir: Path, log_parent: Path, source_rel: str) -> str:
    candidates = [
        session_dir / source_rel,
        log_parent / source_rel,
    ]
    for source_path in candidates:
        if source_path.exists():
            return source_path.read_text(encoding="utf-8")
    return ""


def _build_foundry_prompt(rec: dict[str, Any]) -> str:
    op = rec.get("op")
    dtype = rec.get("dtype")
    shape = rec.get("shape", {})
    spec = rec.get("spec", {})
    kernel = rec.get("kernel", {})

    return (
        "Write a Metal kernel body for this MLX operation.\n"
        f"op: {op}\n"
        f"dtype: {dtype}\n"
        f"shape: {_compact(shape)}\n"
        f"math: {spec.get('math', '')}\n"
        f"constraints: {_compact(spec.get('constraints', {}))}\n"
        f"template_id: {kernel.get('template_id')}\n"
        f"knobs: {_compact(kernel.get('knobs', {}))}\n"
        "Return only Metal source code."
    )


def _build_kd_prompt(rec: dict[str, Any]) -> str:
    op = rec.get("op_name")
    inputs_spec = rec.get("inputs_spec", [])
    outputs_spec = rec.get("outputs_spec", [])
    launch = rec.get("launch_params", {})
    params = rec.get("template_params", {})
    notes = rec.get("notes", {})

    return (
        "Write a Metal kernel body for this discovery target.\n"
        f"op: {op}\n"
        f"inputs: {_compact(inputs_spec)}\n"
        f"outputs: {_compact(outputs_spec)}\n"
        f"template_params: {_compact(params)}\n"
        f"launch: {_compact(launch)}\n"
        f"shape_signature: {_compact(notes.get('shape_signature', {}))}\n"
        "Return only Metal source code."
    )


def _build_discover_prompt(
    target_name: str,
    node: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    strategy = node.get("candidate", {}).get("llm_reasoning", "")
    eval_result = node.get("eval_result", {})
    return (
        "Write an optimized Metal kernel body for this discover target.\n"
        f"target: {target_name}\n"
        f"device_chip: {metadata.get('device_chip', 'unknown')}\n"
        f"strategy_hint: {strategy}\n"
        f"observed_speedup: {eval_result.get('speedup')}\n"
        "Return only Metal source code."
    )


def _compact(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

