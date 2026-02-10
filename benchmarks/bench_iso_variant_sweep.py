#!/usr/bin/env python3
"""Run benchmark variants in isolated subprocesses to avoid cross-run OOM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUITES: dict[str, dict[str, Any]] = {
    "qwen3": {
        "script": "bench_qwen3_a3b_experiments.py",
        "default_variants": [
            "control_patterns_moe_mlp",
            "qwen_router_argpartition_logits",
        ],
        "supports_allow_unsafe": False,
    },
    "glm47": {
        "script": "bench_glm47_flash_experiments.py",
        "default_variants": [
            "control_swiglu_moe",
        ],
        "supports_allow_unsafe": True,
    },
}


def _relpath(repo_root: Path, path: Path) -> str:
    """Return *path* as a repo-relative POSIX path when possible."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _parse_env(pairs: list[str]) -> dict[str, str]:
    env_overrides: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"invalid --env value (expected KEY=VALUE): {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --env key in {pair!r}")
        env_overrides[key] = value
    return env_overrides


def _load_capsule(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    variant_key = next((k for k in obj.keys() if k != "meta"), None)
    if variant_key is None:
        raise ValueError(f"capsule has no variant payload: {path.name}")
    payload = obj[variant_key]
    baseline = payload["baseline"]
    patched = payload["patched"]
    fidelity = payload["fidelity"]
    summary = payload["summary"]
    return {
        "variant": payload["variant"],
        "fidelity": fidelity,
        "decode": {
            "baseline": baseline["median_decode"],
            "patched": patched["median_decode"],
            "speedup": summary["decode_speedup"],
        },
        "prefill": {
            "baseline": baseline["median_prefill"],
            "patched": patched["median_prefill"],
            "change": summary["prefill_change"],
        },
        "memory_gb": {
            "baseline": baseline["peak_mem_gb"],
            "patched": patched["peak_mem_gb"],
        },
    }


def _run_variant(
    *,
    repo_root: Path,
    script_path: Path,
    variant: str,
    runs: int,
    max_tokens: int,
    capsule_path: Path,
    ledger_exec: str,
    ledger_json: str,
    env_overrides: dict[str, str],
    allow_unsafe: bool,
    supports_allow_unsafe: bool,
) -> tuple[int, dict[str, Any]]:
    cmd_exec = [
        sys.executable,
        str(script_path),
        "--runs",
        str(runs),
        "--max-tokens",
        str(max_tokens),
        "--json-out",
        str(capsule_path),
        "--variants",
        variant,
    ]
    if ledger_exec != "":
        cmd_exec.extend(["--ledger", ledger_exec])
    if allow_unsafe and supports_allow_unsafe:
        cmd_exec.append("--allow-unsafe")

    env = os.environ.copy()
    env.update(env_overrides)
    proc = subprocess.run(cmd_exec, env=env, cwd=repo_root)

    cmd_json = [
        "python",
        _relpath(repo_root, script_path),
        "--runs",
        str(runs),
        "--max-tokens",
        str(max_tokens),
        "--json-out",
        _relpath(repo_root, capsule_path),
        "--variants",
        variant,
    ]
    if ledger_json != "":
        cmd_json.extend(["--ledger", ledger_json])
    if allow_unsafe and supports_allow_unsafe:
        cmd_json.append("--allow-unsafe")

    return proc.returncode, {"command": cmd_json, "env": env_overrides}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark variants one-by-one in isolated subprocesses and emit a "
            "summary JSON."
        )
    )
    parser.add_argument("--suite", choices=sorted(SUITES.keys()), required=True)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Output prefix for capsules and summary (default: "
            "<suite>_iso_<UTC timestamp>)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/repro_capsules",
        help="Directory for generated capsules + summary JSON.",
    )
    parser.add_argument(
        "--ledger",
        default="benchmarks/matrix.jsonl",
        help="Matrix JSONL path passed to each child benchmark script ('' to disable).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment overrides for each run (repeatable KEY=VALUE).",
    )
    parser.add_argument(
        "--allow-unsafe",
        action="store_true",
        help=(
            "Forward --allow-unsafe to suite scripts that support it "
            "(currently GLM) to run variants marked unsafe."
        ),
    )
    args = parser.parse_args()

    suite_cfg = SUITES[args.suite]
    variants = args.variants or suite_cfg["default_variants"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = args.prefix or f"{args.suite}_iso_{timestamp}"
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = _parse_env(args.env)
    ledger_exec = args.ledger
    ledger_json = args.ledger
    if ledger_exec != "":
        ledger_path = Path(ledger_exec)
        if not ledger_path.is_absolute():
            ledger_json = ledger_path.as_posix()
            ledger_path = repo_root / ledger_path
        else:
            ledger_json = _relpath(repo_root, ledger_path)
        ledger_exec = str(ledger_path)

    script_path = Path(__file__).resolve().parent / suite_cfg["script"]
    summary: dict[str, Any] = {
        "suite": args.suite,
        "timestamp_utc": timestamp,
        "script": str(script_path.name),
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "variants": variants,
        "env_overrides": env_overrides,
        "results": [],
    }

    for variant in variants:
        capsule_name = f"{prefix}_{variant}.json"
        capsule_path = output_dir / capsule_name
        print(f"\n=== {variant} ===")
        rc, meta = _run_variant(
            repo_root=repo_root,
            script_path=script_path,
            variant=variant,
            runs=args.runs,
            max_tokens=args.max_tokens,
            capsule_path=capsule_path,
            ledger_exec=ledger_exec,
            ledger_json=ledger_json,
            env_overrides=env_overrides,
            allow_unsafe=bool(args.allow_unsafe),
            supports_allow_unsafe=bool(suite_cfg.get("supports_allow_unsafe", False)),
        )
        result: dict[str, Any] = {
            "variant": variant,
            "return_code": rc,
            "capsule": _relpath(repo_root, capsule_path),
            **meta,
        }
        if rc == 0 and capsule_path.exists():
            try:
                result["metrics"] = _load_capsule(capsule_path)
            except Exception as exc:  # pragma: no cover - defensive fallback
                result["metrics_error"] = str(exc)
        else:
            result["error"] = "benchmark failed or capsule missing"
        summary["results"].append(result)

    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
