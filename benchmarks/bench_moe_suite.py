#!/usr/bin/env python3
"""Batch MoE benchmark runner (sequential, cache-dir aware).

Runs benchmarks one model at a time (never parallel), writes per-model JSON,
and aggregates a suite summary for overnight runs.

Example:
  python benchmarks/bench_moe_suite.py \\
    --model-list benchmarks/moe_models.txt \\
    --cache-dir .benchmarks/hf_cache \\
    --runs 3 --max-tokens 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def load_models(path: Path) -> list[str]:
    models: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        models.append(line)
    return models


def run_one(
    model: str,
    *,
    runs: int,
    max_tokens: int,
    fused_max_tokens: int | None,
    out_dir: Path,
    env: dict[str, str],
    resume: bool,
) -> dict:
    safe_name = model.replace("/", "__")
    json_out = out_dir / f"{safe_name}.json"
    log_out = out_dir / f"{safe_name}.log"

    if resume and json_out.exists():
        return json.loads(json_out.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        "benchmarks/bench_moe_e2e.py",
        "--model",
        model,
        "--runs",
        str(runs),
        "--max-tokens",
        str(max_tokens),
        "--json-out",
        str(json_out),
    ]
    if fused_max_tokens is not None:
        cmd += ["--fused-max-tokens", str(fused_max_tokens)]

    with log_out.open("w", encoding="utf-8") as log:
        print(f"\n==> {model}")
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    return json.loads(json_out.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch MoE benchmark runner")
    parser.add_argument(
        "--model-list",
        default="benchmarks/moe_models.txt",
        help="Path to model list file",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_HOME", ".benchmarks/hf_cache"),
        help="HF cache directory (keeps models off main drive)",
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--fused-max-tokens",
        type=int,
        default=None,
        help="Override max token count for fused SwiGLU path",
    )
    parser.add_argument(
        "--out-dir",
        default=".benchmarks",
        help="Output directory for logs + JSON summaries",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip models that already have JSON output",
    )
    args = parser.parse_args()

    model_list = Path(args.model_list)
    if not model_list.exists():
        raise FileNotFoundError(f"Model list not found: {model_list}")

    models = load_models(model_list)
    if not models:
        print("No models found in list.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(cache_dir)

    suite_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_jsonl = out_dir / f"moe_suite_{suite_id}.jsonl"
    suite_json = out_dir / f"moe_suite_{suite_id}.json"

    results: list[dict] = []
    for model in models:
        start = time.time()
        try:
            payload = run_one(
                model,
                runs=args.runs,
                max_tokens=args.max_tokens,
                fused_max_tokens=args.fused_max_tokens,
                out_dir=out_dir,
                env=env,
                resume=args.resume,
            )
            payload["wall_sec"] = time.time() - start
            results.append(payload)
            with suite_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")
        except subprocess.CalledProcessError as exc:
            print(f"FAILED: {model} (exit {exc.returncode})")

    suite_payload = {
        "model_list": str(model_list),
        "cache_dir": str(cache_dir),
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "fused_max_tokens": args.fused_max_tokens,
        "results": results,
    }
    suite_json.write_text(json.dumps(suite_payload, indent=2), encoding="utf-8")
    print(f"\nWrote suite summary: {suite_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
