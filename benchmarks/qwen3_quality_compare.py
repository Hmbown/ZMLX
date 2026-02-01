#!/usr/bin/env python3
"""Compare Qwen3 coding responses baseline vs ZMLX patched.

Generates a small set of coding prompts and captures outputs for:
1) Baseline (unpatched)
2) ZMLX patched (moe_mlp)

Writes a Markdown report for easy side-by-side review.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

DEFAULT_MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"

PROMPTS: list[tuple[str, str]] = [
    (
        "Merge intervals",
        "Write a Python function `merge_intervals(intervals)` that merges "
        "overlapping intervals. Include a short explanation of time complexity "
        "and one example input/output.",
    ),
    (
        "Fix a bug",
        "The following function is intended to return unique items while "
        "preserving order, but it has a bug. Explain the bug and fix it.\n\n"
        "```python\n"
        "def unique(items):\n"
        "    seen = {}\n"
        "    out = []\n"
        "    for item in items:\n"
        "        if item not in seen:\n"
        "            seen[item] = True\n"
        "            out.append(item)\n"
        "    return out\n"
        "```\n",
    ),
    (
        "Log parser",
        "Write a Python function that parses a log file and returns counts per "
        "level (INFO/WARN/ERROR). Include a minimal unit test using `pytest`.",
    ),
]


@dataclass
class Response:
    text: str
    prompt_tps: float
    gen_tps: float
    prompt_tokens: int
    gen_tokens: int
    peak_mem_gb: float


def _build_prompt(tokenizer, user_text: str) -> str:
    if tokenizer.chat_template is None:
        return user_text
    messages = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _warmup(model, tokenizer) -> None:
    import mlx_lm

    _ = mlx_lm.generate(model, tokenizer, prompt="Hi", max_tokens=5)
    mx.eval(mx.zeros(1))
    sync = getattr(mx, "synchronize", None)
    if callable(sync):
        sync()


def _generate(model, tokenizer, prompt: str, max_tokens: int) -> Response:
    import mlx_lm
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    last = None
    token_ids: list[int] = []
    parts: list[str] = []
    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        last = resp
        if hasattr(resp, "token"):
            try:
                token_ids.append(int(resp.token))
            except (TypeError, ValueError):
                pass
        if hasattr(resp, "text") and isinstance(resp.text, str):
            parts.append(resp.text)

    if last is None:
        return Response("", 0.0, 0.0, 0, 0, 0.0)

    text = tokenizer.decode(token_ids) if token_ids else "".join(parts)
    return Response(
        text=text,
        prompt_tps=last.prompt_tps,
        gen_tps=last.generation_tps,
        prompt_tokens=last.prompt_tokens,
        gen_tokens=last.generation_tokens,
        peak_mem_gb=last.peak_memory,
    )


def _clear_gpu() -> None:
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    elif hasattr(mx.metal, "clear_cache"):
        mx.metal.clear_cache()


def _run_config(model_path: str, label: str, patched: bool, max_tokens: int) -> dict[str, Any]:
    import mlx_lm

    from zmlx.patch import patch as zmlx_patch

    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    model, tokenizer = mlx_lm.load(model_path)
    if patched:
        zmlx_patch(model, patterns=["moe_mlp"], verbose=True)
    _warmup(model, tokenizer)

    outputs: list[dict[str, Any]] = []
    for title, user_text in PROMPTS:
        prompt = _build_prompt(tokenizer, user_text)
        resp = _generate(model, tokenizer, prompt, max_tokens)
        outputs.append(
            {
                "title": title,
                "prompt": user_text,
                "response": resp.text,
                "metrics": {
                    "prompt_tps": resp.prompt_tps,
                    "gen_tps": resp.gen_tps,
                    "prompt_tokens": resp.prompt_tokens,
                    "gen_tokens": resp.gen_tokens,
                    "peak_mem_gb": resp.peak_mem_gb,
                },
            }
        )

    del model, tokenizer
    _clear_gpu()
    return {
        "label": label,
        "outputs": outputs,
    }


def _write_markdown(
    out_path: Path,
    model: str,
    max_tokens: int,
    baseline: dict[str, Any],
    patched: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Qwen3-30B-A3B-Instruct response comparison\n")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- MLX version: `{mx.__version__}`")
    lines.append(f"- gather_qmm_swiglu: `{hasattr(mx, 'gather_qmm_swiglu')}`")
    lines.append(f"- max_tokens: `{max_tokens}`")
    lines.append("")

    for idx, item in enumerate(baseline["outputs"]):
        title = item["title"]
        lines.append(f"## {idx + 1}. {title}\n")
        lines.append("**Prompt**")
        lines.append("```")
        lines.append(item["prompt"])
        lines.append("```\n")

        base_resp = item["response"].strip()
        pat_resp = patched["outputs"][idx]["response"].strip()

        lines.append("**Baseline (unpatched)**")
        lines.append("```")
        lines.append(base_resp or "[empty]")
        lines.append("```\n")

        lines.append("**ZMLX patched (`moe_mlp`)**")
        lines.append("```")
        lines.append(pat_resp or "[empty]")
        lines.append("```\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 response comparison")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument(
        "--out",
        default="benchmarks/reports/qwen3_a3b_instruct_quality.md",
        help="Markdown output path",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    print(f"gather_qmm_swiglu available: {hasattr(mx, 'gather_qmm_swiglu')}")

    baseline = _run_config(args.model, "Baseline (unpatched)", False, args.max_tokens)
    patched = _run_config(args.model, "ZMLX patched (moe_mlp)", True, args.max_tokens)

    _write_markdown(out_path, args.model, args.max_tokens, baseline, patched)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
