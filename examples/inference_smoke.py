"""Minimal end-to-end inference smoke CLI for patched mlx-lm models."""

from __future__ import annotations

import argparse
import sys
from typing import Any, TextIO


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Load an mlx-lm model, apply ZMLX patch defaults, and generate text.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier or local model path (for mlx_lm.load).",
    )
    parser.add_argument(
        "--prompt",
        default="Explain mixture-of-experts in one sentence.",
        help="Prompt text to generate from.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(argv)


def _print_patch_summary(model: Any, *, out: TextIO) -> None:
    """Print best-effort patch summary from model patch metadata."""
    patch_result = getattr(model, "_zmlx_patch_result", None)
    if patch_result is None:
        print("[patch] Applied patch(model) (no patch summary metadata found).", file=out)
        return

    summary_fn = getattr(patch_result, "summary", None)
    if callable(summary_fn):
        summary = str(summary_fn())
    else:
        patched_count = int(getattr(patch_result, "patched_count", 0) or 0)
        pattern_counts = dict(getattr(patch_result, "pattern_counts", {}) or {})
        summary_lines = [f"Patched {patched_count} modules:"]
        for pattern_name, count in sorted(pattern_counts.items()):
            summary_lines.append(f"  {pattern_name}: {count}")
        summary = "\n".join(summary_lines)

    for line in summary.splitlines():
        print(f"[patch] {line}", file=out)


def run(
    args: argparse.Namespace,
    *,
    mlx_lm_module: Any | None = None,
    patch_fn: Any | None = None,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    """Run the inference smoke flow and return process exit code."""
    if args.max_tokens <= 0:
        print("[error] --max-tokens must be a positive integer.", file=stderr)
        return 2

    if mlx_lm_module is None or patch_fn is None:
        try:
            import mlx_lm

            from zmlx.patch import patch
        except Exception as exc:  # pragma: no cover - dependency/system specific
            print(
                "[error] Failed to import dependencies (mlx_lm and zmlx.patch). "
                f"Details: {exc}",
                file=stderr,
            )
            return 1

        if mlx_lm_module is None:
            mlx_lm_module = mlx_lm
        if patch_fn is None:
            patch_fn = patch

    print(f"[load] model={args.model_id}", file=stdout)
    try:
        loaded = mlx_lm_module.load(args.model_id)
        model, tokenizer = loaded[0], loaded[1]
    except Exception as exc:
        print(
            f"[error] Failed to load model '{args.model_id}'. Details: {exc}",
            file=stderr,
        )
        return 1

    print("[patch] Applying zmlx.patch.patch(model) with safe defaults", file=stdout)
    try:
        patch_fn(model)
    except Exception as exc:
        print(f"[error] Failed to patch model. Details: {exc}", file=stderr)
        return 1

    _print_patch_summary(model, out=stdout)

    print(
        f"[generate] prompt={args.prompt!r} max_tokens={args.max_tokens}",
        file=stdout,
    )
    try:
        generated = mlx_lm_module.generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:
        print(f"[error] Generation failed. Details: {exc}", file=stderr)
        return 1

    print("[output]", file=stdout)
    print(str(generated), file=stdout)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
