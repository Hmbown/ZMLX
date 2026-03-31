#!/usr/bin/env python3
"""ZMLX-patched OpenAI-compatible server.

Drop-in replacement for ``ollama serve`` that uses mlx-lm's HTTP server
with ZMLX kernel patches applied to every loaded model.  Any client that
speaks the OpenAI chat/completions API (curl, Open WebUI, Continue,
LangChain, etc.) can point at this server.

Usage::

    # Serve Qwen3.5-35B-A3B with ZMLX patches on port 8080 (default)
    python -m integrations.ollama_compat.serve \
        --model /path/to/Qwen3.5-35B-A3B-MLX-4bit

    # Or from the repo root:
    python integrations/ollama_compat/serve.py \
        --model Qwen/Qwen3.5-35B-A3B-MLX-4bit \
        --port 11434  # same port as Ollama

Then query it::

    curl http://localhost:11434/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model": "default", "messages": [{"role": "user", "content": "Hi"}]}'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import mlx.core as mx

# ---------------------------------------------------------------------------
# ZMLX-patched ModelProvider
# ---------------------------------------------------------------------------

from mlx_lm.server import ModelProvider, run


class ZMLXModelProvider(ModelProvider):
    """ModelProvider that applies zmlx.patch() after every model load."""

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        model, tokenizer = super().load(model_path, adapter_path, draft_model_path)

        # Only patch if not already patched (check for marker attribute)
        if getattr(model, "_zmlx_patched", False):
            return model, tokenizer

        try:
            from zmlx.patch import patch

            t0 = time.perf_counter()
            patch(model)
            dt = time.perf_counter() - t0

            # patch() attaches _zmlx_patch_result to the model
            result = getattr(model, "_zmlx_patch_result", None)
            if result and result.patched_count > 0:
                logging.info(
                    "ZMLX: patched %d modules (%s) in %.2fs — %s",
                    result.patched_count,
                    ", ".join(
                        f"{k}={v}" for k, v in result.pattern_counts.items()
                    ),
                    dt,
                    result.estimated_speedup or "no estimate",
                )
            else:
                logging.info("ZMLX: no patterns matched for this model")

            # Mark as patched so we don't re-patch on cache hit
            model._zmlx_patched = True

        except Exception as exc:
            logging.warning("ZMLX: patch failed, running unpatched — %s", exc)

        return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="ZMLX-patched OpenAI-compatible server (Ollama drop-in)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID or local path to MLX model weights",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter path",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind (default: 8080)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in tokenizer",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Override chat template",
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling (default: 0, disabled)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Min-p sampling (default: 0.0, disabled)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Default max tokens (default: 4096)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        default="{}",
        help="JSON args for chat template (e.g. '{\"enable_thinking\":false}')",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    # mlx-lm server expects these but we set sensible defaults
    parser.add_argument("--draft-model", type=str, default=None)
    parser.add_argument("--num-draft-tokens", type=int, default=3)
    parser.add_argument("--decode-concurrency", type=int, default=32)
    parser.add_argument("--prompt-concurrency", type=int, default=8)
    parser.add_argument("--pipeline", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)

    logging.info("Starting ZMLX server on %s:%d", args.host, args.port)
    logging.info("Model: %s", args.model)
    logging.info("ZMLX patches will be applied automatically on model load")

    provider = ZMLXModelProvider(args)
    run(args.host, args.port, provider)


if __name__ == "__main__":
    main()
