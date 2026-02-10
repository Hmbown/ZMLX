"""Dump KV cache from mlx-lm inference to NPZ for offline calibration.

Usage::

    python -m zmlx.kvtc dump --model mlx-community/LFM2-8B-A1B-4bit \\
        --prompt "The quick brown fox" --max-tokens 512 --out cache.npz

The resulting NPZ contains arrays ``k`` and ``v`` with shape
``(layers, batch, heads, seq, head_dim)``.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Run mlx-lm inference and save KV cache to NPZ.",
    )
    ap.add_argument("--model", required=True, help="HuggingFace model path or local dir")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True, help="Output .npz file path")
    ap.add_argument(
        "--preset", default=None,
        help="KVTC preset name for automatic adapter selection",
    )
    args = ap.parse_args(argv)

    try:
        import numpy as np
    except ModuleNotFoundError:
        print("KVTC requires numpy. Install with: pip install zmlx[kvtc]", file=sys.stderr)
        sys.exit(1)

    try:
        import mlx.core as mx  # noqa: F401
        import mlx_lm
    except ModuleNotFoundError:
        print(
            "dump requires mlx and mlx-lm. "
            "Install with: pip install mlx mlx-lm",
            file=sys.stderr,
        )
        sys.exit(1)

    from .adapters import GLMMlaCacheAdapter, MLXLMCacheAdapter
    from .presets import model_preset
    from .utils import to_numpy

    print(f"Loading model: {args.model}")
    loaded = mlx_lm.load(args.model)
    model = loaded[0]
    tokenizer = loaded[1]

    tokens = tokenizer.encode(args.prompt)
    print(f"Prompt tokens: {len(tokens)}, generating up to {args.max_tokens} more...")

    result = mlx_lm.generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
    )
    print(f"Generated {len(result.split()) - len(args.prompt.split())} tokens (approx)")

    # Extract cache — mlx-lm exposes cache on the model after generation
    cache = getattr(model, "cache", None)
    if cache is None:
        print("Could not access model.cache after generation.", file=sys.stderr)
        sys.exit(1)

    # Select adapter
    adapter: GLMMlaCacheAdapter | MLXLMCacheAdapter
    if args.preset:
        preset = model_preset(args.preset)
        if preset.mode == "single_stream":
            adapter = GLMMlaCacheAdapter()
        else:
            adapter = MLXLMCacheAdapter()
    else:
        adapter = MLXLMCacheAdapter()

    k_layers, v_layers = adapter.extract_kv(cache)

    # Stack into (L, B, H, T, D)
    k_arr = np.stack([to_numpy(k) for k in k_layers], axis=0)
    v_arr = np.stack([to_numpy(v) for v in v_layers], axis=0)

    print(f"Saving cache to {args.out}: k={k_arr.shape}, v={v_arr.shape}")
    np.savez(args.out, k=k_arr, v=v_arr)
    print("Done.")


if __name__ == "__main__":
    main()
