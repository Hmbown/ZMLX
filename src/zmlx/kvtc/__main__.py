"""CLI dispatcher for KVTC subcommands.

Usage::

    python -m zmlx.kvtc presets       # list model presets
    python -m zmlx.kvtc dump ...      # dump KV cache to NPZ
    python -m zmlx.kvtc calibrate ... # calibrate from NPZ
"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m zmlx.kvtc {presets,dump,calibrate}")
        print()
        print("Subcommands:")
        print("  presets    List available model presets")
        print("  dump       Run mlx-lm inference and save KV cache to NPZ")
        print("  calibrate  Calibrate KVTC from NPZ cache dumps")
        sys.exit(1)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    if cmd == "presets":
        from .presets import list_presets

        for p in list_presets():
            print(f"  {p.name:12s}  {p.description}")
            print(f"{'':14s}  layers={p.layers}  kv_heads={p.kv_heads}  "
                  f"head_dim={p.head_dim}  mode={p.mode}")
            print(f"{'':14s}  rope: dim={p.rope.dim}  base={p.rope.base}  "
                  f"traditional={p.rope.traditional}  offset={p.rope.offset}")
            print()

    elif cmd == "dump":
        # Remove the subcommand from argv so argparse sees the right args
        sys.argv = [sys.argv[0] + " dump"] + rest
        from .dump import main as dump_main
        dump_main(rest)

    elif cmd == "calibrate":
        sys.argv = [sys.argv[0] + " calibrate"] + rest
        from .calibrate_cli import main as cal_main
        cal_main(rest)

    else:
        print(f"Unknown subcommand: {cmd}", file=sys.stderr)
        print("Usage: python -m zmlx.kvtc {presets,dump,calibrate}")
        sys.exit(1)


if __name__ == "__main__":
    main()
