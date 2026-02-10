"""Calibrate KVTC from NPZ cache dumps.

Usage::

    python -m zmlx.kvtc calibrate --npz cache1.npz cache2.npz \\
        --out calibration/lfm2/ --preset lfm2

Or with manual RoPE config::

    python -m zmlx.kvtc calibrate --npz cache.npz --out cal/ \\
        --rope-dim 64 --rope-base 1000000
"""

from __future__ import annotations

import argparse
import os
import random
import sys

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "KVTC requires numpy. Install with: pip install zmlx[kvtc]"
    ) from e

from .calibration.dp_calibrate import calibrate_dp_plan, save_calibration_dir
from .rope import RotaryConfig, RotaryEmbedding


def _load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "k" not in data or "v" not in data:
        raise ValueError(f"{path} must contain arrays 'k' and 'v'")
    return data["k"], data["v"]


def _sample_positions(
    seq_len: int, s: int, w: int, n: int, rng: random.Random
) -> np.ndarray:
    lo = s
    hi = seq_len - w
    if hi <= lo:
        return np.zeros((0,), dtype=np.int64)
    avail = list(range(lo, hi))
    if n >= len(avail):
        return np.array(avail, dtype=np.int64)
    return np.array(rng.sample(avail, n), dtype=np.int64)


def _flatten_at_positions(
    kv: np.ndarray,
    positions: np.ndarray,
    rope: RotaryEmbedding | None,
    apply_rope: bool,
) -> np.ndarray:
    L, B, H, T, D = kv.shape
    if B != 1:
        raise ValueError("This tool assumes batch=1")
    n = positions.shape[0]
    parts = []
    for layer_idx in range(L):
        seg = kv[layer_idx, 0, :, positions, :].astype(np.float32)
        if apply_rope and rope is not None:
            seg = rope.apply(seg, positions, inverse=True)
        seg2 = np.transpose(seg, (1, 0, 2)).reshape(n, H * D)
        parts.append(seg2.astype(np.float16))
    return np.concatenate(parts, axis=1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate KVTC from NPZ cache dumps.",
    )
    ap.add_argument(
        "--npz", nargs="+", required=True,
        help="One or more .npz files containing k/v arrays",
    )
    ap.add_argument("--out", required=True, help="Output calibration directory")
    ap.add_argument("--samples-per-file", type=int, default=256)
    ap.add_argument("--s", type=int, default=4)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)

    # PCA / DP controls
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--max-bit-budget", type=int, default=None)
    ap.add_argument("--compression-ratio", type=float, default=None)

    # RoPE controls
    ap.add_argument("--rope-dim", type=int, default=None)
    ap.add_argument("--rope-base", type=float, default=10000.0)
    ap.add_argument("--rope-traditional", action="store_true")
    ap.add_argument("--rope-offset", type=int, default=0)
    ap.add_argument("--apply-rope-to-values", action="store_true")

    # Preset (overrides RoPE controls)
    ap.add_argument(
        "--preset", default=None,
        help="Model preset name (e.g. lfm2, qwen3, glm) for auto RoPE/dim config",
    )

    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    # Resolve RoPE config from preset or manual args
    rope_dim = args.rope_dim
    rope_base = args.rope_base
    rope_traditional = args.rope_traditional
    rope_offset = args.rope_offset
    mode = "dual_stream"

    if args.preset:
        from .presets import model_preset
        preset = model_preset(args.preset)
        rope_dim = preset.rope.dim
        rope_base = preset.rope.base
        rope_traditional = preset.rope.traditional
        rope_offset = preset.rope.offset
        mode = preset.mode
        print(f"Using preset '{preset.name}': {preset.description}")

    rope = None
    if rope_dim is not None:
        rope = RotaryEmbedding(RotaryConfig(
            dim=int(rope_dim),
            base=float(rope_base),
            traditional=rope_traditional,
            offset=rope_offset,
        ))

    k_mats = []
    v_mats = []
    p = None

    for path in args.npz:
        k, v = _load_npz(path)
        L, B, H, T, D = k.shape
        if B != 1:
            raise ValueError("This tool assumes batch=1")
        if p is None:
            p = L * H * D
        elif p != L * H * D:
            raise ValueError("All inputs must share L/H/D")

        pos = _sample_positions(T, args.s, args.w, args.samples_per_file, rng)
        if pos.size == 0:
            continue

        k_mats.append(_flatten_at_positions(k, pos, rope, apply_rope=True))

        if mode == "dual_stream":
            v_mats.append(
                _flatten_at_positions(v, pos, rope, apply_rope=bool(args.apply_rope_to_values))
            )

    if not k_mats:
        print("No calibration samples collected (check s/w vs seq_len)", file=sys.stderr)
        sys.exit(1)

    Ck = np.concatenate(k_mats, axis=0).astype(np.float32)

    if args.max_bit_budget is None:
        if args.compression_ratio is None:
            print("Provide --max-bit-budget or --compression-ratio", file=sys.stderr)
            sys.exit(1)
        if p is None:
            print("internal: p unknown", file=sys.stderr)
            sys.exit(1)
        baseline_bits = int(p * 16)
        args.max_bit_budget = int(baseline_bits / float(args.compression_ratio))

    print(f"Calibrating keys: max_bit_budget={args.max_bit_budget}, Ck shape={Ck.shape}")
    kbasis, kplan = calibrate_dp_plan(Ck, max_bit_budget=int(args.max_bit_budget), r=args.rank)

    if mode == "dual_stream" and v_mats:
        Cv = np.concatenate(v_mats, axis=0).astype(np.float32)
        print(f"Calibrating values: max_bit_budget={args.max_bit_budget}, Cv shape={Cv.shape}")
        vbasis, vplan = calibrate_dp_plan(
            Cv, max_bit_budget=int(args.max_bit_budget), r=args.rank
        )
    else:
        # single_stream: dummy V artifacts
        from .calibration.dp_calibrate import PCABasis
        from .plan import QuantPlan
        vbasis = PCABasis(mu=np.zeros(1, dtype=np.float32), V=np.zeros((1, 1), dtype=np.float32))
        vplan = QuantPlan(groups=[])
        print("Single-stream mode: V artifacts are dummy placeholders.")

    meta = {
        "s": int(args.s),
        "w": int(args.w),
        "mode": mode,
        "rope": None if rope_dim is None else {
            "dim": int(rope_dim),
            "base": float(rope_base),
            "traditional": rope_traditional,
            "offset": rope_offset,
        },
        "apply_rope_to_values": bool(args.apply_rope_to_values),
        "samples_per_file": int(args.samples_per_file),
        "npz_files": [os.path.basename(p) for p in args.npz],
    }

    save_calibration_dir(args.out, kbasis, kplan, vbasis, vplan, meta=meta)
    print(f"Saved calibration to {args.out}")
    print(f"Key plan r={kplan.r()} groups={len(kplan.groups)}")
    print(f"Val plan r={vplan.r()} groups={len(vplan.groups)}")


if __name__ == "__main__":
    main()
