from __future__ import annotations

import mlx.core as mx

from zmlx.kernels.rope import _rope_cos_sin, rope_concat_qk_decode_pos


def test_rope_concat_qk_decode_pos_matches_mx_fast_rope_float32():
    mx.random.seed(0)

    B = 1
    Hq = 3
    D_nope = 32
    D_rope = 64
    offset = 7
    base = 1_000_000.0
    scale = 1.0

    q_nope = mx.random.normal((B, Hq, 1, D_nope)).astype(mx.float32)
    q_rope = mx.random.normal((B, Hq, 1, D_rope)).astype(mx.float32)
    kv_nope = mx.random.normal((B, 1, 1, D_nope)).astype(mx.float32)
    k_rope = mx.random.normal((B, 1, 1, D_rope)).astype(mx.float32)

    # Precompute just enough for this offset.
    table_len = 1 << (offset + 1 - 1).bit_length()
    tables = _rope_cos_sin(table_len, D_rope, base, scale)
    cos = tables.cos[offset]
    sin = tables.sin[offset]

    q_out, k_out = rope_concat_qk_decode_pos(q_nope, q_rope, kv_nope, k_rope, cos, sin)

    q_rope_ref = mx.fast.rope(
        q_rope, D_rope, traditional=True, base=base, scale=scale, offset=offset
    )
    k_rope_ref = mx.fast.rope(
        k_rope, D_rope, traditional=True, base=base, scale=scale, offset=offset
    )
    q_ref = mx.concatenate([q_nope, q_rope_ref], axis=-1)
    k_ref = mx.concatenate([kv_nope, k_rope_ref], axis=-1)

    mx.eval(q_out, k_out, q_ref, k_ref)
    assert mx.array_equal(q_out, q_ref).item()
    assert mx.array_equal(k_out, k_ref).item()

