import mlx.core as mx

from zmlx.kvtc.skv_mla import (
    SKVMLALatentCacheRuntime,
    skv_compress_glm_latent,
    skv_compute_basis,
    skv_decompress_glm_latent,
    skv_dequantize_rank_chunk,
    skv_glm_compressed_attention_scores,
    skv_reconstruct_glm_keys,
    skv_split_glm_keys,
)


def test_skv_split_glm_keys_shapes():
    keys = mx.random.normal((1, 1, 17, 576)).astype(mx.float32)
    latent, rope = skv_split_glm_keys(keys, kv_lora_rank=512, rope_dim=64)
    assert latent.shape == (1, 1, 17, 512)
    assert rope.shape == (1, 1, 17, 64)


def test_skv_glm_latent_roundtrip_shape_and_error():
    mx.random.seed(0)
    keys = mx.random.normal((1, 1, 33, 576)).astype(mx.float32)
    latent, rope = skv_split_glm_keys(keys, kv_lora_rank=512, rope_dim=64)
    latent3 = mx.transpose(latent, axes=(2, 1, 0, 3)).reshape(33, 1, 512)
    basis = skv_compute_basis(latent3, rank=128)

    state = skv_compress_glm_latent(latent, basis, bits=4, group_size=32)
    latent_hat = skv_decompress_glm_latent(state, basis)
    keys_hat = skv_reconstruct_glm_keys(state, basis, rope)
    mx.eval(latent_hat, keys_hat)

    assert latent_hat.shape == latent.shape
    assert keys_hat.shape == keys.shape
    mse = float(mx.mean((latent_hat - latent) ** 2).item())
    assert mse < 0.2


def test_skv_mla_runtime_transitions_to_compressed_storage():
    mx.random.seed(7)
    runtime = SKVMLALatentCacheRuntime(
        kv_lora_rank=64,
        rope_dim=8,
        rank=16,
        bits=4,
        group_size=16,
        warmup_tokens=8,
    )

    keys_a = mx.random.normal((1, 1, 5, 72)).astype(mx.float32)
    keys_b = mx.random.normal((1, 1, 7, 72)).astype(mx.float32)
    latent_a, _ = skv_split_glm_keys(keys_a, kv_lora_rank=64, rope_dim=8)
    latent_b, _ = skv_split_glm_keys(keys_b, kv_lora_rank=64, rope_dim=8)
    latent_ref = mx.concatenate([latent_a, latent_b], axis=2)

    runtime.ingest(keys_a)
    assert not runtime.ready()
    assert int(runtime.offset) == 5

    runtime.ingest(keys_b)
    assert runtime.ready()
    assert runtime.latent_dense is None
    assert int(runtime.offset) == 12

    keys_dense, values_dense = runtime.materialize_keys_values()
    latent_hat = values_dense
    mx.eval(keys_dense, latent_hat)

    assert keys_dense.shape == (1, 1, 12, 72)
    assert values_dense.shape == (1, 1, 12, 64)
    mse = float(mx.mean((latent_hat - latent_ref) ** 2).item())
    assert mse < 0.5


def test_skv_mla_strategy_b_scores_track_dense_scores():
    mx.random.seed(11)
    num_heads = 4
    kv_lora_rank = 32
    rope_dim = 8
    scale = (kv_lora_rank + rope_dim) ** -0.5

    runtime = SKVMLALatentCacheRuntime(
        kv_lora_rank=kv_lora_rank,
        rope_dim=rope_dim,
        rank=8,
        bits=4,
        group_size=8,
        warmup_tokens=4,
    )

    keys = mx.random.normal((1, 1, 10, kv_lora_rank + rope_dim)).astype(mx.float32)
    runtime.ingest(keys)
    assert runtime.ready()

    q_nope = mx.random.normal((1, num_heads, 1, kv_lora_rank)).astype(mx.float32)
    q_pe = mx.random.normal((1, num_heads, 1, rope_dim)).astype(mx.float32)

    score_nope = skv_glm_compressed_attention_scores(
        q_nope,
        runtime.compressed_state,
        runtime.basis,
        num_heads=num_heads,
        scale=scale,
    )
    q_pe_shd = mx.transpose(q_pe, axes=(2, 1, 0, 3)).reshape(1, num_heads, rope_dim)
    k_rope_shd = runtime.rope_shd()
    k_rope_heads = mx.repeat(k_rope_shd, repeats=num_heads, axis=1)
    score_rope = mx.einsum("qhd,khd->hqk", q_pe_shd, k_rope_heads) * scale
    score_total = score_nope + score_rope

    latent_shd = runtime.dense_latent_shd()
    q_nope_shd = mx.transpose(q_nope, axes=(2, 1, 0, 3)).reshape(1, num_heads, kv_lora_rank)
    keys_dense_heads = mx.repeat(
        mx.concatenate([latent_shd, k_rope_shd], axis=-1),
        repeats=num_heads,
        axis=1,
    )
    q_full_shd = mx.concatenate([q_nope_shd, q_pe_shd], axis=-1)
    score_ref = mx.einsum("qhd,khd->hqk", q_full_shd, keys_dense_heads) * scale
    mx.eval(score_total, score_ref)

    assert mx.allclose(score_total, score_ref, atol=2e-1)


def test_skv_mla_runtime_appends_compressed_chunks():
    mx.random.seed(19)
    runtime = SKVMLALatentCacheRuntime(
        kv_lora_rank=32,
        rope_dim=8,
        rank=16,
        bits=4,
        group_size=8,
        warmup_tokens=4,
    )

    k0 = mx.random.normal((1, 1, 4, 40)).astype(mx.float32)
    k1 = mx.random.normal((1, 1, 1, 40)).astype(mx.float32)
    runtime.ingest(k0)
    runtime.ingest(k1)

    assert runtime.ready()
    assert len(runtime.compressed_chunks) == 2
    merged = runtime.compressed_state
    assert merged is not None
    assert tuple(merged["shape"]) == (5, 1, 16)

    z_tail = skv_dequantize_rank_chunk(runtime.compressed_chunks[-1])
    assert z_tail.shape == (1, 1, 16)
