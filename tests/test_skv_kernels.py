import mlx.core as mx
import numpy as np

from zmlx.kernels import skv


def _group_quant_ref(x: np.ndarray, bits: int, group_size: int):
    flat = x.reshape(-1)
    n = flat.shape[0]
    ng = (n + group_size - 1) // group_size
    q = np.zeros_like(flat, dtype=np.float32)
    scales = np.zeros((ng,), dtype=np.float32)
    zeros = np.zeros((ng,), dtype=np.float32)
    qmax = float((1 << bits) - 1)
    for g in range(ng):
        st = g * group_size
        ed = min(st + group_size, n)
        seg = flat[st:ed]
        vmin = float(seg.min())
        vmax = float(seg.max())
        scale = (vmax - vmin) / qmax if vmax > vmin else 1.0
        qseg = np.clip(np.round((seg - vmin) / scale), 0, qmax)
        q[st:ed] = qseg
        scales[g] = scale
        zeros[g] = vmin
    return q.reshape(x.shape), scales, zeros


def test_skv_project_quantize_roundtrip_matches_reference():
    mx.random.seed(0)
    seq, heads, dim, rank = 9, 1, 16, 6
    bits, group_size = 4, 8

    kv = mx.random.normal((seq, heads, dim)).astype(mx.float32)
    x = np.array(kv)
    cov = (x[:, 0, :].T @ x[:, 0, :]) / x.shape[0]
    _, evecs = np.linalg.eigh(cov)
    basis_np = evecs[:, -rank:].astype(np.float32)
    basis = mx.array(basis_np).reshape(1, dim, rank)

    state = skv.skv_fused_project_quantize(
        kv, basis, bits=bits, group_size=group_size, compute_dtype=mx.float32
    )
    kv_hat = skv.skv_fused_dequantize_unproject(state, basis, compute_dtype=mx.float32)
    mx.eval(kv_hat, state["q_data"], state["scales"], state["zeros"])

    proj_ref = np.einsum("shd,hdr->shr", np.array(kv), np.array(basis))
    q_ref, s_ref, z_ref = _group_quant_ref(proj_ref, bits, group_size)
    proj_hat_ref = (q_ref.reshape(-1) * np.repeat(s_ref, group_size)[: q_ref.size]) + np.repeat(z_ref, group_size)[: q_ref.size]
    proj_hat_ref = proj_hat_ref.reshape(proj_ref.shape)
    kv_hat_ref = np.einsum("shr,hdr->shd", proj_hat_ref, np.array(basis))

    assert np.allclose(np.array(state["q_data"]), q_ref, atol=1e-3)
    assert np.allclose(np.array(state["scales"]), s_ref, atol=1e-4)
    assert np.allclose(np.array(state["zeros"]), z_ref, atol=1e-4)
    assert np.allclose(np.array(kv_hat), kv_hat_ref, atol=2e-3)


def test_skv_compressed_attention_matches_dense_dequant():
    mx.random.seed(1)
    q_len, kv_len, heads, rank = 7, 13, 4, 8
    bits, group_size = 4, 8
    scale = 0.125

    q_rank = mx.random.normal((q_len, heads, rank)).astype(mx.float32)
    kv_rank = mx.random.normal((kv_len, 1, rank)).astype(mx.float32)
    basis = mx.eye(rank, dtype=mx.float32).reshape(1, rank, rank)

    state = skv.skv_fused_project_quantize(
        kv_rank, basis, bits=bits, group_size=group_size, compute_dtype=mx.float32
    )
    kv_rank_hat = skv.skv_fused_dequantize_unproject(state, basis, compute_dtype=mx.float32)
    kv_rank_hat_h = mx.repeat(kv_rank_hat, repeats=heads, axis=1)

    scores_ref = mx.einsum("qhr,khr->hqk", q_rank, kv_rank_hat_h) * scale
    scores = skv.skv_compressed_attention(
        q_rank,
        state,
        num_heads=heads,
        num_kv_heads=1,
        scale=scale,
        compute_dtype=mx.float32,
    )
    mx.eval(scores, scores_ref)

    assert mx.allclose(scores, scores_ref, atol=2e-3)
