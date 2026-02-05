import numpy as np
import pytest


def _kk_sigmoid(x):
    import mlx.core as mx

    y = 1 / (1 + mx.exp(mx.abs(x)))
    return mx.where(x < 0, y, 1 - y)


def _reference_router_topk_sigmoid(
    logits,
    bias,
    *,
    k: int = 8,
    n_group: int = 1,
    topk_group: int = 1,
):
    """Pure-MLX reference with stable tie-break by lower expert index."""
    import mlx.core as mx

    if logits.ndim < 1:
        raise ValueError("logits must have rank >= 1")
    D = int(logits.shape[-1])
    if bias.ndim != 1 or int(bias.shape[0]) != D:
        raise ValueError(f"bias must have shape ({D},)")
    if k <= 0 or k > D:
        raise ValueError("invalid k")

    affinity = _kk_sigmoid(logits.astype(mx.float32))
    scores = affinity + bias.astype(mx.float32)

    # Optional: DeepSeek group selection (mask lowest-scoring groups to 0.0)
    if n_group > 1 and topk_group < n_group:
        if D % n_group != 0:
            raise ValueError("n_group must divide D")
        per_group = D // n_group
        grouped = mx.unflatten(scores, axis=-1, shape=(n_group, per_group))
        group_scores = mx.topk(grouped, 2, axis=-1).sum(axis=-1, keepdims=False)
        k_drop = n_group - topk_group
        drop_idx = mx.argpartition(group_scores, kth=k_drop - 1, axis=-1)[..., :k_drop]
        keep = mx.ones_like(group_scores)
        keep = mx.put_along_axis(
            keep,
            drop_idx,
            mx.array(0.0, dtype=keep.dtype),
            axis=-1,
        )
        grouped = grouped * keep[..., :, None]
        scores = mx.flatten(grouped, -2, -1)

    idx = mx.arange(D, dtype=mx.int32)
    neg_inf = mx.array(-float("inf"), dtype=scores.dtype)

    chosen_idx = []
    chosen_aff = []
    work = scores
    for _ in range(int(k)):
        max_val = mx.max(work, axis=-1, keepdims=True)
        mask = work == max_val
        min_idx = mx.min(mx.where(mask, idx, D), axis=-1)
        chosen_idx.append(min_idx)
        chosen_aff.append(
            mx.take_along_axis(affinity, min_idx[..., None], axis=-1)[..., 0]
        )
        work = mx.where(idx == min_idx[..., None], neg_inf, work)

    indices = mx.stack(chosen_idx, axis=-1).astype(mx.uint32)
    weights = mx.stack(chosen_aff, axis=-1)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    return weights, indices


@pytest.mark.metal
@pytest.mark.parametrize("d_experts", [256, 384])
def test_deepseek_router_topk_sigmoid_matches_reference(d_experts: int):
    import mlx.core as mx

    from zmlx.kernels.moe import deepseek_router_topk_sigmoid

    k = 8
    B = 3

    # Make selection robust to tiny sigmoid differences by using a large bias boost
    # for exactly K experts.
    logits_row = mx.linspace(-2.0, 2.0, d_experts, dtype=mx.float32)
    logits = mx.stack([logits_row + 0.1 * i for i in range(B)], axis=0)

    boost = [0, 7, 13, 64, 127, 128, d_experts - 2, d_experts - 1]
    bias_np = np.zeros((d_experts,), dtype=np.float32)
    bias_np[boost] = 10.0
    bias = mx.array(bias_np)

    w_fused, i_fused = deepseek_router_topk_sigmoid(logits, bias, k=k)
    w_ref, i_ref = _reference_router_topk_sigmoid(logits, bias, k=k)

    np.testing.assert_array_equal(np.array(i_fused.tolist()), np.array(i_ref.tolist()))
    np.testing.assert_allclose(
        np.array(w_fused.tolist()),
        np.array(w_ref.tolist()),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.metal
@pytest.mark.parametrize("d_experts", [256, 384])
def test_deepseek_router_topk_sigmoid_stable_tiebreak_lower_index(d_experts: int):
    import mlx.core as mx

    from zmlx.kernels.moe import deepseek_router_topk_sigmoid

    logits = mx.zeros((1, d_experts), dtype=mx.float32)
    bias = mx.zeros((d_experts,), dtype=mx.float32)

    weights, indices = deepseek_router_topk_sigmoid(logits, bias, k=8)
    assert indices.tolist()[0] == list(range(8))
    np.testing.assert_allclose(
        np.array(weights.tolist())[0],
        np.full((8,), 1 / 8, dtype=np.float32),
        rtol=0,
        atol=0,
    )


@pytest.mark.metal
def test_deepseek_router_topk_sigmoid_group_select_matches_reference():
    import mlx.core as mx

    from zmlx.kernels.moe import deepseek_router_topk_sigmoid

    d_experts = 256
    k = 8
    n_group = 8
    topk_group = 4

    logits = mx.zeros((1, d_experts), dtype=mx.float32)

    # Bias boosts: 7 high-score experts in the kept groups (0..3),
    # plus 1 moderately high expert in a dropped group (7). Group selection
    # should mask the dropped group to 0.0, so the 8th expert comes from the
    # lowest-index remaining kept group.
    bias_np = np.zeros((d_experts,), dtype=np.float32)
    bias_np[[0, 1, 32, 33, 64, 65, 96]] = 10.0
    bias_np[224] = 9.0  # group 7 (dropped)
    bias = mx.array(bias_np)

    w_fused, i_fused = deepseek_router_topk_sigmoid(
        logits,
        bias,
        k=k,
        n_group=n_group,
        topk_group=topk_group,
    )

    w_ref, i_ref = _reference_router_topk_sigmoid(
        logits,
        bias,
        k=k,
        n_group=n_group,
        topk_group=topk_group,
    )

    # Compare after sorting by expert index (order doesn't matter for MoE sum).
    order_f = mx.argsort(i_fused.astype(mx.int32), axis=-1)
    order_r = mx.argsort(i_ref.astype(mx.int32), axis=-1)
    i_f = mx.take_along_axis(i_fused, order_f, axis=-1)
    w_f = mx.take_along_axis(w_fused, order_f, axis=-1)
    i_r = mx.take_along_axis(i_ref, order_r, axis=-1)
    w_r = mx.take_along_axis(w_ref, order_r, axis=-1)
    mx.eval(i_f, w_f, i_r, w_r)

    np.testing.assert_array_equal(np.array(i_f.tolist()), np.array(i_r.tolist()))
    np.testing.assert_allclose(
        np.array(w_f.tolist()),
        np.array(w_r.tolist()),
        rtol=1e-6,
        atol=1e-7,
    )
