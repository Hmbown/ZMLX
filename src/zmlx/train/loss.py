"""Fused loss functions backed by ZMLX Metal kernels."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def fused_cross_entropy(
    logits: Any,
    targets: Any,
    *,
    threadgroup: int = 256,
    compute_dtype: Any | None = None,
    reduction: str = "mean",
) -> Any:
    """Fused softmax + cross-entropy loss using ZMLX Metal kernel.

    Args:
        logits: (..., vocab_size) float array of unnormalized logits.
        targets: (...) integer array of target class indices.
        threadgroup: Metal threadgroup size.
        compute_dtype: Compute dtype for the kernel.
        reduction: "mean", "sum", or "none".

    Returns:
        Scalar loss (if reduction is mean/sum) or per-sample losses.
    """
    from zmlx.kernels.loss import softmax_cross_entropy

    per_sample = softmax_cross_entropy(
        logits,
        targets.astype(mx.uint32),
        threadgroup=threadgroup,
        compute_dtype=compute_dtype,
    )

    if reduction == "mean":
        return mx.mean(per_sample)
    elif reduction == "sum":
        return mx.sum(per_sample)
    return per_sample


def standard_cross_entropy(
    logits: Any,
    targets: Any,
    *,
    reduction: str = "mean",
) -> Any:
    """Standard cross-entropy using MLX ops (fallback).

    Uses log_softmax + nll_loss pattern.
    """
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Gather target log probs
    batch_shape = targets.shape
    flat_targets = targets.reshape(-1)
    flat_log_probs = log_probs.reshape(-1, logits.shape[-1])
    nll = -mx.take_along_axis(
        flat_log_probs,
        flat_targets[:, None].astype(mx.int32),
        axis=-1,
    ).squeeze(-1)
    nll = nll.reshape(batch_shape)

    if reduction == "mean":
        return mx.mean(nll)
    elif reduction == "sum":
        return mx.sum(nll)
    return nll
