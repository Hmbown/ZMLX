from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "KVTC requires numpy. Install with: pip install zmlx[kvtc]"
    ) from e

from ..plan import GroupSpec, QuantPlan


@dataclass(frozen=True)
class PCABasis:
    mu: np.ndarray  # (p,)
    V: np.ndarray   # (p, r)  right singular vectors (PCA basis), columns are PCs


def compute_pca_basis(
    C: np.ndarray,
    r: int | None = None,
    dtype=np.float32,
) -> PCABasis:
    """Compute PCA basis via SVD on centered data.

    C: (n, p) calibration matrix.
    r: optional rank truncation.
    """
    C = np.asarray(C, dtype=dtype)
    mu = C.mean(axis=0, keepdims=False)
    X = C - mu
    _U, _S, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    if r is not None:
        V = V[:, :r]
    return PCABasis(mu=mu.astype(dtype, copy=False), V=V.astype(dtype, copy=False))


def _simulate_quantization_uniform(block: np.ndarray, bits: int) -> np.ndarray:
    """Simulate per-sample uniform quantization with per-block shift+scale."""
    if bits <= 0:
        return np.zeros_like(block)

    levels = 1 << bits
    vmin = block.min(axis=1, keepdims=True)
    vmax = block.max(axis=1, keepdims=True)
    span = vmax - vmin
    scale = np.where(span > 0, span / (levels - 1), 1.0).astype(block.dtype, copy=False)
    q = np.rint((block - vmin) / scale).astype(np.int32)
    q = np.clip(q, 0, levels - 1).astype(np.float32)
    deq = q * scale + vmin
    return deq.astype(block.dtype, copy=False)


def _bits_for_qtype(qtype: str) -> int:
    return {"none": 0, "int2": 2, "int4": 4, "fp8": 8}[qtype]


def _used_bits(block_size: int, bits: int) -> int:
    if bits <= 0:
        return 0
    return block_size * bits + 32  # 16-bit shift + 16-bit scale


def calibrate_dp_plan(
    C: np.ndarray,
    max_bit_budget: int,
    allowed_block_sizes: list[int] | None = None,
    qtypes: list[str] | None = None,
    r: int | None = None,
    dtype=np.float32,
) -> tuple[PCABasis, QuantPlan]:
    """Compute PCA basis and a DP quantization plan.

    Args:
      C: (batch, p) calibration matrix.
      max_bit_budget: maximum bits per token (payload + group overhead).
      allowed_block_sizes: group sizes to consider.
      qtypes: quantization types to consider.
      r: optional PCA rank truncation.
    """
    if allowed_block_sizes is None:
        allowed_block_sizes = [1, 16, 64, 256, 1024]
    if qtypes is None:
        qtypes = ["none", "int2", "int4", "fp8"]

    basis = compute_pca_basis(C, r=r, dtype=dtype)
    mu, V = basis.mu, basis.V

    P = (np.asarray(C, dtype=dtype) - mu) @ V
    _batch, num_features = P.shape

    initial_reconstruction_error = float(np.sum(P * P))

    best_error = np.full(
        (num_features + 1, max_bit_budget + 1), initial_reconstruction_error, dtype=np.float64
    )
    best_type = np.full((num_features + 1, max_bit_budget + 1), "none", dtype=object)
    best_block = np.zeros((num_features + 1, max_bit_budget + 1), dtype=np.int32)
    best_cost = np.zeros((num_features + 1, max_bit_budget + 1), dtype=np.int32)

    for i in range(1, num_features + 1):
        for block_size in allowed_block_sizes:
            if block_size > i:
                continue

            block = P[:, i - block_size : i]
            zero_err = float(np.sum(block * block))

            per_type = []
            for qt in qtypes:
                bits = _bits_for_qtype(qt)
                used = _used_bits(block_size, bits)
                deq = _simulate_quantization_uniform(block, bits)
                qerr = float(np.sum((block - deq) ** 2))
                error_change = -zero_err + qerr
                per_type.append((qt, used, error_change))

            for budget in range(0, max_bit_budget + 1):
                if budget > 0 and best_error[i, budget] > best_error[i, budget - 1]:
                    best_error[i, budget] = best_error[i, budget - 1]
                    best_type[i, budget] = best_type[i, budget - 1]
                    best_block[i, budget] = best_block[i, budget - 1]
                    best_cost[i, budget] = best_cost[i, budget - 1]

                for qt, used, error_change in per_type:
                    if used > budget:
                        continue
                    cand = error_change + best_error[i - block_size, budget - used]
                    if cand < best_error[i, budget]:
                        best_error[i, budget] = cand
                        best_type[i, budget] = qt
                        best_block[i, budget] = block_size
                        best_cost[i, budget] = used

    # Backtrack
    groups: list[GroupSpec] = []
    i = num_features
    budget = max_bit_budget
    while i > 0:
        bs = int(best_block[i, budget])
        qt = str(best_type[i, budget])
        used = int(best_cost[i, budget])

        if bs == 0:
            bs = 1
            qt = "none"
            used = 0

        groups.append(GroupSpec(size=bs, qtype=qt))  # type: ignore[arg-type]
        i -= bs
        budget = max(0, budget - used)

    groups.reverse()

    # Trim trailing zero-bit groups
    while groups and groups[-1].qtype == "none":
        groups.pop()

    plan = QuantPlan(groups=groups)

    # Trim V to used dimensionality
    r_used = plan.r()
    basis = PCABasis(mu=basis.mu, V=basis.V[:, :r_used].copy())

    return basis, plan


def save_calibration_dir(
    out_dir: str,
    key_basis: PCABasis,
    key_plan: QuantPlan,
    val_basis: PCABasis,
    val_plan: QuantPlan,
    meta: dict[str, Any] | None = None,
) -> None:
    """Save calibration artifacts to a directory."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "k_mu.npy"), key_basis.mu.astype(np.float16))
    np.save(os.path.join(out_dir, "k_V.npy"), key_basis.V.astype(np.float16))
    with open(os.path.join(out_dir, "k_plan.json"), "w", encoding="utf-8") as f:
        json.dump(key_plan.to_json(), f, indent=2)

    np.save(os.path.join(out_dir, "v_mu.npy"), val_basis.mu.astype(np.float16))
    np.save(os.path.join(out_dir, "v_V.npy"), val_basis.V.astype(np.float16))
    with open(os.path.join(out_dir, "v_plan.json"), "w", encoding="utf-8") as f:
        json.dump(val_plan.to_json(), f, indent=2)

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta or {}, f, indent=2)
