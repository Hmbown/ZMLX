from __future__ import annotations

"""SKV helpers for MLA-style caches (GLM-4.7-Flash).

This module is explicitly SKV-labeled and keeps RoPE slices separate:
only the latent KV portion is compressed.
"""

from typing import Any

import mlx.core as mx
import numpy as np

from ..kernels.skv import (
    skv_compressed_attention,
    skv_fused_dequantize_unproject,
    skv_fused_project_quantize,
)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


class SKVMLALatentCacheRuntime:
    """Runtime cache state for GLM MLA latents with SKV compression.

    The persistent cache stores:
      - RoPE slice uncompressed: ``k_rope`` (required for positional fidelity)
      - Latent slice compressed once warmup is reached

    The latent dense cache is held only before basis calibration and is dropped
    after transitioning to compressed mode.
    """

    def __init__(
        self,
        *,
        kv_lora_rank: int,
        rope_dim: int,
        rank: int,
        bits: int,
        group_size: int,
        warmup_tokens: int,
    ) -> None:
        _require(kv_lora_rank > 1, "SKVMLALatentCacheRuntime: kv_lora_rank must be > 1")
        _require(0 < rank < kv_lora_rank, "SKVMLALatentCacheRuntime: rank must be in (0, kv_lora_rank)")
        _require(bits in (2, 4, 8), "SKVMLALatentCacheRuntime: bits must be one of {2,4,8}")
        _require(group_size > 0, "SKVMLALatentCacheRuntime: group_size must be > 0")

        self.kv_lora_rank = int(kv_lora_rank)
        self.rope_dim = int(rope_dim)
        self.rank = int(rank)
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.warmup_tokens = int(max(1, warmup_tokens))

        self.offset = 0
        self.basis: Any | None = None
        self._compressed_chunks: list[dict[str, Any]] = []
        self._merged_state_cache: dict[str, Any] | None = None
        self.latent_dense: Any | None = None  # (B,1,S,D) before compression
        self.k_rope_dense: Any | None = None  # (B,1,S,rope_dim)

    def reset(self) -> None:
        self.offset = 0
        self.basis = None
        self._compressed_chunks = []
        self._merged_state_cache = None
        self.latent_dense = None
        self.k_rope_dense = None

    def ready(self) -> bool:
        return self.basis is not None and len(self._compressed_chunks) > 0

    @property
    def compressed_chunks(self) -> list[dict[str, Any]]:
        return self._compressed_chunks

    @property
    def compressed_state(self) -> dict[str, Any] | None:
        """Merged compressed state for compatibility APIs.

        Runtime decode paths should prefer iterating ``compressed_chunks`` to
        avoid unnecessary concatenation.
        """
        if not self._compressed_chunks:
            return None
        if len(self._compressed_chunks) == 1:
            return self._compressed_chunks[0]
        if self._merged_state_cache is not None:
            return self._merged_state_cache

        first = self._compressed_chunks[0]
        q_data = mx.concatenate([c["q_data"] for c in self._compressed_chunks], axis=0)
        scales = mx.concatenate([c["scales"] for c in self._compressed_chunks], axis=0)
        zeros = mx.concatenate([c["zeros"] for c in self._compressed_chunks], axis=0)
        total_s = sum(int(c["shape"][0]) for c in self._compressed_chunks)
        merged = {
            "q_data": q_data,
            "scales": scales,
            "zeros": zeros,
            "shape": (total_s, int(first["shape"][1]), int(first["shape"][2])),
            "bits": int(first["bits"]),
            "group_size": int(first["group_size"]),
            "batch": 1,
        }
        self._merged_state_cache = merged
        return merged

    def _append_seq(self, current: Any | None, delta: Any) -> Any:
        if current is None:
            return delta
        return mx.concatenate([current, delta], axis=2)

    def _maybe_build_basis_and_compress(self) -> None:
        if self.ready() or self.latent_dense is None:
            return
        seq_len = int(self.latent_dense.shape[2])
        if seq_len < self.warmup_tokens:
            return

        latent3 = mx.transpose(self.latent_dense, axes=(2, 1, 0, 3)).reshape(
            seq_len, 1, self.kv_lora_rank
        )
        self.basis = skv_compute_basis(latent3, self.rank)
        first_chunk = skv_compress_glm_latent(
            self.latent_dense,
            self.basis,
            bits=self.bits,
            group_size=self.group_size,
        )
        self._compressed_chunks = [first_chunk]
        self._merged_state_cache = None
        # Drop dense latent storage once compressed mode starts.
        self.latent_dense = None

    def ingest(self, keys_step: Any) -> None:
        """Append a new key chunk ``(B,1,L,kv_lora_rank+rope_dim)``."""
        latent, k_rope = skv_split_glm_keys(
            keys_step,
            kv_lora_rank=self.kv_lora_rank,
            rope_dim=self.rope_dim,
        )
        B, H, L, _ = map(int, latent.shape)
        _require(B == 1 and H == 1, "SKVMLALatentCacheRuntime: currently supports batch=1, kv_heads=1")

        self.k_rope_dense = self._append_seq(self.k_rope_dense, k_rope)

        if not self._compressed_chunks:
            self.latent_dense = self._append_seq(self.latent_dense, latent)
            self.offset += L
            self._maybe_build_basis_and_compress()
            return

        _require(self.basis is not None, "SKVMLALatentCacheRuntime: missing basis in compressed mode")

        # Fast append path: compress only incoming chunk and append.
        # This requires chunk boundaries to align with quantization groups.
        n_chunk = int(latent.shape[2]) * self.rank
        if n_chunk % self.group_size == 0:
            next_chunk = skv_compress_glm_latent(
                latent,
                self.basis,
                bits=self.bits,
                group_size=self.group_size,
            )
            self._compressed_chunks.append(next_chunk)
            self._merged_state_cache = None
            self.offset += L
            return

        # Fallback path when chunk/group alignment is not guaranteed.
        prev_latent = self.dense_latent_b1hld()
        latent_all = mx.concatenate([prev_latent, latent], axis=2)
        merged = skv_compress_glm_latent(
            latent_all,
            self.basis,
            bits=self.bits,
            group_size=self.group_size,
        )
        self._compressed_chunks = [merged]
        self._merged_state_cache = None
        self.offset += L

    def dense_latent_b1hld(self) -> Any:
        if self._compressed_chunks:
            _require(self.basis is not None, "SKVMLALatentCacheRuntime: missing basis")
            parts = [
                skv_decompress_glm_latent(state, self.basis)
                for state in self._compressed_chunks
            ]
            if len(parts) == 1:
                return parts[0]
            return mx.concatenate(parts, axis=2)
        _require(self.latent_dense is not None, "SKVMLALatentCacheRuntime: no latent state available")
        return self.latent_dense

    def dense_latent_shd(self) -> Any:
        latent = self.dense_latent_b1hld()
        S = int(latent.shape[2])
        return mx.transpose(latent, axes=(2, 1, 0, 3)).reshape(S, 1, self.kv_lora_rank)

    def rope_shd(self) -> Any:
        _require(self.k_rope_dense is not None, "SKVMLALatentCacheRuntime: no rope state available")
        S = int(self.k_rope_dense.shape[2])
        return mx.transpose(self.k_rope_dense, axes=(2, 1, 0, 3)).reshape(S, 1, self.rope_dim)

    def materialize_keys_values(self) -> tuple[Any, Any]:
        latent = self.dense_latent_b1hld()
        _require(self.k_rope_dense is not None, "SKVMLALatentCacheRuntime: no rope state available")
        keys = mx.concatenate([latent, self.k_rope_dense], axis=-1)
        values = latent
        return keys, values


def skv_dequantize_rank_chunk(
    state: dict[str, Any],
    *,
    compute_dtype: Any = mx.float32,
) -> Any:
    """Dequantize projected rank coefficients without unprojection.

    Returns:
        ``(seq, heads, rank)`` in ``compute_dtype``.
    """
    q_data = state["q_data"].astype(compute_dtype)
    scales = state["scales"].astype(compute_dtype)
    zeros = state["zeros"].astype(compute_dtype)
    group_size = int(state["group_size"])
    n = int(q_data.size)
    group_ids = mx.arange(n, dtype=mx.int32) // group_size
    flat = q_data.reshape(-1) * scales[group_ids] + zeros[group_ids]
    S, H, R = map(int, state["shape"])
    return flat.reshape(S, H, R)


def skv_split_glm_keys(
    keys: Any,
    *,
    kv_lora_rank: int = 512,
    rope_dim: int = 64,
) -> tuple[Any, Any]:
    """Split GLM key cache into ``(latent, k_rope)``.

    Args:
        keys: ``(B, 1, seq, kv_lora_rank + rope_dim)``.
    """
    _require(keys.ndim == 4, "skv_split_glm_keys: keys must be rank-4")
    _require(int(keys.shape[-1]) == int(kv_lora_rank + rope_dim), "skv_split_glm_keys: trailing dim mismatch")
    latent = keys[..., :kv_lora_rank]
    k_rope = keys[..., kv_lora_rank:]
    return latent, k_rope


def skv_compute_basis(
    kv: Any,
    rank: int,
) -> Any:
    """Compute per-head PCA basis for ``kv`` using NumPy eigendecomposition.

    Args:
        kv: ``(seq, heads, dim)``.
        rank: target rank.
    Returns:
        ``(heads, dim, rank)``
    """
    _require(kv.ndim == 3, "skv_compute_basis: kv must be rank-3")
    seq, heads, dim = map(int, kv.shape)
    _require(0 < rank < dim, "skv_compute_basis: rank must be in (0, dim)")

    out = []
    for h in range(heads):
        x = np.array(kv[:, h, :].astype(mx.float32))
        cov = (x.T @ x) / max(1, seq)
        _, evecs = np.linalg.eigh(cov)
        out.append(mx.array(evecs[:, -rank:].astype(np.float32)))
    return mx.stack(out).astype(kv.dtype)


def skv_compress_glm_latent(
    latent: Any,
    basis: Any,
    *,
    bits: int = 4,
    group_size: int = 32,
) -> dict[str, Any]:
    """Compress GLM latent cache slice.

    Args:
        latent: ``(B, 1, seq, kv_lora_rank)``.
        basis: ``(1, kv_lora_rank, rank)``.
    """
    _require(latent.ndim == 4, "skv_compress_glm_latent: latent must be rank-4")
    B, H, S, D = map(int, latent.shape)
    _require(B == 1, "skv_compress_glm_latent: currently supports batch=1")
    _require(H == 1, "skv_compress_glm_latent: expected MLA keys with one KV head")
    kv = mx.transpose(latent, axes=(2, 1, 0, 3)).reshape(S, H, D)
    state = skv_fused_project_quantize(kv, basis, bits=bits, group_size=group_size)
    state["batch"] = B
    return state


def skv_decompress_glm_latent(
    state: dict[str, Any],
    basis: Any,
) -> Any:
    """Decompress GLM latent cache slice back to ``(B,1,seq,kv_lora_rank)``."""
    latent_hat = skv_fused_dequantize_unproject(state, basis)
    S, H, D = map(int, latent_hat.shape)
    B = int(state.get("batch", 1))
    _require(B == 1, "skv_decompress_glm_latent: currently supports batch=1")
    out = latent_hat.reshape(S, H, D, B)
    out = mx.transpose(out, axes=(3, 1, 0, 2))
    return out


def skv_reconstruct_glm_keys(
    state: dict[str, Any],
    basis: Any,
    k_rope: Any,
) -> Any:
    """Rebuild full GLM keys as ``concat(latent_hat, k_rope)``."""
    latent_hat = skv_decompress_glm_latent(state, basis)
    return mx.concatenate([latent_hat, k_rope], axis=-1)


def skv_project_glm_queries_to_rank(
    q_nope_embed: Any,
    basis: Any,
) -> Any:
    """Project GLM no-PE queries into SKV rank space.

    Args:
        q_nope_embed: ``(B, heads, q_len, kv_lora_rank)``
        basis: ``(1, kv_lora_rank, rank)``
    Returns:
        ``(q_len * B, heads, rank)``
    """
    _require(q_nope_embed.ndim == 4, "skv_project_glm_queries_to_rank: q_nope_embed must be rank-4")
    _require(int(basis.shape[0]) == 1, "skv_project_glm_queries_to_rank: basis must have one KV head")
    b0 = basis[0]
    q_rank = mx.einsum("bhqd,dr->bhqr", q_nope_embed, b0)
    B, H, Q, R = map(int, q_rank.shape)
    q_rank = mx.transpose(q_rank, axes=(2, 0, 1, 3)).reshape(Q * B, H, R)
    return q_rank


def skv_glm_compressed_attention_scores(
    q_nope_embed: Any,
    k_state: dict[str, Any],
    basis: Any,
    *,
    num_heads: int,
    scale: float,
) -> Any:
    """Strategy-B score path for GLM: attention in rank space.

    Notes:
        GLM's no-PE query branch already maps into ``kv_lora_rank``. Projecting
        that query with the same SKV basis enables direct score computation
        against compressed latent keys.
    """
    q_rank = skv_project_glm_queries_to_rank(q_nope_embed, basis)
    return skv_compressed_attention(
        q_rank,
        k_state,
        num_heads=num_heads,
        num_kv_heads=1,
        scale=scale,
    )


__all__ = [
    "SKVMLALatentCacheRuntime",
    "skv_dequantize_rank_chunk",
    "skv_split_glm_keys",
    "skv_compute_basis",
    "skv_compress_glm_latent",
    "skv_decompress_glm_latent",
    "skv_reconstruct_glm_keys",
    "skv_project_glm_queries_to_rank",
    "skv_glm_compressed_attention_scores",
]
