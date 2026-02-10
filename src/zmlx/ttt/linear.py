"""TTT-Linear layer — pure MLX reference implementation.

The inner model is a linear map f(x; W, b) = xW + b. During inference,
W and b are updated each token via the gradient of an L2 reconstruction
loss, making the layer's "hidden state" a learned weight matrix that
compresses all past context.

Two code paths:
  - **prefill** (dual form): processes mini-batches of K tokens in parallel
    using a causal attention-like matmul formulation.
  - **decode** (primal form): processes one token at a time with explicit
    gradient accumulation and weight update.

All operations use standard MLX ops. A fused Metal kernel for the decode
path is in ``kernel.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TTTLinearConfig:
    """Configuration for a TTT-Linear layer."""

    hidden_size: int = 2048
    num_heads: int = 32
    head_dim: int = 64
    mini_batch_size: int = 16
    ttt_base_lr: float = 1.0
    rope_theta: float = 10000.0
    conv_kernel: int = 4
    eps: float = 1e-6


# ---------------------------------------------------------------------------
# Cache (carries W, b, grad accumulators across tokens)
# ---------------------------------------------------------------------------


class TTTCache:
    """Mutable state for a TTT-Linear layer during generation.

    Holds the inner model weights (W1, b1) and their gradient accumulators,
    plus the current sequence position.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        W1_init: mx.array,
        b1_init: mx.array,
    ):
        # Tile learned init across batch: [B*nh, f, f] and [B*nh, 1, f]
        nh = num_heads
        self.W1 = mx.broadcast_to(
            W1_init, (batch_size * nh, head_dim, head_dim)
        ).astype(mx.float32)
        self.b1 = mx.broadcast_to(
            b1_init, (batch_size * nh, 1, head_dim)
        ).astype(mx.float32)

        self.W1_grad = mx.zeros_like(self.W1)
        self.b1_grad = mx.zeros_like(self.b1)

        self.seq_offset: int = 0
        self.batch_size = batch_size
        self.num_heads = nh
        self.head_dim = head_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layernorm_fwd(x: mx.array, weight: mx.array, bias: mx.array, eps: float = 1e-6):
    """Forward LayerNorm, returning (output, x_hat, std) for backward."""
    mu = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    std = mx.sqrt(var + eps)
    x_hat = (x - mu) / std
    out = weight * x_hat + bias
    return out, x_hat, std


def _layernorm_bwd(
    grad_output: mx.array,
    x_hat: mx.array,
    std: mx.array,
    weight: mx.array,
):
    """Backward through LayerNorm given pre-computed forward values."""
    f = x_hat.shape[-1]
    dl_dx_hat = grad_output * weight
    term1 = f * dl_dx_hat
    term2 = mx.sum(dl_dx_hat, axis=-1, keepdims=True)
    term3 = x_hat * mx.sum(dl_dx_hat * x_hat, axis=-1, keepdims=True)
    return (term1 - term2 - term3) / (std * f)


# ---------------------------------------------------------------------------
# TTT-Linear Module
# ---------------------------------------------------------------------------


class TTTLinear(nn.Module):
    """TTT-Linear layer: replaces attention with a test-time-trained linear map.

    During inference, the inner model weights W1, b1 are updated each token
    via gradient descent on an L2 reconstruction loss. The update rule:

        Z1 = XK @ W1 + b1           (forward)
        loss = ||LN(Z1) - (XV - XK)||^2
        W1 -= eta * d(loss)/d(W1)   (update)
        output = XQ @ W1_updated + b1_updated   (predict)

    This is equivalent to linear attention when eta=1/(2*f) and W_0=0.
    """

    def __init__(self, config: TTTLinearConfig):
        super().__init__()
        self.config = config
        c = config

        # Fused Q/K/V + gate + learned LR projection
        # Output: [QK (2*D), Gate (D), V (D), lr_logits (nh)]
        proj_size = 3 * c.hidden_size + c.num_heads
        self.qkv_gate_lr_proj = nn.Linear(c.hidden_size, proj_size, bias=False)

        # Output projection
        self.o_proj = nn.Linear(c.hidden_size, c.hidden_size, bias=False)

        # Inner model init parameters (learned, per head)
        self.W1 = mx.random.normal((c.num_heads, c.head_dim, c.head_dim)) * 0.02
        self.b1 = mx.zeros((c.num_heads, 1, c.head_dim))

        # LayerNorm for inner TTT loss
        self.ttt_norm_weight = mx.ones((c.num_heads, 1, c.head_dim))
        self.ttt_norm_bias = mx.zeros((c.num_heads, 1, c.head_dim))

        # Post-norm (after TTT output, before gating)
        self.post_norm = nn.LayerNorm(c.hidden_size)

        # Learned LR bias
        self.lr_bias = mx.zeros((1, 1, c.num_heads))

        # Token index bias (learned, per mini-batch position)
        self.token_idx_bias = mx.zeros((c.mini_batch_size,))

    def create_cache(self, batch_size: int = 1) -> TTTCache:
        """Create a fresh cache for generation."""
        return TTTCache(
            batch_size=batch_size,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            W1_init=self.W1,
            b1_init=self.b1,
        )

    def _split_projections(self, hidden_states: mx.array):
        """Project and split into XQ, XK, XV, XGate, ttt_lr."""
        c = self.config
        D = c.hidden_size

        proj = self.qkv_gate_lr_proj(hidden_states)  # [B, N, 3*D + nh]

        # Split: Q(D) | K(D) | V(D) | lr_logits(nh)
        xq = proj[..., :D]
        xk = proj[..., D:2 * D]
        xv_raw = proj[..., 2 * D:3 * D]
        lr_logits = proj[..., 3 * D:]  # [B, N, nh]

        # Compute per-head learning rate
        ttt_lr = (
            c.ttt_base_lr
            * mx.sigmoid(lr_logits + self.lr_bias)
            / c.head_dim
        )  # [B, N, nh]

        return xq, xk, xv_raw, ttt_lr

    def _reshape_heads(self, x: mx.array) -> mx.array:
        """[B, N, D] -> [B*nh, N, f]"""
        B, N, _ = x.shape
        c = self.config
        return x.reshape(B, N, c.num_heads, c.head_dim).transpose(0, 2, 1, 3).reshape(
            B * c.num_heads, N, c.head_dim
        )

    def _unreshape_heads(self, x: mx.array, B: int) -> mx.array:
        """[B*nh, N, f] -> [B, N, D]"""
        c = self.config
        N = x.shape[1]
        return x.reshape(B, c.num_heads, N, c.head_dim).transpose(0, 2, 1, 3).reshape(
            B, N, c.hidden_size
        )

    def _decode_step(
        self,
        xq: mx.array,
        xk: mx.array,
        xv: mx.array,
        ttt_lr: mx.array,
        cache: TTTCache,
    ) -> mx.array:
        """Single-token decode (primal form).

        All inputs: [B*nh, 1, f] except ttt_lr: [B*nh, 1, 1]
        """
        c = self.config
        eps = c.eps

        # Token position within current mini-batch
        pos_in_mb = cache.seq_offset % c.mini_batch_size

        # Token index normalizer
        raw_idx = 1.0 / (pos_in_mb + 1)
        token_idx = mx.maximum(raw_idx + self.token_idx_bias[pos_in_mb], mx.array(0.0))

        W1 = cache.W1  # [B*nh, f, f]
        b1 = cache.b1  # [B*nh, 1, f]

        # --- Step 1: Forward pass of inner model on XK ---
        # xk: [B*nh, 1, f], W1: [B*nh, f, f] -> Z1: [B*nh, 1, f]
        Z1 = xk @ W1 + b1

        # --- Step 2: Reconstruction target ---
        l2_target = xv - xk  # [B*nh, 1, f]

        # --- Steps 3-4: LayerNorm forward ---
        # Per-head norm weights need broadcasting
        ln_w = self.ttt_norm_weight  # [nh, 1, f]
        ln_b = self.ttt_norm_bias    # [nh, 1, f]
        # Tile across batch: [B*nh, 1, f]
        B_nh = Z1.shape[0]
        nh = c.num_heads
        B = B_nh // nh
        ln_w = mx.broadcast_to(
            mx.tile(ln_w, (B, 1, 1)), Z1.shape
        )
        ln_b = mx.broadcast_to(
            mx.tile(ln_b, (B, 1, 1)), Z1.shape
        )

        LN_out, Z1_hat, std = _layernorm_fwd(Z1, ln_w, ln_b, eps)

        # --- Step 5: L2 loss gradient ---
        dl_dLN = LN_out - l2_target  # [B*nh, 1, f]

        # --- Step 6: Backward through LayerNorm ---
        dl_dZ1 = _layernorm_bwd(dl_dLN, Z1_hat, std, ln_w)

        # --- Step 7: Scale by learning rate ---
        scaled_grad = ttt_lr * dl_dZ1  # [B*nh, 1, f]

        # --- Step 8: Accumulate gradients ---
        # W1_grad += XK^T @ scaled_grad  =>  [B*nh, f, 1] @ [B*nh, 1, f] = [B*nh, f, f]
        cache.W1_grad = cache.W1_grad + mx.transpose(xk, (0, 2, 1)) @ scaled_grad
        cache.b1_grad = cache.b1_grad + scaled_grad

        # --- Step 9: Apply gradient update ---
        W1_bar = W1 - token_idx * cache.W1_grad  # [B*nh, f, f]
        b1_bar = b1 - token_idx * cache.b1_grad  # [B*nh, 1, f]

        # --- Step 10: Forward with updated weights on XQ ---
        Z1_bar = xq @ W1_bar + b1_bar  # [B*nh, 1, f]

        # --- Step 11: LayerNorm on output ---
        LN_bar, _, _ = _layernorm_fwd(Z1_bar, ln_w, ln_b, eps)

        # --- Step 12: Residual ---
        output = xq + LN_bar  # [B*nh, 1, f]

        # --- State update at mini-batch boundary ---
        if (cache.seq_offset + 1) % c.mini_batch_size == 0:
            cache.W1 = W1_bar
            cache.b1 = b1_bar
            cache.W1_grad = mx.zeros_like(cache.W1_grad)
            cache.b1_grad = mx.zeros_like(cache.b1_grad)

        cache.seq_offset += 1

        return cast(mx.array, output)

    def _prefill_minibatch(
        self,
        xq: mx.array,
        xk: mx.array,
        xv: mx.array,
        ttt_lr: mx.array,
        W1: mx.array,
        b1: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Process one mini-batch with the dual form.

        Inputs:
            xq, xk, xv: [B*nh, K, f]
            ttt_lr: [B*nh, K, 1]
            W1: [B*nh, f, f]
            b1: [B*nh, 1, f]

        Returns:
            (output, W1_new, b1_new)
        """
        c = self.config
        K = xq.shape[1]
        eps = c.eps

        # Token indices
        raw_idx = 1.0 / mx.arange(1, K + 1, dtype=mx.float32)
        tok_idx = mx.maximum(raw_idx + self.token_idx_bias[:K], mx.array(0.0))
        # Build the lower-triangular eta matrix: [K, K]
        # eta[i, j] = tok_idx[i] * ttt_lr[i] for j <= i, else 0
        # Simplified: eta = tril(outer(tok_idx, ones) * ttt_lr_broadcast)
        eta_diag = tok_idx[None, :, None] * ttt_lr  # [B*nh, K, 1]

        # --- Forward pass of current inner model ---
        Z1 = xk @ W1 + b1  # [B*nh, K, f]
        l2_target = xv - xk  # [B*nh, K, f]

        # --- Fused LN forward + L2 grad + LN backward ---
        B_nh = Z1.shape[0]
        nh = c.num_heads
        B = B_nh // nh
        ln_w = mx.broadcast_to(
            mx.tile(self.ttt_norm_weight, (B, 1, 1)),
            (B_nh, 1, c.head_dim),
        )
        ln_b = mx.broadcast_to(
            mx.tile(self.ttt_norm_bias, (B, 1, 1)),
            (B_nh, 1, c.head_dim),
        )
        # Expand to match K
        ln_w_k = mx.broadcast_to(ln_w, Z1.shape)
        ln_b_k = mx.broadcast_to(ln_b, Z1.shape)

        LN_out, Z1_hat, std = _layernorm_fwd(Z1, ln_w_k, ln_b_k, eps)
        dl_dLN = LN_out - l2_target
        dl_dZ1 = _layernorm_bwd(dl_dLN, Z1_hat, std, ln_w_k)  # [B*nh, K, f]

        # --- Dual form ---
        # Causal attention-like matrix
        Attn1 = xq @ mx.transpose(xk, (0, 2, 1))  # [B*nh, K, K]
        # Apply causal mask
        mask = mx.tril(mx.ones((K, K)), 0)
        Attn1 = Attn1 * mask

        # Simplified eta scaling for bias update
        eta_scaled = eta_diag  # [B*nh, K, 1]

        # Bias: cumulative gradient-weighted sum
        b1_bar_all = b1 - mx.cumsum(eta_scaled * dl_dZ1, axis=1)  # [B*nh, K, f]

        # Output: Z1_bar = XQ @ W1 - (eta * Attn1) @ dl_dZ1 + b1_bar
        eta_attn = (eta_diag * Attn1)  # [B*nh, K, K]
        Z1_bar = xq @ W1 - eta_attn @ dl_dZ1 + b1_bar_all  # [B*nh, K, f]

        # State update for next mini-batch
        W1_new = W1 - (eta_diag[:, -1:, :] * mx.transpose(xk, (0, 2, 1))) @ dl_dZ1
        b1_new = b1_bar_all[:, -1:, :]

        # Post-LayerNorm + residual
        LN_bar, _, _ = _layernorm_fwd(Z1_bar, ln_w_k, ln_b_k, eps)
        output = xq + LN_bar

        return output, W1_new, b1_new

    def __call__(
        self,
        hidden_states: mx.array,
        cache: TTTCache | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            hidden_states: [B, N, D]
            cache: If provided, uses decode path. Otherwise prefill.

        Returns:
            Output tensor [B, N, D].
        """
        c = self.config
        B, N, _D = hidden_states.shape

        # --- Projections ---
        xq, xk, xv, ttt_lr = self._split_projections(hidden_states)

        # Reshape to per-head: [B*nh, N, f]
        xq = self._reshape_heads(xq)
        xk = self._reshape_heads(xk)
        xv = self._reshape_heads(xv)
        # ttt_lr: [B, N, nh] -> [B*nh, N, 1]
        ttt_lr = ttt_lr.transpose(0, 2, 1).reshape(B * c.num_heads, N, 1)

        if cache is not None and N == 1:
            # --- Decode path ---
            output = self._decode_step(xq, xk, xv, ttt_lr, cache)
        else:
            # --- Prefill path ---
            if cache is None:
                # Tile learned init across batch: [nh, f, f] -> [B*nh, f, f]
                W1 = mx.tile(
                    self.W1, (B, 1, 1)
                ).astype(mx.float32)
                b1 = mx.tile(
                    self.b1, (B, 1, 1)
                ).astype(mx.float32)
            else:
                W1 = cache.W1
                b1 = cache.b1

            # Process in mini-batches
            K = c.mini_batch_size
            outputs = []
            for start in range(0, N, K):
                end = min(start + K, N)
                out_mb, W1, b1 = self._prefill_minibatch(
                    xq[:, start:end],
                    xk[:, start:end],
                    xv[:, start:end],
                    ttt_lr[:, start:end],
                    W1, b1,
                )
                outputs.append(out_mb)

            output = mx.concatenate(outputs, axis=1)

            if cache is not None:
                cache.W1 = W1
                cache.b1 = b1
                cache.seq_offset += N

        # Unreshape: [B*nh, N, f] -> [B, N, D]
        output = self._unreshape_heads(output, B)

        # Post-norm
        output = self.post_norm(output)

        # Output projection
        output = self.o_proj(output)

        return output
