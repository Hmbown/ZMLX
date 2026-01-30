from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .kernels.attention import paged_attention as _paged_attention_kernel
from .kernels.moe import moe_combine, moe_dispatch


class PagedAttention(nn.Module):
    """Paged Attention layer for high-throughput serving.
    
    This layer uses the ZMLX paged attention kernel to perform attention
    over non-contiguous KV cache blocks.
    """
    def __init__(self, n_heads: int, n_kv_heads: int, head_dim: int, scale: float | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = scale or (head_dim ** -0.5)

    def __call__(
        self,
        q: Any,
        k_cache: Any,
        v_cache: Any,
        block_table: Any,
        context_lens: Any,
    ) -> Any:
        return _paged_attention_kernel(
            q, k_cache, v_cache, block_table, context_lens, scale=self.scale
        )

class MoE(nn.Module):
    """Mixture of Experts layer using ZMLX fused kernels.
    
    Fuses the dispatch and combine steps to reduce memory bandwidth usage.
    """
    def __init__(self, gate: nn.Module, experts: list[nn.Module]):
        super().__init__()
        self.gate = gate
        self.experts = experts

    def __call__(self, x: Any) -> Any:
        # 1. Gating
        logits = self.gate(x)
        from .kernels.moe import top2_gating_softmax
        weights, indices = top2_gating_softmax(logits)
        
        # 2. Fused Dispatch
        dispatched = moe_dispatch(x, indices)
        
        # 3. Expert execution
        # Note: We still loop over the experts for now, but we use the 
        # dispatched tensors which are structured for the experts.
        # Ideally, we'd have a fused expert kernel for even more gain.
        expert_outputs = []
        for i in range(indices.shape[-1]):
            # This is still a bit slow in Python; a true batch expert 
            # kernel would be better.
            expert_outputs.append(self.experts[i](dispatched[:, i]))
            
        expert_outputs = mx.stack(expert_outputs, axis=1)
        
        # 4. Fused Combine
        return moe_combine(expert_outputs, weights)
