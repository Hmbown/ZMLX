import unittest

import mlx.core as mx

from zmlx.jit_compiler import jit
from zmlx.kernels.attention import paged_attention
from zmlx.kernels.moe import moe_combine, moe_dispatch
from zmlx.kernels.rope import rope_and_cache_update
from zmlx.optimizers import AdamW


class TestNewFeatures(unittest.TestCase):
    def test_fused_adamw(self):
        p = mx.array([1.0, 2.0], dtype=mx.float32)
        g = mx.array([0.1, 0.2], dtype=mx.float32)
        state = {"step": 1, "m": mx.zeros_like(p), "v": mx.zeros_like(p)}
        
        opt = AdamW(learning_rate=0.1)
        # We need to set the state manually since we are testing internal _update_single
        new_p = opt._update_single(p, g, state)
        
        self.assertEqual(new_p.shape, p.shape)
        self.assertTrue("m" in state)
        self.assertTrue("v" in state)

    def test_paged_attention(self):
        B, H, D = 2, 8, 64
        HKV = 4
        BS = 16

        q = mx.random.normal((B, H, D))
        k_cache = mx.random.normal((20, BS, HKV, D))
        v_cache = mx.random.normal((20, BS, HKV, D))
        block_table = mx.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]], dtype=mx.uint32)
        context_lens = mx.array([32, 64], dtype=mx.int32)
        
        out = paged_attention(q, k_cache, v_cache, block_table, context_lens)
        self.assertEqual(out.shape, (B, H, D))

    def test_rope_cache_update(self):
        B, H, D = 2, 8, 64
        HKV = 4
        MAX_SEQ = 128
        
        q = mx.random.normal((B, H, D))
        k = mx.random.normal((B, HKV, D))
        v = mx.random.normal((B, HKV, D))
        cos = mx.random.normal((MAX_SEQ, D // 2))
        sin = mx.random.normal((MAX_SEQ, D // 2))
        k_cache = mx.zeros((B, MAX_SEQ, HKV, D))
        v_cache = mx.zeros((B, MAX_SEQ, HKV, D))
        offset = mx.array([10, 20], dtype=mx.int32)
        
        new_q, new_k_cache, new_v_cache = rope_and_cache_update(q, k, v, cos, sin, k_cache, v_cache, offset)
        
        self.assertEqual(new_q.shape, q.shape)
        self.assertEqual(new_k_cache.shape, k_cache.shape)
        # Check if cache was updated at offset
        # B=0, offset=10
        # new_k_cache[0, 10] should be non-zero
        self.assertFalse(mx.all(new_k_cache[0, 10] == 0))

    def test_moe_kernels(self):
        B, D, K = 4, 128, 2
        x = mx.random.normal((B, D))
        indices = mx.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=mx.uint32)
        weights = mx.array([[0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]])
        
        dispatched = moe_dispatch(x, indices)
        self.assertEqual(dispatched.shape, (B, K, D))
        
        expert_outputs = mx.random.normal((B, K, D))
        combined = moe_combine(expert_outputs, weights)
        self.assertEqual(combined.shape, (B, D))

    def test_jit(self):
        @jit
        def my_op(x, y):
            return x * mx.sigmoid(x) + y
            
        a = mx.array([1.0, 2.0], dtype=mx.float32)
        b = mx.array([0.5, 0.5], dtype=mx.float32)
        
        res = my_op(a, b)
        expected = a * mx.sigmoid(a) + b
        
        self.assertTrue(mx.allclose(res, expected))

    def test_paged_rope_cache_update(self):
        from zmlx.kernels.rope import paged_rope_and_cache_update
        B, H, D = 2, 8, 64
        HKV = 4
        BS = 16
        N_BLOCKS = 20
        MAX_BLOCKS = 10
        MAX_SEQ = MAX_BLOCKS * BS
        
        q = mx.random.normal((B, H, D))
        k = mx.random.normal((B, HKV, D))
        v = mx.random.normal((B, HKV, D))
        cos = mx.random.normal((MAX_SEQ, D // 2))
        sin = mx.random.normal((MAX_SEQ, D // 2))
        k_cache = mx.zeros((N_BLOCKS, BS, HKV, D))
        v_cache = mx.zeros((N_BLOCKS, BS, HKV, D))
        offset = mx.array([10, 20], dtype=mx.int32)
        block_table = mx.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
                                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]], dtype=mx.uint32)
        
        new_q, new_k_cache, new_v_cache = paged_rope_and_cache_update(
            q, k, v, cos, sin, k_cache, v_cache, offset, block_table
        )
        
        self.assertEqual(new_q.shape, q.shape)
        self.assertEqual(new_k_cache.shape, k_cache.shape)
        # B=0, offset=10 -> block_logical_idx=0, token_block_idx=10, physical_block=0
        self.assertFalse(mx.all(new_k_cache[0, 10] == 0))
        # B=1, offset=20 -> block_logical_idx=1, token_block_idx=4, physical_block=11
        self.assertFalse(mx.all(new_k_cache[11, 4] == 0))

    def test_nn_layers(self):
        import mlx.nn as nn

        from zmlx.nn import MoE, PagedAttention
        
        # PagedAttention
        pa = PagedAttention(n_heads=8, n_kv_heads=4, head_dim=64)
        B, H, D = 2, 8, 64
        q = mx.random.normal((B, H, D))
        k_cache = mx.random.normal((20, 16, 4, 64))
        v_cache = mx.random.normal((20, 16, 4, 64))
        block_table = mx.zeros((2, 10), dtype=mx.uint32)
        context_lens = mx.array([16, 16], dtype=mx.int32)
        out = pa(q, k_cache, v_cache, block_table, context_lens)
        self.assertEqual(out.shape, (B, H, D))
        
        # MoE
        gate = nn.Linear(128, 4)
        experts = [nn.Linear(128, 128) for _ in range(4)]
        moe_layer = MoE(gate, experts)
        x = mx.random.normal((4, 128))
        out = moe_layer(x)
        self.assertEqual(out.shape, (4, 128))

if __name__ == "__main__":
    unittest.main()
