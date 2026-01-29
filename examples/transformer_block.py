import mlx.core as mx

from zmlx.kernels import attention, rope, transformer


def zmlx_transformer_fragment(x, residual, weight, cos, sin, mask):
    # 1. Fused RMSNorm(x + residual)
    # Returns the normalized x and the updated residual (x + residual)
    normed_x, updated_res = transformer.rmsnorm_residual(x, residual, weight)
    
    # 2. RoPE (assuming x is query/key sequence)
    # Applying RoPE to the normed output
    # Note: real transformer would project first, but here we demo the kernel
    q = rope.apply_rope(normed_x, cos, sin)
    
    # 3. Scale + Mask + Softmax (Attention scores)
    # Simulate an attention score matrix (B, H, S, S)
    # We'll just do it on a single row for simplicity in demo
    scores = attention.scale_mask_softmax(q, mask, scale=0.125)
    
    return scores, updated_res

if __name__ == "__main__":
    B, S, D = 1, 128, 64
    x = mx.random.normal((B, S, D)).astype(mx.float16)
    res = mx.random.normal((B, S, D)).astype(mx.float16)
    w = mx.ones((D,), dtype=mx.float16)
    
    cos = mx.random.normal((S, D // 2)).astype(mx.float16)
    sin = mx.random.normal((S, D // 2)).astype(mx.float16)
    # Note: mask shape matches the scores tensor (B, S, D here). In a real
    # transformer the scores would be (B, H, S, S) and the mask (1, 1, S, S).
    mask = mx.ones((B, S, D), dtype=mx.bool_)
    
    scores, next_res = zmlx_transformer_fragment(x, res, w, cos, sin, mask)
    
    mx.eval(scores, next_res)
    print("Transformer fragment executed successfully.")
    print(f"Scores shape: {scores.shape}")
    print(f"Residual shape: {next_res.shape}")
