import time

import mlx.core as mx
import numpy as np

from zmlx.kernels import indexing, norms, reductions, scan, softmax, transformer


def benchmark(name, fn, inputs, iters=100):
    # warmup
    for _ in range(10):
        mx.eval(fn(*inputs))
    
    start = time.perf_counter_ns()
    for _ in range(iters):
        mx.eval(fn(*inputs))
    end = time.perf_counter_ns()
    
    avg_ms = (end - start) / iters / 1e6
    print(f"{name:.<30} {avg_ms:.4f} ms")
    return avg_ms

if __name__ == "__main__":
    B, S, D = 16, 1024, 1024
    print(f"Benchmarking shapes: B={B}, S={S}, D={D}")
    
    x = mx.random.normal((B, S, D)).astype(mx.float16)
    w = mx.ones((D,), dtype=mx.float16)
    
    # Softmax
    benchmark("MLX Softmax", lambda x: mx.softmax(x, axis=-1), [x])
    benchmark("ZMLX Softmax", lambda x: softmax.softmax_lastdim(x), [x])
    
    # RMSNorm
    # Note: MLX has mx.fast.rms_norm
    benchmark("MLX Fast RMSNorm", lambda x, w: mx.fast.rms_norm(x, w, 1e-6), [x, w])
    benchmark("ZMLX RMSNorm", lambda x, w: norms.rmsnorm(x, w, eps=1e-6), [x, w])
    
    # SwiGLU
    # MLX doesn't have a single SwiGLU op, typically done as silu(x1) * x2
    def mlx_swiglu(z):
        a, b = mx.split(z, 2, axis=-1)
        return (a * mx.sigmoid(a)) * b
    
    z = mx.random.normal((B, S, 2 * D)).astype(mx.float16)
    benchmark("MLX SwiGLU (split+mul)", mlx_swiglu, [z])
    benchmark("ZMLX SwiGLU (fused)", transformer.swiglu, [z])
    
    # Reductions
    benchmark("MLX Sum", lambda x: x.sum(axis=-1), [x])
    benchmark("ZMLX Sum", reductions.sum_lastdim, [x])
    
    # Dropout
    benchmark("MLX Dropout (mx.random)", lambda x: x * (mx.random.uniform(shape=x.shape) > 0.5) * 2.0, [x])
    benchmark("ZMLX Dropout (fused RNG)", lambda x: transformer.dropout(x, 0.5), [x])
    
    # Top-K
    benchmark("MLX Top-K (k=5)", lambda x: mx.topk(x, k=5, axis=-1), [x])
    benchmark("ZMLX Top-K (k=5)", lambda x: reductions.topk_lastdim(x, k=5), [x])
    
    # CumSum
    benchmark("MLX CumSum", lambda x: mx.cumsum(x, axis=-1), [x])
    benchmark("ZMLX CumSum", lambda x: scan.cumsum_lastdim(x), [x])
    
    # Gather-Add
    indices = mx.array(np.random.randint(0, B*S, (B*S,)), dtype=mx.uint32)
    src = mx.random.normal((B*S, D)).astype(mx.float16)
    other = mx.random.normal((B*S, D)).astype(mx.float16)
    from zmlx.kernels import indexing
    
    def mlx_gather_add(s, idx, o):
        return s[idx] + o
        
    benchmark("MLX Gather + Add", mlx_gather_add, [src, indices, other])
    benchmark("ZMLX Fused Gather-Add", indexing.fused_gather_add, [src, indices, other])
