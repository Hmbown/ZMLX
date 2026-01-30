import time

import mlx.core as mx

import zmlx
from zmlx.kernels import quant, transformer


def benchmark(name, fn, inputs, iters=50):
    # warmup
    for _ in range(10):
        mx.eval(fn(*inputs))
    
    mx.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iters):
        mx.eval(fn(*inputs))
    mx.synchronize()
    end = time.perf_counter_ns()
    
    avg_ms = (end - start) / iters / 1e6
    print(f"{name:.<40} {avg_ms:.4f} ms")
    return avg_ms

if __name__ == "__main__":
    B, S, D = 8, 1024, 4096
    print(f"Intelligence Bench: B={B}, S={S}, D={D}")
    
    x = mx.random.normal((B, S, D)).astype(mx.float16)
    res = mx.random.normal((B, S, D)).astype(mx.float16)
    w = mx.ones((D,), dtype=mx.float16)
    
    # 1. Fused Residual + RMSNorm
    def mlx_residual_rmsnorm(x, res, w):
        return mx.fast.rms_norm(x + res, w, 1e-6)
        
    benchmark("MLX Residual + RMSNorm (2 ops)", mlx_residual_rmsnorm, [x, res, w])
    benchmark("ZMLX Fused Residual-RMSNorm (1 kernel)", transformer.fused_add_rmsnorm, [x, res, w])

    # 2. Autotuned MapReduce (Softmax)
    my_softmax_fixed = zmlx.map_reduce(
        pass1={"init": "-INFINITY", "update": "max(acc1, x)", "reduce": "max(a, b)"},
        pass2={"init": "0.0f", "update": "acc2 + exp(x - s1)", "reduce": "a + b"},
        write="exp(x - s1) / s2",
        name="bench_softmax_fixed",
        threadgroup=256
    )
    
    my_softmax_auto = zmlx.map_reduce(
        pass1={"init": "-INFINITY", "update": "max(acc1, x)", "reduce": "max(a, b)"},
        pass2={"init": "0.0f", "update": "acc2 + exp(x - s1)", "reduce": "a + b"},
        write="exp(x - s1) / s2",
        name="bench_softmax_auto",
        threadgroup="auto"
    )

    # Warmup and tune
    mx.eval(my_softmax_auto(x))
    mx.synchronize()
    
    benchmark("ZMLX Softmax (Fixed TG=256)", my_softmax_fixed, [x])
    benchmark("ZMLX Softmax (Autotuned - Cached)", my_softmax_auto, [x])

    # 3. Enhanced JIT with Vectorization
    @zmlx.jit
    def math_op(a, b):
        c = a * b
        d = mx.sigmoid(c)
        return d + a
        
    def mlx_math_op(a, b):
        c = a * b
        d = mx.sigmoid(c)
        return d + a

    benchmark("MLX Scalar Ops (multi-kernel)", mlx_math_op, [x, res])
    benchmark("ZMLX JIT (Fused + Vectorized)", math_op, [x, res])

    # 4. Block-wise Quantization Fusions
    # Mock quantized data
    q_x = mx.random.randint(0, 255, (B, S, D), dtype=mx.uint8)
    scales = mx.random.normal((B, S, D // 128)).astype(mx.float16)
    
    # MLX doesn't have a direct equivalent for block-wise dequant + swiglu in one kernel
    # Typically one would dequant then apply swiglu
    def mlx_quant_swiglu(q1, s1, q2, s2):
        # Very rough approximation of MLX overhead
        x1 = q1.astype(mx.float32) * mx.repeat(s1, 128, axis=-1)
        x2 = q2.astype(mx.float32) * mx.repeat(s2, 128, axis=-1)
        return (x1 * mx.sigmoid(x1)) * x2

    benchmark("MLX Dequant + SwiGLU (simulated)", mlx_quant_swiglu, [q_x, scales, q_x, scales])
    benchmark("ZMLX Fused Quant-SwiGLU", quant.fused_swiglu_quant, [q_x, scales, q_x, scales])
