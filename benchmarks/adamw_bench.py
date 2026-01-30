import time

import mlx.core as mx
import mlx.optimizers as optim

from zmlx.optimizers import AdamW as FusedAdamW


def benchmark_adamw(n_params=100, size_per_param=1024*1024, iters=50):
    print(f"Benchmarking AdamW update on {n_params} parameters of size {size_per_param} each...")
    
    # Initialize parameters and gradients as a dict
    params_orig = {f"p{i}": mx.random.normal((size_per_param,), dtype=mx.float16) for i in range(n_params)}
    grads = {f"p{i}": mx.random.normal((size_per_param,), dtype=mx.float16) for i in range(n_params)}
    
    # MLX Standard AdamW
    opt_std = optim.AdamW(learning_rate=0.1)
    params_std = params_orig.copy()
    # Warmup
    params_std = opt_std.apply_gradients(grads, params_std)
    mx.eval(params_std)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        params_std = opt_std.apply_gradients(grads, params_std)
    mx.eval(params_std)
    t1 = time.perf_counter()
    std_time = (t1 - t0) / iters * 1000
    print(f"Standard MLX AdamW: {std_time:.3f} ms/step")
    
    # ZMLX Fused AdamW
    opt_fused = FusedAdamW(learning_rate=0.1)
    params_fused = params_orig.copy()
    # Warmup
    params_fused = opt_fused.apply_gradients(grads, params_fused)
    mx.eval(params_fused)
    
    t0 = time.perf_counter()
    for _ in range(iters):
        params_fused = opt_fused.apply_gradients(grads, params_fused)
    mx.eval(params_fused)
    t1 = time.perf_counter()

    fused_time = (t1 - t0) / iters * 1000
    print(f"ZMLX Fused AdamW:    {fused_time:.3f} ms/step")
    
    speedup = std_time / fused_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_adamw()
